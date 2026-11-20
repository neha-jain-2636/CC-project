"""Local-only baseline: train three independent GRU models, one per RSU.

Each RSU trains on its own partition only, with no federation or data
sharing.  This is the lower-bound baseline — it shows what each RSU can
learn from its local data alone.

Usage (from repository root)::

    python -m src.local_training \
        --data-dir data/rsu \
        --epochs 30 \
        --batch-size 256 \
        --checkpoint-dir models/local \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.model import VehicleStateGRU
from src.preprocessing import load_rsu_dataset
from src.train_utils import (
    NormalizationStats,
    compute_per_horizon_rmse,
    create_dataloader,
    evaluate,
    measure_inference_latency,
    save_checkpoint,
    train_one_epoch,
)

logger = logging.getLogger(__name__)

RSU_IDS = ("rsu_1", "rsu_2", "rsu_3")

DEFAULT_MODEL_CONFIG = {
    "input_size": 4,
    "hidden_size": 64,
    "prediction_steps": 3,
    "num_layers": 1,
}


def compute_global_norm_stats(data_dir: str = "data/rsu") -> NormalizationStats:
    """Compute normalization stats from all RSU training partitions.

    Uses the same statistics across all training modes for fair comparison.
    """
    all_X, all_y = [], []
    for rsu_id in RSU_IDS:
        X, y = load_rsu_dataset(rsu_id, split="train", data_directory=data_dir)
        all_X.append(np.array(X))
        all_y.append(np.array(y))
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    stats = NormalizationStats.from_training_data(X_all, y_all)
    logger.info("Global norm stats computed from %d total training samples", len(X_all))
    return stats


def train_local_model(
    rsu_id: str,
    norm_stats: NormalizationStats,
    data_dir: str = "data/rsu",
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    model_config: Optional[dict] = None,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[VehicleStateGRU, dict]:
    """Train a single local model for one RSU.

    Returns
    -------
    tuple
        (trained_model, training_history_dict)
    """
    if model_config is None:
        model_config = dict(DEFAULT_MODEL_CONFIG)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    X_train_raw, y_train_raw = load_rsu_dataset(rsu_id, "train", data_dir)
    X_val_raw, y_val_raw = load_rsu_dataset(rsu_id, "validation", data_dir)

    X_train = norm_stats.normalize_input(np.array(X_train_raw))
    y_train = norm_stats.normalize_target(np.array(y_train_raw))
    X_val = norm_stats.normalize_input(np.array(X_val_raw))
    y_val = norm_stats.normalize_target(np.array(y_val_raw))

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    logger.info(
        "%s: %d train samples, %d val samples",
        rsu_id, len(X_train), len(X_val),
    )

    model = VehicleStateGRU.from_config(model_config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            logger.info(
                "%s  epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                rsu_id, epoch, epochs, train_loss, val_loss,
            )

    return model, history


def run_local_training(
    data_dir: str = "data/rsu",
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "models/local",
    model_config: Optional[dict] = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train all three local models and save checkpoints.

    Returns
    -------
    dict
        Per-RSU training history and checkpoint paths.
    """
    if model_config is None:
        model_config = dict(DEFAULT_MODEL_CONFIG)

    logger.info("=" * 60)
    logger.info("LOCAL-ONLY BASELINE — 3 independent RSU models")
    logger.info("=" * 60)
    logger.info("  epochs        = %d", epochs)
    logger.info("  batch_size    = %d", batch_size)
    logger.info("  learning_rate = %g", learning_rate)
    logger.info("  seed          = %d", seed)
    logger.info("  data_dir      = %s", data_dir)
    logger.info("  checkpoint_dir = %s", checkpoint_dir)

    norm_stats = compute_global_norm_stats(data_dir)
    results = {}

    for i, rsu_id in enumerate(RSU_IDS):
        logger.info("-" * 40)
        logger.info("Training %s ...", rsu_id)

        start_time = time.time()
        model, history = train_local_model(
            rsu_id=rsu_id,
            norm_stats=norm_stats,
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_config=model_config,
            seed=seed + i,
            device=device,
        )
        elapsed = time.time() - start_time

        # Save checkpoint
        ckpt_path = Path(checkpoint_dir) / f"local_{rsu_id}.pt"
        metadata = {
            "training_mode": "local",
            "rsu_id": rsu_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed + i,
            "elapsed_seconds": round(elapsed, 1),
        }
        saved_path = save_checkpoint(model, norm_stats, ckpt_path, metadata)
        logger.info("%s checkpoint saved to %s", rsu_id, saved_path)

        results[rsu_id] = {
            "checkpoint_path": str(saved_path),
            "history": history,
            "elapsed_seconds": round(elapsed, 1),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        }

    # Save combined metrics
    metrics_path = Path(checkpoint_dir) / "local_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for rsu_id, data in results.items():
        serializable[rsu_id] = {
            "checkpoint_path": data["checkpoint_path"],
            "elapsed_seconds": data["elapsed_seconds"],
            "final_train_loss": data["final_train_loss"],
            "final_val_loss": data["final_val_loss"],
            "train_loss_history": data["history"]["train_loss"],
            "val_loss_history": data["history"]["val_loss"],
        }
    metrics_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    logger.info("Local training metrics saved to %s", metrics_path)

    return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-only baseline: train one independent model per RSU",
    )
    parser.add_argument("--data-dir", type=str, default="data/rsu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="models/local")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    model_config = {
        "input_size": 4,
        "hidden_size": args.hidden_size,
        "prediction_steps": 3,
        "num_layers": args.num_layers,
    }
    run_local_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        model_config=model_config,
        seed=args.seed,
        device="cpu",
    )


if __name__ == "__main__":
    main()

