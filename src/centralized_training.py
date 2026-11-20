"""Centralized baseline: pool all RSU data and train a single model.

This is the upper-bound reference — it uses all trajectory data from all
three RSUs in one training set, which requires raw data to leave the RSU
boundary.  This is explicitly labelled as **privacy-unsafe** and is
provided only for comparison.

Usage (from repository root)::

    python -m src.centralized_training \
        --data-dir data/rsu \
        --epochs 30 \
        --batch-size 256 \
        --checkpoint-path models/centralized_global.pt \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
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
    create_dataloader,
    evaluate,
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
    """Compute normalization stats from all RSU training partitions."""
    all_X, all_y = [], []
    for rsu_id in RSU_IDS:
        X, y = load_rsu_dataset(rsu_id, split="train", data_directory=data_dir)
        all_X.append(np.array(X))
        all_y.append(np.array(y))
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    return NormalizationStats.from_training_data(X_all, y_all)


def run_centralized_training(
    data_dir: str = "data/rsu",
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "models/centralized_global.pt",
    model_config: Optional[dict] = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train a single centralized model on pooled data from all RSUs.

    Returns
    -------
    dict
        Training history and checkpoint path.

    .. warning::
        This baseline is **privacy-unsafe**: it requires all raw trajectory
        data to be pooled at a central server.  It is provided as an upper
        bound reference only.
    """
    if model_config is None:
        model_config = dict(DEFAULT_MODEL_CONFIG)

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("CENTRALIZED BASELINE — pooled data (PRIVACY-UNSAFE)")
    logger.info("=" * 60)
    logger.info("  epochs        = %d", epochs)
    logger.info("  batch_size    = %d", batch_size)
    logger.info("  learning_rate = %g", learning_rate)
    logger.info("  seed          = %d", seed)
    logger.info("  data_dir      = %s", data_dir)
    logger.info("  checkpoint    = %s", checkpoint_path)

    # Compute normalization stats
    norm_stats = compute_global_norm_stats(data_dir)

    # Pool all RSU data
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    for rsu_id in RSU_IDS:
        X_tr, y_tr = load_rsu_dataset(rsu_id, "train", data_dir)
        X_va, y_va = load_rsu_dataset(rsu_id, "validation", data_dir)
        all_X_train.append(np.array(X_tr))
        all_y_train.append(np.array(y_tr))
        all_X_val.append(np.array(X_va))
        all_y_val.append(np.array(y_va))
        logger.info(
            "  %s: %d train, %d val samples loaded",
            rsu_id, len(X_tr), len(X_va),
        )

    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_val = np.concatenate(all_y_val, axis=0)

    logger.info(
        "Pooled data: %d train, %d val samples total",
        len(X_train), len(X_val),
    )

    # Normalize
    X_train_norm = norm_stats.normalize_input(X_train)
    y_train_norm = norm_stats.normalize_target(y_train)
    X_val_norm = norm_stats.normalize_input(X_val)
    y_val_norm = norm_stats.normalize_target(y_val)

    train_loader = create_dataloader(
        X_train_norm, y_train_norm, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        X_val_norm, y_val_norm, batch_size=batch_size, shuffle=False
    )

    # Create model
    model = VehicleStateGRU.from_config(model_config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            logger.info(
                "Centralized  epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch, epochs, train_loss, val_loss,
            )

    elapsed = time.time() - start_time
    logger.info("Centralized training finished in %.1f s", elapsed)

    # Save checkpoint
    metadata = {
        "training_mode": "centralized",
        "privacy_note": "PRIVACY-UNSAFE: all raw data pooled at central server",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "total_train_samples": len(X_train),
        "total_val_samples": len(X_val),
        "elapsed_seconds": round(elapsed, 1),
    }
    saved_path = save_checkpoint(model, norm_stats, checkpoint_path, metadata)
    logger.info("Centralized checkpoint saved to %s", saved_path)

    # Save metrics
    metrics_path = Path(checkpoint_path).parent / "centralized_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "checkpoint_path": str(saved_path),
        "elapsed_seconds": round(elapsed, 1),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "total_train_samples": len(X_train),
        "total_val_samples": len(X_val),
    }
    metrics_path.write_text(
        json.dumps(metrics_data, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "checkpoint_path": str(saved_path),
        "history": history,
        "elapsed_seconds": round(elapsed, 1),
        "total_train_samples": len(X_train),
        "total_val_samples": len(X_val),
    }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centralized baseline: pool all RSU data (PRIVACY-UNSAFE)",
    )
    parser.add_argument("--data-dir", type=str, default="data/rsu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint-path", type=str, default="models/centralized_global.pt"
    )
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
    run_centralized_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        seed=args.seed,
        device="cpu",
    )


if __name__ == "__main__":
    main()

