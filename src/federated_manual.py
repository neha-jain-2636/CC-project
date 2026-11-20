"""Manual FedAvg simulation without Flower's ray-based simulator.

This module provides an in-process FedAvg loop that replicates the same
logic as Flower's start_simulation but without requiring ray.  It directly
orchestrates the RSU clients and aggregation.

Usage (from repository root)::

    python -m src.federated_manual \
        --data-dir data/rsu \
        --rounds 10 \
        --local-epochs 1 \
        --batch-size 512 \
        --checkpoint-path models/fedavg_global.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import OrderedDict
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


def _get_model_params(model: VehicleStateGRU) -> list[np.ndarray]:
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().detach().numpy().copy() for val in model.state_dict().values()]


def _set_model_params(model: VehicleStateGRU, params: list[np.ndarray]) -> None:
    """Load parameters into a model from a list of numpy arrays."""
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(keys, params)}
    )
    model.load_state_dict(state_dict, strict=True)


def _fedavg_aggregate(
    client_params: list[list[np.ndarray]],
    client_weights: list[int],
) -> list[np.ndarray]:
    """Weighted average of client parameters (FedAvg)."""
    total_weight = sum(client_weights)
    num_layers = len(client_params[0])
    aggregated = []
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(client_params[0][layer_idx], dtype=np.float64)
        for client_idx, params in enumerate(client_params):
            weighted_sum += params[layer_idx].astype(np.float64) * client_weights[client_idx]
        aggregated.append((weighted_sum / total_weight).astype(np.float32))
    return aggregated


def run_federated_manual(
    data_dir: str = "data/rsu",
    rounds: int = 10,
    local_epochs: int = 1,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "models/fedavg_global.pt",
    model_config: Optional[dict] = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run FedAvg manually in-process without ray/Flower simulation.

    Parameters
    ----------
    data_dir : str
        Root directory containing rsu_1/, rsu_2/, rsu_3/ data.
    rounds : int
        Number of federated rounds.
    local_epochs : int
        Local training epochs per round per client.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Client-side optimizer learning rate.
    checkpoint_path : str
        Where to save the final global checkpoint.
    model_config : dict, optional
        Model hyper-parameters.
    seed : int
        Global random seed.
    device : str
        Torch device.

    Returns
    -------
    dict
        Training history including per-round metrics.
    """
    if model_config is None:
        model_config = dict(DEFAULT_MODEL_CONFIG)

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("FEDERATED TRAINING — Manual FedAvg with %d RSU clients", len(RSU_IDS))
    logger.info("=" * 60)
    logger.info("  rounds        = %d", rounds)
    logger.info("  local_epochs  = %d", local_epochs)
    logger.info("  batch_size    = %d", batch_size)
    logger.info("  learning_rate = %g", learning_rate)
    logger.info("  seed          = %d", seed)
    logger.info("  data_dir      = %s", data_dir)
    logger.info("  checkpoint    = %s", checkpoint_path)

    # --- Compute global normalization stats ---
    norm_stats = compute_global_norm_stats(data_dir)

    # --- Prepare client data ---
    client_data = {}
    for rsu_id in RSU_IDS:
        X_tr_raw, y_tr_raw = load_rsu_dataset(rsu_id, "train", data_dir)
        X_va_raw, y_va_raw = load_rsu_dataset(rsu_id, "validation", data_dir)

        X_tr = norm_stats.normalize_input(np.array(X_tr_raw))
        y_tr = norm_stats.normalize_target(np.array(y_tr_raw))
        X_va = norm_stats.normalize_input(np.array(X_va_raw))
        y_va = norm_stats.normalize_target(np.array(y_va_raw))

        train_loader = create_dataloader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(X_va, y_va, batch_size=batch_size, shuffle=False)

        client_data[rsu_id] = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "num_train": len(X_tr),
            "num_val": len(X_va),
        }
        logger.info("  %s: %d train, %d val samples", rsu_id, len(X_tr), len(X_va))

    # --- Initialize global model ---
    global_model = VehicleStateGRU.from_config(model_config)
    global_params = _get_model_params(global_model)
    param_bytes = sum(p.nbytes for p in global_params)
    logger.info("Model: %d parameter arrays, %d bytes (%.2f KB)",
                len(global_params), param_bytes, param_bytes / 1024)

    # --- FedAvg rounds ---
    history = {
        "round_train_loss": [],
        "round_val_loss": [],
        "per_client_train_loss": [],
        "per_client_val_loss": [],
        "payload_bytes_per_round": [],
    }

    start_time = time.time()
    for round_num in range(1, rounds + 1):
        round_start = time.time()
        client_updated_params = []
        client_weights = []
        client_train_losses = {}
        client_val_losses = {}

        for i, rsu_id in enumerate(RSU_IDS):
            # Create a fresh local model and load global parameters
            local_model = VehicleStateGRU.from_config(model_config)
            _set_model_params(local_model, global_params)
            local_model.to(device)

            # Local training
            optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            total_train_loss = 0.0
            for epoch in range(local_epochs):
                epoch_loss = train_one_epoch(
                    local_model, client_data[rsu_id]["train_loader"],
                    optimizer, criterion, device
                )
                total_train_loss += epoch_loss

            avg_train_loss = total_train_loss / max(local_epochs, 1)
            client_train_losses[rsu_id] = avg_train_loss

            # Validation
            val_loss = evaluate(
                local_model, client_data[rsu_id]["val_loader"],
                criterion, device
            )
            client_val_losses[rsu_id] = val_loss

            # Collect updated parameters
            updated_params = _get_model_params(local_model)
            client_updated_params.append(updated_params)
            client_weights.append(client_data[rsu_id]["num_train"])

        # Aggregate (FedAvg)
        global_params = _fedavg_aggregate(client_updated_params, client_weights)

        # Compute weighted average metrics
        total_samples = sum(client_weights)
        avg_train_loss = sum(
            client_train_losses[r] * client_data[r]["num_train"]
            for r in RSU_IDS
        ) / total_samples
        avg_val_loss = sum(
            client_val_losses[r] * client_data[r]["num_val"]
            for r in RSU_IDS
        ) / sum(client_data[r]["num_val"] for r in RSU_IDS)

        round_elapsed = time.time() - round_start
        history["round_train_loss"].append(avg_train_loss)
        history["round_val_loss"].append(avg_val_loss)
        history["per_client_train_loss"].append(client_train_losses)
        history["per_client_val_loss"].append(client_val_losses)
        history["payload_bytes_per_round"].append(param_bytes * len(RSU_IDS))

        logger.info(
            "Round %2d/%d  train_loss=%.6f  val_loss=%.6f  (%.1f s)",
            round_num, rounds, avg_train_loss, avg_val_loss, round_elapsed,
        )
        for rsu_id in RSU_IDS:
            logger.info(
                "  %s: train=%.6f  val=%.6f",
                rsu_id, client_train_losses[rsu_id], client_val_losses[rsu_id],
            )

    elapsed = time.time() - start_time
    logger.info("Federated training finished in %.1f s", elapsed)

    # --- Load final params into global model and save ---
    _set_model_params(global_model, global_params)

    metadata = {
        "training_mode": "federated",
        "strategy": "FedAvg",
        "num_rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_clients": len(RSU_IDS),
        "rsu_ids": list(RSU_IDS),
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "implementation": "manual_in_process",
    }
    saved_path = save_checkpoint(global_model, norm_stats, checkpoint_path, metadata)
    logger.info("Global checkpoint saved to %s", saved_path)

    return {
        "checkpoint_path": str(saved_path),
        "history": history,
        "elapsed_seconds": round(elapsed, 1),
    }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual FedAvg simulation (no ray dependency)",
    )
    parser.add_argument("--data-dir", type=str, default="data/rsu")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-path", type=str, default="models/fedavg_global.pt")
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
    run_federated_manual(
        data_dir=args.data_dir,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        model_config=model_config,
        seed=args.seed,
        device="cpu",
    )


if __name__ == "__main__":
    main()
