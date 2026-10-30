"""Federated training orchestration using Flower simulation.

Runs three RSU clients and a FedAvg server **in-process** via
``flwr.simulation.start_simulation``.  No network sockets are opened;
training is purely CPU-based.

Usage (from repository root)
----------------------------
::

    python -m src.federated_training \\
        --data-dir data/rsu \\
        --rounds 5 \\
        --local-epochs 1 \\
        --batch-size 256 \\
        --checkpoint-path models/fedavg_global.pt \\
        --seed 42

All paths, hyper-parameters, and the random seed are configurable via CLI
flags.  The script uses Person 2's checkpoint format so that
``RSUStatePredictor.from_checkpoint(...)`` works directly on the saved file.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.model import VehicleStateGRU
from src.train_utils import (
    NormalizationStats,
    save_checkpoint,
)
from src.preprocessing import load_rsu_dataset
from src.base_station import create_fedavg_strategy
from src.rsu_client import RSUFlowerClient, make_rsu_client

logger = logging.getLogger(__name__)

# The three logical RSU identifiers agreed with Person 1
RSU_IDS = ("rsu_1", "rsu_2", "rsu_3")

# Default model configuration matching Person 2's defaults
DEFAULT_MODEL_CONFIG = {
    "input_size": 4,
    "hidden_size": 64,
    "prediction_steps": 3,
    "num_layers": 1,
}


def compute_global_norm_stats(
    data_dir: str = "data/rsu",
) -> NormalizationStats:
    """Compute normalization statistics from **all** RSU training partitions.

    All RSU clients must normalise with the same statistics for consistent
    loss comparisons and for the final checkpoint to be usable by the
    ``RSUStatePredictor``.

    Returns
    -------
    NormalizationStats
        Statistics computed from the union of all RSU training sets.
    """
    all_X = []
    all_y = []
    for rsu_id in RSU_IDS:
        X, y = load_rsu_dataset(rsu_id, split="train", data_directory=data_dir)
        all_X.append(np.array(X))   # copy from mmap
        all_y.append(np.array(y))
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    stats = NormalizationStats.from_training_data(X_all, y_all)
    logger.info(
        "Global norm stats computed from %d total training samples", len(X_all)
    )
    return stats


def _build_client_fn(
    model_config: dict,
    norm_stats: NormalizationStats,
    data_dir: str,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
):
    """Return a closure that Flower simulation calls to create each client.

    Flower's simulation backend calls ``client_fn(cid)`` where ``cid`` is a
    string like ``"0"``, ``"1"``, ``"2"``.  We map these to RSU IDs.
    """

    def client_fn(cid: str) -> fl.client.NumPyClient:
        idx = int(cid)
        if idx < 0 or idx >= len(RSU_IDS):
            raise ValueError(
                f"Client ID {cid} out of range; expected 0..{len(RSU_IDS)-1}"
            )
        rsu_id = RSU_IDS[idx]
        logger.info("Creating Flower client for %s (cid=%s)", rsu_id, cid)
        return make_rsu_client(
            rsu_id=rsu_id,
            model_config=model_config,
            norm_stats=norm_stats,
            data_dir=data_dir,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + idx,   # slightly different seed per client
            device=device,
        )

    return client_fn


def run_federated_training(
    data_dir: str = "data/rsu",
    rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "models/fedavg_global.pt",
    model_config: Optional[dict] = None,
    seed: int = 42,
    device: str = "cpu",
) -> Path:
    """Run federated training and save the final global checkpoint.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``rsu_1/``, ``rsu_2/``, ``rsu_3/`` data.
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
        Model hyper-parameters; defaults to ``DEFAULT_MODEL_CONFIG``.
    seed : int
        Global random seed.
    device : str
        Torch device (default ``"cpu"``).

    Returns
    -------
    Path
        Absolute path to the saved checkpoint.
    """
    if model_config is None:
        model_config = dict(DEFAULT_MODEL_CONFIG)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("FEDERATED TRAINING — FedAvg with %d RSU clients", len(RSU_IDS))
    logger.info("=" * 60)
    logger.info("  rounds        = %d", rounds)
    logger.info("  local_epochs  = %d", local_epochs)
    logger.info("  batch_size    = %d", batch_size)
    logger.info("  learning_rate = %g", learning_rate)
    logger.info("  seed          = %d", seed)
    logger.info("  data_dir      = %s", data_dir)
    logger.info("  checkpoint    = %s", checkpoint_path)
    logger.info("  model_config  = %s", model_config)
    logger.info("  device        = %s", device)

    # ---------- Normalization (all RSUs, training split only) ----------
    norm_stats = compute_global_norm_stats(data_dir=data_dir)

    # ---------- Initial global model ----------
    global_model = VehicleStateGRU.from_config(model_config)
    initial_params = [
        val.cpu().numpy() for _, val in global_model.state_dict().items()
    ]
    param_bytes = sum(p.nbytes for p in initial_params)
    logger.info(
        "Initial global model: %d parameter arrays, %d bytes total",
        len(initial_params),
        param_bytes,
    )

    # ---------- Strategy ----------
    num_clients = len(RSU_IDS)
    strategy = create_fedavg_strategy(
        initial_parameters=initial_params,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # ---------- Client factory ----------
    client_fn = _build_client_fn(
        model_config=model_config,
        norm_stats=norm_stats,
        data_dir=data_dir,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )

    # ---------- Simulation ----------
    start_time = time.time()
    logger.info("Starting Flower simulation …")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    elapsed = time.time() - start_time
    logger.info("Simulation finished in %.1f s", elapsed)

    # ---------- Log history ----------
    if history.losses_distributed:
        for rnd, loss in history.losses_distributed:
            logger.info("  Round %d  distributed val_loss = %.6f", rnd, loss)
    if history.metrics_distributed:
        for key, values in history.metrics_distributed.items():
            for rnd, val in values:
                logger.info("  Round %d  %s = %.6f", rnd, key, val)

    # ---------- Extract final parameters & save checkpoint ----------
    # The strategy keeps the latest aggregated parameters
    final_params_obj = (
        strategy.latest_parameters 
        if strategy.latest_parameters is not None 
        else strategy.initial_parameters
    )
    # After simulation, Flower's strategy stores the latest parameters
    # internally.  We use the history to verify completion, then extract
    # from the strategy.
    #
    # Flower ≥ 1.5 stores the latest parameters accessible via the
    # strategy object.  We rebuild the model from the latest params.
    # Because start_simulation returns a History and we used FedAvg, the
    # final global params were the last ones sent to clients.  We
    # re-extract them by running the client_fn trick: create a temporary
    # client, call get_parameters after the last round's set_parameters.
    #
    # A simpler approach: after simulation, directly read the last
    # aggregated parameters from the strategy's state.
    final_ndarrays = parameters_to_ndarrays(final_params_obj)

    # Load into model
    keys = list(global_model.state_dict().keys())
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(keys, final_ndarrays)}
    )
    global_model.load_state_dict(state_dict, strict=True)

    # Save using Person 2's agreed checkpoint format
    metadata = {
        "training_mode": "federated",
        "strategy": "FedAvg",
        "num_rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_clients": num_clients,
        "rsu_ids": list(RSU_IDS),
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
    }
    saved_path = save_checkpoint(
        global_model, norm_stats, checkpoint_path, metadata
    )
    logger.info("Global checkpoint saved to %s", saved_path)
    return saved_path


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated training (FedAvg) with Flower simulation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/rsu",
        help="Root directory for RSU-partitioned data (default: data/rsu)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated rounds (default: 5)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local training epochs per round (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Client optimizer learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/fedavg_global.pt",
        help="Path to save the final global checkpoint (default: models/fedavg_global.pt)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="GRU hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of GRU layers (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
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

    run_federated_training(
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
