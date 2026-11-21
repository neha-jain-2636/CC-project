"""Flower FedAvg orchestration with one child process per simulated RSU.

Each child loads only its assigned RSU partition and sends model parameters
and scalar metrics to the coordinating process. This creates an application
boundary for the simulation, but shared-host OS permissions are not a hardened
security boundary.

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
import multiprocessing as mp
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
from src.base_station import create_fedavg_strategy
from src.rsu_client import make_rsu_client

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


def compute_shared_norm_stats() -> NormalizationStats:
    """Return fixed shared scales without reading any RSU arrays."""
    return NormalizationStats.from_public_physical_scales()

def _rsu_worker(
    connection,
    rsu_id: str,
    model_config: dict,
    norm_stats: NormalizationStats,
    data_dir: str,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> None:
    """Serve one RSU client in a dedicated process; raw partitions stay there."""
    try:
        client = make_rsu_client(
            rsu_id=rsu_id, model_config=model_config, norm_stats=norm_stats,
            data_dir=data_dir, local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, seed=seed, device=device,
        )
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command is None:
                break
            action, parameters, config = command
            if action == "fit":
                result = client.fit(parameters, config)
            elif action == "evaluate":
                result = client.evaluate(parameters, config)
            else:
                raise ValueError(f"Unknown RSU worker action: {action}")
            connection.send(("ok", result))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _worker_result(connection, rsu_id):
    status, payload = connection.recv()
    if status != "ok":
        raise RuntimeError(f"RSU worker {rsu_id} failed: {payload}")
    return payload


def run_federated_training(
    data_dir: str = "data/rsu",
    rounds: int = 20,
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

    # ---------- Shared fixed normalization; server reads no RSU arrays ----------
    norm_stats = compute_shared_norm_stats()

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

    # ---------- Process-isolated Flower clients ----------
    start_time = time.time()
    logger.info("Starting process-isolated Flower FedAvg workers")
    from flwr.common import Code, EvaluateRes, FitRes, Status

    context = mp.get_context("spawn")
    workers = []
    try:
        for idx, rsu_id in enumerate(RSU_IDS):
            parent_conn, child_conn = context.Pipe()
            process = context.Process(
                target=_rsu_worker,
                args=(child_conn, rsu_id, model_config, norm_stats, data_dir,
                      local_epochs, batch_size, learning_rate, seed + idx, device),
                name=f"{rsu_id}-flower-client",
            )
            process.start()
            child_conn.close()
            workers.append((rsu_id, parent_conn, process))

        for rsu_id, connection, _ in workers:
            status, payload = connection.recv()
            if status != "ready":
                raise RuntimeError(f"RSU worker {rsu_id} failed to initialize: {payload}")

        round_train_loss = []
        round_val_loss = []
        global_ndarrays = initial_params
        for server_round in range(1, rounds + 1):
            logger.info("Flower FedAvg round %d/%d", server_round, rounds)
            for _, connection, _ in workers:
                connection.send(("fit", global_ndarrays, {"local_epochs": local_epochs}))
            fit_results = []
            for rsu_id, connection, _ in workers:
                weights, num_examples, metrics = _worker_result(connection, rsu_id)
                fit_res = FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(weights),
                    num_examples=num_examples,
                    metrics=metrics,
                )
                fit_results.append((None, fit_res))

            aggregated, fit_metrics = strategy.aggregate_fit(server_round, fit_results, [])
            if aggregated is None:
                raise RuntimeError(f"Flower FedAvg failed to aggregate round {server_round}")
            global_ndarrays = parameters_to_ndarrays(aggregated)
            round_train_loss.append(float(fit_metrics.get("train_loss", float("nan"))))

            for _, connection, _ in workers:
                connection.send(("evaluate", global_ndarrays, {}))
            eval_results = []
            for rsu_id, connection, _ in workers:
                loss, num_examples, metrics = _worker_result(connection, rsu_id)
                eval_res = EvaluateRes(
                    status=Status(code=Code.OK, message=""),
                    loss=float(loss),
                    num_examples=num_examples,
                    metrics=metrics,
                )
                eval_results.append((None, eval_res))
            val_loss, _ = strategy.aggregate_evaluate(server_round, eval_results, [])
            if val_loss is None:
                raise RuntimeError(f"Flower FedAvg failed to aggregate validation round {server_round}")
            round_val_loss.append(float(val_loss))
            logger.info("Round %d complete: train_loss=%.6f val_loss=%.6f",
                        server_round, round_train_loss[-1], round_val_loss[-1])
    finally:
        for _, connection, _ in workers:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for _, connection, process in workers:
            connection.close()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()

    elapsed = time.time() - start_time
    logger.info("Process-isolated Flower FedAvg finished in %.1f s", elapsed)

    class _SimulationHistory:
        losses_distributed = list(enumerate(round_val_loss, start=1))
        metrics_distributed_fit = {"train_loss": list(enumerate(round_train_loss, start=1))}
        metrics_distributed = {}

    history = _SimulationHistory()

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
    fit_metrics = getattr(history, "metrics_distributed_fit", {}) or {}
    round_train_loss = [
        float(value) for _, value in fit_metrics.get("train_loss", [])
    ]
    round_val_loss = [float(value) for _, value in history.losses_distributed]

    metadata = {
        "training_mode": "federated",
        "client_execution": "separate_processes",
        "position_encoding": "relative_to_last_observation",
        "strategy": "FedAvg",
        "num_rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_clients": num_clients,
        "rsu_ids": list(RSU_IDS),
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "history": {"round_train_loss": round_train_loss, "round_val_loss": round_val_loss},
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
        default=20,
        help="Number of federated rounds (default: 20)",
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
