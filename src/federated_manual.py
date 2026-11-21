"""Backward-compatible entry point that delegates to Flower FedAvg.

The former manual implementation loaded each RSU partition into one process.
This alias delegates to the Flower client/server simulation instead.
"""

from __future__ import annotations

import warnings
from typing import Optional

from src.federated_training import run_federated_training
from src.train_utils import load_checkpoint


def run_federated_manual(
    data_dir: str = "data/rsu",
    rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "models/fedavg_global.pt",
    model_config: Optional[dict] = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Deprecated compatibility alias; delegates without pooling RSU arrays."""
    warnings.warn(
        "run_federated_manual is deprecated; use run_federated_training",
        DeprecationWarning,
        stacklevel=2,
    )
    saved = run_federated_training(
        data_dir=data_dir,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        seed=seed,
        device=device,
    )
    _, _, metadata = load_checkpoint(saved, device=device)
    return {
        "checkpoint_path": str(saved),
        "history": metadata.get("history", {}),
        "elapsed_seconds": metadata.get("elapsed_seconds", 0.0),
    }


def parse_args(argv: Optional[list[str]] = None):
    """Keep the legacy CLI while routing it through Flower."""
    import argparse

    parser = argparse.ArgumentParser(description="Deprecated Flower FedAvg alias")
    parser.add_argument("--data-dir", type=str, default="data/rsu")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-path", type=str, default="models/fedavg_global.pt")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_federated_manual(
        data_dir=args.data_dir,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        model_config={
            "input_size": 4,
            "hidden_size": args.hidden_size,
            "prediction_steps": 3,
            "num_layers": args.num_layers,
        },
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
