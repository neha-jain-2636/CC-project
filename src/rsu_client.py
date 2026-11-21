"""Flower client wrapping one RSU's local data and the shared GRU model.

Each RSU client:
- Loads its own partition via ``load_rsu_dataset(rsu_id, split)``.
- Normalizes data using supplied ``NormalizationStats``.
- Trains using Person 2's ``train_one_epoch`` / ``evaluate`` utilities.
- Exposes ``get_parameters`` / ``set_parameters`` for Flower FedAvg.

This module does **not** create a second model architecture; it imports
``VehicleStateGRU`` from ``src.model``.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
)

from src.model import VehicleStateGRU
from src.train_utils import (
    NormalizationStats,
    create_dataloader,
    evaluate,
    encode_relative_position_windows,
    train_one_epoch,
)
from src.preprocessing import load_rsu_dataset

logger = logging.getLogger(__name__)


def _parameter_byte_size(ndarrays: NDArrays) -> int:
    """Total bytes across all parameter arrays (for logging payload sizes)."""
    return sum(arr.nbytes for arr in ndarrays)


class RSUFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for one RSU.

    Parameters
    ----------
    rsu_id : str
        One of ``"rsu_1"``, ``"rsu_2"``, ``"rsu_3"``.
    model : VehicleStateGRU
        Shared GRU model instance (weights are overwritten each round).
    norm_stats : NormalizationStats
        Normalization statistics computed from training data.
    data_dir : str
        Root directory for RSU-partitioned data (default ``"data/rsu"``).
    local_epochs : int
        Number of local training epochs per federated round.
    batch_size : int
        Mini-batch size for training/evaluation.
    learning_rate : float
        Learning rate for the local Adam optimizer.
    seed : int
        Random seed for reproducibility within this client.
    device : str
        Torch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        rsu_id: str,
        model: VehicleStateGRU,
        norm_stats: NormalizationStats,
        data_dir: str = "data/rsu",
        local_epochs: int = 1,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.rsu_id = rsu_id
        self.model = model
        self.norm_stats = norm_stats
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        # Seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ---------- Load & normalise this RSU's data ----------
        logger.info(
            "RSU client %s: loading data from %s", rsu_id, data_dir
        )
        X_train_raw, y_train_raw = load_rsu_dataset(
            rsu_id, split="train", data_directory=data_dir
        )
        X_val_raw, y_val_raw = load_rsu_dataset(
            rsu_id, split="validation", data_directory=data_dir
        )

        # Copy from read-only memory maps, encode local displacement, then scale.
        X_train_raw, y_train_raw = encode_relative_position_windows(
            np.array(X_train_raw), np.array(y_train_raw)
        )
        X_val_raw, y_val_raw = encode_relative_position_windows(
            np.array(X_val_raw), np.array(y_val_raw)
        )
        X_train = norm_stats.normalize_input(X_train_raw)
        y_train = norm_stats.normalize_target(y_train_raw)
        X_val = norm_stats.normalize_input(X_val_raw)
        y_val = norm_stats.normalize_target(y_val_raw)

        self.train_loader = create_dataloader(
            X_train, y_train, batch_size=batch_size, shuffle=True
        )
        self.val_loader = create_dataloader(
            X_val, y_val, batch_size=batch_size, shuffle=False
        )
        self.num_train_samples = len(X_train)
        self.num_val_samples = len(X_val)

        logger.info(
            "RSU client %s: %d train samples, %d val samples",
            rsu_id,
            self.num_train_samples,
            self.num_val_samples,
        )

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as a list of NumPy arrays."""
        params = [
            val.cpu().detach().numpy().copy()
            for _, val in self.model.state_dict().items()
        ]
        logger.debug(
            "RSU %s get_parameters: %d arrays, %d bytes",
            self.rsu_id,
            len(params),
            _parameter_byte_size(params),
        )
        return params

    def set_parameters(self, parameters: NDArrays) -> None:
        """Replace model parameters with the server-provided arrays."""
        keys = list(self.model.state_dict().keys())
        if len(parameters) != len(keys):
            raise ValueError(
                f"RSU {self.rsu_id}: expected {len(keys)} parameter arrays, "
                f"got {len(parameters)}"
            )
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)
        logger.debug(
            "RSU %s set_parameters: loaded %d bytes",
            self.rsu_id,
            _parameter_byte_size(parameters),
        )

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Local training round."""
        self.set_parameters(parameters)
        
        # --- ADD THIS LINE HERE ---
        self.model.train()

        # Allow the server to override local_epochs per round
        epochs = int(config.get("local_epochs", self.local_epochs))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        total_train_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = train_one_epoch(
                self.model, self.train_loader, optimizer, criterion, self.device
            )
            total_train_loss += epoch_loss
            logger.info(
                "RSU %s  epoch %d/%d  train_loss=%.6f",
                self.rsu_id,
                epoch + 1,
                epochs,
                epoch_loss,
            )

        avg_train_loss = total_train_loss / max(epochs, 1)

        updated_params = self.get_parameters(config={})
        metrics: dict[str, Scalar] = {
            "rsu_id": self.rsu_id,
            "train_loss": float(avg_train_loss),
            "local_epochs": epochs,
            "payload_bytes": _parameter_byte_size(updated_params),
        }
        return updated_params, self.num_train_samples, metrics
    
    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model on this RSU's validation data.

        Returns
        -------
        tuple
            (val_loss, num_val_samples, metrics_dict)
        """
        self.set_parameters(parameters)
        criterion = nn.MSELoss()
        val_loss = evaluate(
            self.model, self.val_loader, criterion, self.device
        )
        metrics: dict[str, Scalar] = {
            "rsu_id": self.rsu_id,
            "val_loss": float(val_loss),
        }
        logger.info(
            "RSU %s  evaluate  val_loss=%.6f", self.rsu_id, val_loss
        )
        return float(val_loss), self.num_val_samples, metrics


def make_rsu_client(
    rsu_id: str,
    model_config: dict,
    norm_stats: NormalizationStats,
    data_dir: str = "data/rsu",
    local_epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> RSUFlowerClient:
    """Factory: build a fresh model and wrap it in an RSUFlowerClient.

    This is the preferred entry point so each client gets its own model
    instance.  The model weights will be overwritten by the server at the
    start of each round.
    """
    model = VehicleStateGRU.from_config(model_config)
    return RSUFlowerClient(
        rsu_id=rsu_id,
        model=model,
        norm_stats=norm_stats,
        data_dir=data_dir,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
