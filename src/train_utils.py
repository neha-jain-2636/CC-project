"""Training utilities shared by local, federated, and centralized training.

This module provides:
- NormalizationStats: compute, apply, and inverse-transform normalization
  using training-data-only statistics.
- DataLoader creation from numpy arrays.
- train_one_epoch / evaluate helpers that can be called by any training mode.
- Checkpoint save/load with normalization metadata.

All functions are CPU-compatible and do not import Flower, Pandas, or
Matplotlib.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import VehicleStateGRU


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    """Per-feature normalization values for the four state features.

    The main experiment pipeline uses shared fixed physical scales so that RSU
    clients can normalize independently without pooling private samples. The
    optional ``from_training_data`` constructor remains for standalone uses.
    """

    input_mean: np.ndarray   # [4]
    input_std: np.ndarray    # [4]
    target_mean: np.ndarray  # [4]
    target_std: np.ndarray   # [4]

    def __post_init__(self) -> None:
        for name, arr in [
            ("input_mean", self.input_mean),
            ("input_std", self.input_std),
            ("target_mean", self.target_mean),
            ("target_std", self.target_std),
        ]:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != (4,):
                raise ValueError(
                    f"{name} must have shape (4,), got {arr.shape}"
                )
            setattr(self, name, arr)
        # Prevent division by zero
        if np.any(self.input_std < 1e-8):
            raise ValueError(
                "input_std contains near-zero values; cannot normalize safely"
            )
        if np.any(self.target_std < 1e-8):
            raise ValueError(
                "target_std contains near-zero values; cannot normalize safely"
            )

    @classmethod
    def from_public_physical_scales(cls) -> "NormalizationStats":
        """Return shared, data-independent scales for all training regimes.

        Fixed engineering scales avoid pooling RSU samples to calculate
        normalization statistics. Position features are displacement from the
        final observed state and both axes use a 15 m scale; speed uses 30 m/s
        and heading uses pi radians.
        """
        means = np.zeros(4, dtype=np.float32)
        scales = np.array([15.0, 15.0, 30.0, np.pi], dtype=np.float32)
        return cls(means, scales, means.copy(), scales.copy())

    @classmethod
    def from_training_data(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "NormalizationStats":
        """Compute statistics from training arrays.

        Parameters
        ----------
        X_train : ndarray
            Shape ``[N, 5, 4]`` — training input windows.
        y_train : ndarray
            Shape ``[N, 3, 4]`` — training target windows.
        """
        if X_train.ndim != 3 or X_train.shape[2] != 4:
            raise ValueError(
                f"X_train must be [N, T, 4], got shape {X_train.shape}"
            )
        if y_train.ndim != 3 or y_train.shape[2] != 4:
            raise ValueError(
                f"y_train must be [N, T, 4], got shape {y_train.shape}"
            )

        # Flatten time dimension for per-feature statistics
        x_flat = X_train.reshape(-1, 4).astype(np.float64)
        y_flat = y_train.reshape(-1, 4).astype(np.float64)

        return cls(
            input_mean=x_flat.mean(axis=0).astype(np.float32),
            input_std=x_flat.std(axis=0).astype(np.float32),
            target_mean=y_flat.mean(axis=0).astype(np.float32),
            target_std=y_flat.std(axis=0).astype(np.float32),
        )

    def normalize_input(self, X: np.ndarray) -> np.ndarray:
        """Normalize input array in-place-safe manner.

        Parameters
        ----------
        X : ndarray
            Shape ``[..., 4]``.

        Returns
        -------
        ndarray
            Normalized copy, same shape, float32.
        """
        return ((X.astype(np.float32) - self.input_mean) / self.input_std).astype(
            np.float32
        )

    def normalize_target(self, y: np.ndarray) -> np.ndarray:
        """Normalize target array."""
        return ((y.astype(np.float32) - self.target_mean) / self.target_std).astype(
            np.float32
        )

    def inverse_transform_target(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to physical units.

        This must be called before computing RMSE in physical units.

        Parameters
        ----------
        y_norm : ndarray or Tensor
            Shape ``[..., 4]``, normalized.

        Returns
        -------
        ndarray
            Physical-unit values, float32.
        """
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.detach().cpu().numpy()
        return (y_norm.astype(np.float32) * self.target_std + self.target_mean).astype(
            np.float32
        )

    def to_dict(self) -> dict:
        """Serialize to a dictionary for checkpoint storage."""
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationStats":
        """Deserialize from a checkpoint dictionary."""
        return cls(
            input_mean=np.array(d["input_mean"], dtype=np.float32),
            input_std=np.array(d["input_std"], dtype=np.float32),
            target_mean=np.array(d["target_mean"], dtype=np.float32),
            target_std=np.array(d["target_std"], dtype=np.float32),
        )


def encode_relative_position_windows(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode position in each window relative to its last observed state.

    The function mutates writable float32 arrays to avoid unnecessary copies
    of the large NGSIM partitions. Read-only or non-float32 inputs are copied.
    Speed and heading remain in their original physical units.
    """
    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    if X_array.ndim != 3 or X_array.shape[1:] != (5, 4):
        raise ValueError(f"X must have shape [N, 5, 4], got {X_array.shape}")
    if y_array.ndim != 3 or y_array.shape[0] != X_array.shape[0] or y_array.shape[2] != 4:
        raise ValueError(f"y must have shape [N, K, 4] matching X, got {y_array.shape}")
    if not X_array.flags.writeable:
        X_array = X_array.copy()
    if not y_array.flags.writeable:
        y_array = y_array.copy()

    origin = X_array[:, -1:, :2].copy()
    X_array[:, :, :2] -= origin
    y_array[:, :, :2] -= origin
    return X_array, y_array


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays.

    Parameters
    ----------
    X : ndarray
        Input array, shape ``[N, 5, 4]``, float32.
    y : ndarray
        Target array, shape ``[N, 3, 4]``, float32.
    batch_size : int
        Batch size (default 256).
    shuffle : bool
        Whether to shuffle (default True; set False for evaluation).

    Returns
    -------
    DataLoader
    """
    if X.ndim != 3 or y.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays, got X:{X.ndim}D and y:{y.ndim}D"
        )
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples: {len(X)} vs {len(y)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    X_tensor = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y).astype(np.float32))
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: VehicleStateGRU,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> float:
    """Train the model for one epoch.

    Parameters
    ----------
    model : VehicleStateGRU
        The shared GRU model.
    dataloader : DataLoader
        Training data loader.
    optimizer : Optimizer
        PyTorch optimizer.
    criterion : Module
        Loss function (typically ``nn.MSELoss()``).
    device : device or str
        Device to train on (default "cpu").

    Returns
    -------
    float
        Average training loss over all batches.
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: VehicleStateGRU,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate the model on a dataset.

    Parameters
    ----------
    model : VehicleStateGRU
        The shared GRU model.
    dataloader : DataLoader
        Evaluation data loader (should have shuffle=False).
    criterion : Module
        Loss function (typically ``nn.MSELoss()``).
    device : device or str
        Device (default "cpu").

    Returns
    -------
    float
        Average loss over all batches.
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Per-horizon RMSE (physical units)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_horizon_rmse(
    model: VehicleStateGRU,
    dataloader: DataLoader,
    norm_stats: NormalizationStats,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Compute position and speed RMSE at each prediction horizon.

    Returns a dictionary with:
    - ``"position_rmse"``: ndarray [prediction_steps] — Euclidean position RMSE
      in meters at each horizon step.
    - ``"speed_rmse"``: ndarray [prediction_steps] — speed RMSE in m/s at each
      horizon step.
    - ``"per_feature_rmse"``: ndarray [prediction_steps, 4] — per-feature RMSE
      at each horizon step in physical units.

    All predictions are inverse-transformed to physical units before RMSE
    computation.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

    if not all_preds:
        raise ValueError("Empty dataloader; cannot compute RMSE")

    preds_norm = np.concatenate(all_preds, axis=0)   # [N, K, 4]
    targets_norm = np.concatenate(all_targets, axis=0)  # [N, K, 4]

    # Inverse transform to physical units
    preds_phys = norm_stats.inverse_transform_target(preds_norm)
    targets_phys = norm_stats.inverse_transform_target(targets_norm)

    # Per-feature squared error: [N, K, 4]
    sq_error = (preds_phys - targets_phys) ** 2

    # Per-feature RMSE at each horizon: [K, 4]
    per_feature_rmse = np.sqrt(sq_error.mean(axis=0))

    # Position RMSE: sqrt(mean(dx^2 + dy^2)) at each horizon
    position_sq_error = sq_error[:, :, 0] + sq_error[:, :, 1]  # [N, K]
    position_rmse = np.sqrt(position_sq_error.mean(axis=0))  # [K]

    # Speed RMSE at each horizon
    speed_rmse = per_feature_rmse[:, 2]  # [K]

    return {
        "position_rmse": position_rmse,
        "speed_rmse": speed_rmse,
        "per_feature_rmse": per_feature_rmse,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: VehicleStateGRU,
    norm_stats: NormalizationStats,
    path: str | Path,
    metadata: Optional[dict] = None,
) -> Path:
    """Save model checkpoint with normalization and config metadata.

    Parameters
    ----------
    model : VehicleStateGRU
    norm_stats : NormalizationStats
    path : str or Path
        Destination file path.
    metadata : dict, optional
        Extra metadata (e.g. training_mode, rsu_id, epochs).

    Returns
    -------
    Path
        Absolute path to the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "normalization": norm_stats.to_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    return path.resolve()


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[VehicleStateGRU, NormalizationStats, dict]:
    """Load a checkpoint and return (model, norm_stats, metadata).

    Parameters
    ----------
    path : str or Path
        Checkpoint file path.
    device : device or str
        Device for the model (default "cpu").

    Returns
    -------
    tuple
        ``(model, norm_stats, metadata)``
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = VehicleStateGRU.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    norm_stats = NormalizationStats.from_dict(checkpoint["normalization"])
    metadata = checkpoint.get("metadata", {})

    return model, norm_stats, metadata


# ---------------------------------------------------------------------------
# Inference latency measurement
# ---------------------------------------------------------------------------


def measure_inference_latency(
    model: VehicleStateGRU,
    input_shape: tuple[int, ...] = (1, 5, 4),
    warmup_runs: int = 10,
    timed_runs: int = 100,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Measure single-sample inference latency.

    Performs ``warmup_runs`` untimed forward passes followed by
    ``timed_runs`` measured passes.

    Returns
    -------
    dict
        ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
        ``warmup_runs``, ``timed_runs``.
    """
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_shape, device=device)

    # Warm up
    for _ in range(warmup_runs):
        _ = model(dummy)

    # Timed runs
    latencies = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        _ = model(dummy)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    latencies_arr = np.array(latencies)
    return {
        "mean_ms": float(latencies_arr.mean()),
        "std_ms": float(latencies_arr.std()),
        "min_ms": float(latencies_arr.min()),
        "max_ms": float(latencies_arr.max()),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
    }
