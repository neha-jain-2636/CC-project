"""Inference API for vehicle state prediction.

This module provides ``RSUStatePredictor``, a lightweight inference wrapper
that loads a trained checkpoint and predicts future vehicle states from a
history of observed states.

Dependencies
------------
Only ``torch`` and ``numpy`` are required.  This module intentionally does
**not** import Flower, Pandas, or Matplotlib so that it can be embedded in
edge/inference deployments without heavy dependencies.

Usage
-----
>>> from src.predictor import RSUStatePredictor
>>> predictor = RSUStatePredictor.from_checkpoint("models/fedavg_global.pt")
>>> history = np.array([[x, y, speed, heading]] * 5, dtype=np.float32)  # [5, 4]
>>> future = predictor.predict(history)  # [3, 4]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.model import VehicleStateGRU


class RSUStatePredictor:
    """Lightweight inference wrapper around a trained VehicleStateGRU.

    This class handles:
    - Loading a checkpoint (model weights + normalization stats + config).
    - Normalizing input using training-only statistics.
    - Running the GRU forward pass on CPU.
    - Inverse-transforming predictions back to physical units.

    Parameters
    ----------
    model : VehicleStateGRU
        A loaded model in eval mode.
    input_mean : ndarray [4]
        Per-feature mean from training data (input side).
    input_std : ndarray [4]
        Per-feature std from training data (input side).
    target_mean : ndarray [4]
        Per-feature mean from training data (target side).
    target_std : ndarray [4]
        Per-feature std from training data (target side).
    metadata : dict
        Checkpoint metadata for provenance tracking.
    """

    def __init__(
        self,
        model: VehicleStateGRU,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        self._model = model
        self._model.eval()

        self._input_mean = torch.from_numpy(
            np.asarray(input_mean, dtype=np.float32)
        )
        self._input_std = torch.from_numpy(
            np.asarray(input_std, dtype=np.float32)
        )
        self._target_mean = torch.from_numpy(
            np.asarray(target_mean, dtype=np.float32)
        )
        self._target_std = torch.from_numpy(
            np.asarray(target_std, dtype=np.float32)
        )
        self._metadata = metadata or {}

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "RSUStatePredictor":
        """Load a predictor from a saved checkpoint file.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pt`` checkpoint saved by
            ``train_utils.save_checkpoint()``.
        device : str
            Device for inference (default ``"cpu"``).

        Returns
        -------
        RSUStatePredictor
            Ready-to-use predictor instance.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        KeyError
            If the checkpoint is missing required fields.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Validate required keys
        for key in ("model_state_dict", "model_config", "normalization"):
            if key not in checkpoint:
                raise KeyError(
                    f"Checkpoint is missing required key: '{key}'"
                )

        model = VehicleStateGRU.from_config(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        norm = checkpoint["normalization"]
        for nkey in ("input_mean", "input_std", "target_mean", "target_std"):
            if nkey not in norm:
                raise KeyError(
                    f"Checkpoint normalization is missing key: '{nkey}'"
                )

        return cls(
            model=model,
            input_mean=np.array(norm["input_mean"], dtype=np.float32),
            input_std=np.array(norm["input_std"], dtype=np.float32),
            target_mean=np.array(norm["target_mean"], dtype=np.float32),
            target_std=np.array(norm["target_std"], dtype=np.float32),
            metadata=checkpoint.get("metadata", {}),
        )

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict future vehicle states from observed history.

        Parameters
        ----------
        history : ndarray
            Either a single sample ``[5, 4]`` or a batch ``[B, 5, 4]``.
            Features are ``(x_m, y_m, speed_mps, heading_rad)``.
            Accepts float32 or float64 input.

        Returns
        -------
        ndarray
            Predicted future states in physical units:
            ``[3, 4]`` for a single sample, ``[B, 3, 4]`` for a batch.

        Raises
        ------
        ValueError
            If the input shape is invalid.
        """
        history = np.asarray(history, dtype=np.float32)
        single = False

        if history.ndim == 2:
            # Single sample [5, 4] → [1, 5, 4]
            if history.shape != (5, self._model.input_size):
                raise ValueError(
                    f"Single-sample input must be [5, {self._model.input_size}], "
                    f"got {history.shape}"
                )
            history = history[np.newaxis, :, :]
            single = True
        elif history.ndim == 3:
            if history.shape[1] != 5 or history.shape[2] != self._model.input_size:
                raise ValueError(
                    f"Batch input must be [B, 5, {self._model.input_size}], "
                    f"got {history.shape}"
                )
        else:
            raise ValueError(
                f"Input must be 2D [5, 4] or 3D [B, 5, 4], got {history.ndim}D"
            )

        relative_encoding = (
            self._metadata.get("position_encoding") == "relative_to_last_observation"
        )
        position_origin = None
        if relative_encoding:
            history = history.copy()
            position_origin = history[:, -1:, :2].copy()
            history[:, :, :2] -= position_origin

        # Normalize
        x_tensor = torch.from_numpy(history)
        x_norm = (x_tensor - self._input_mean) / self._input_std

        # Forward pass (no grad)
        with torch.no_grad():
            pred_norm = self._model(x_norm)

        # Inverse transform, then restore the absolute position origin.
        pred_phys = pred_norm * self._target_std + self._target_mean
        if relative_encoding and position_origin is not None:
            pred_phys[:, :, :2] += torch.from_numpy(position_origin)
        result = pred_phys.numpy().astype(np.float32)

        if single:
            return result[0]  # [3, 4]
        return result  # [B, 3, 4]

    def predict_recursive(
        self,
        history: np.ndarray,
        total_steps: int = 5,
    ) -> np.ndarray:
        """Predict ``total_steps`` ahead, recursively, for one sample or a batch.

        The trained model emits three steps per call. When more are requested,
        its newest three predictions are appended to the rolling five-state
        history and the next model output is used to continue the forecast.
        """
        history = np.asarray(history, dtype=np.float32)
        single = history.ndim == 2
        if single:
            if history.shape != (5, self._model.input_size):
                raise ValueError(
                    f"history must be [5, {self._model.input_size}], got {history.shape}"
                )
            history = history[np.newaxis, :, :]
        elif history.ndim != 3 or history.shape[1:] != (5, self._model.input_size):
            raise ValueError(
                f"history must be [5, {self._model.input_size}] or "
                f"[B, 5, {self._model.input_size}], got {history.shape}"
            )
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")

        outputs = []
        current = history.copy()
        remaining = total_steps
        while remaining:
            next_states = self.predict(current)
            take = min(remaining, next_states.shape[1])
            outputs.append(next_states[:, :take, :])
            remaining -= take
            if remaining:
                current = np.concatenate((current[:, 3:, :], next_states), axis=1)[:, -5:, :]

        result = np.concatenate(outputs, axis=1)
        return result[0] if single else result

    @property
    def metadata(self) -> dict:
        """Checkpoint metadata (read-only)."""
        return dict(self._metadata)

    @property
    def model_config(self) -> dict:
        """Model configuration dictionary."""
        return self._model.get_config()
