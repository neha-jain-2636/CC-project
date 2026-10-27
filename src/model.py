"""Shared GRU model for vehicle state prediction.

This is the single model architecture used across local, federated, and
centralized training.  Every other module imports from here — do not create
a competing architecture elsewhere.

Architecture
------------
- Input:  [B, T_obs, 4]  — 4 features: x_m, y_m, speed_mps, heading_rad
- Output: [B, T_pred, 4] — predicted future states
- Core: PyTorch GRU encoder → fully-connected decoder

Heading handling
----------------
Heading is kept as a raw radian value (no sin/cos decomposition).  This is a
documented simplification that works well for small angular changes between
consecutive 100 ms time steps.  It does not handle wraparound at ±π; the MSE
loss treats heading linearly.  We chose this to keep the feature dimension
fixed at 4 and avoid complicating the inverse transform.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VehicleStateGRU(nn.Module):
    """GRU-based vehicle state predictor.

    Parameters
    ----------
    input_size : int
        Number of input features per time step (default 4).
    hidden_size : int
        GRU hidden dimension, recommended 32–64 (default 64).
    prediction_steps : int
        Number of future time steps to predict (default 3).
    num_layers : int
        Number of stacked GRU layers (default 1).

    Forward pass
    ------------
    Input:  Tensor of shape ``[B, observation_steps, input_size]``
    Output: Tensor of shape ``[B, prediction_steps, input_size]``

    The encoder GRU processes the full observation sequence.  The final hidden
    state is passed through a linear projection to produce all prediction steps
    at once (single-shot decoding).  This avoids autoregressive error
    accumulation during the primary 3-step prediction.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        prediction_steps: int = 3,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if prediction_steps <= 0:
            raise ValueError(
                f"prediction_steps must be positive, got {prediction_steps}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prediction_steps = prediction_steps
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Single-shot decoder: hidden → all prediction steps flattened
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_steps * input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``[B, T_obs, input_size]``.

        Returns
        -------
        Tensor
            Shape ``[B, prediction_steps, input_size]``.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input [B, T, F], got {x.ndim}D with shape {x.shape}"
            )
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {x.shape[2]}"
            )

        # Encode observation sequence
        _, hidden = self.gru(x)  # hidden: [num_layers, B, hidden_size]

        # Use the last layer's hidden state
        last_hidden = hidden[-1]  # [B, hidden_size]

        # Decode to all prediction steps at once
        flat_output = self.decoder(last_hidden)  # [B, prediction_steps * input_size]

        # Reshape to [B, prediction_steps, input_size]
        output = flat_output.view(-1, self.prediction_steps, self.input_size)
        return output

    def get_config(self) -> dict:
        """Return model configuration as a serializable dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "prediction_steps": self.prediction_steps,
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_config(cls, config: dict) -> "VehicleStateGRU":
        """Create a model instance from a configuration dictionary."""
        return cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            prediction_steps=config["prediction_steps"],
            num_layers=config["num_layers"],
        )
