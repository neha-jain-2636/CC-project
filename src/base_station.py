"""Base-station (server-side) aggregation strategy for Flower FedAvg."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import flwr as fl
from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
)
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


def _weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_samples = sum(n for n, _ in metrics)
    if total_samples == 0:
        return {"val_loss": float("nan")}

    weighted_loss = sum(n * float(m.get("val_loss", 0.0)) for n, m in metrics)
    avg_loss = weighted_loss / total_samples

    for n, m in metrics:
        logger.info("  Client %s: val_loss=%.6f  samples=%d", m.get("rsu_id", "?"), m.get("val_loss", float("nan")), n)
    logger.info("  Aggregated val_loss=%.6f  total_samples=%d", avg_loss, total_samples)
    return {"val_loss": avg_loss}


def _fit_metrics_aggregation(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_samples = sum(n for n, _ in metrics)
    if total_samples == 0:
        return {"train_loss": float("nan")}

    weighted_loss = sum(n * float(m.get("train_loss", 0.0)) for n, m in metrics)
    avg_loss = weighted_loss / total_samples
    total_payload = sum(int(m.get("payload_bytes", 0)) for _, m in metrics)

    for n, m in metrics:
        logger.info(
            "  Client %s: train_loss=%.6f  epochs=%s  samples=%d  payload=%d B",
            m.get("rsu_id", "?"),
            m.get("train_loss", float("nan")),
            m.get("local_epochs", "?"),
            n,
            int(m.get("payload_bytes", 0)),
        )
    logger.info("  Aggregated train_loss=%.6f  total_samples=%d  total_payload=%d B", avg_loss, total_samples, total_payload)
    return {"train_loss": avg_loss, "total_payload_bytes": total_payload}


class TrackedFedAvg(FedAvg):
    """FedAvg subclass that stores the latest aggregated model parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Optional[Parameters] = kwargs.get("initial_parameters", None)

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, metrics


def create_fedavg_strategy(
    initial_parameters: NDArrays,
    min_fit_clients: int = 3,
    min_evaluate_clients: int = 3,
    min_available_clients: int = 3,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
) -> TrackedFedAvg:
    init_params_obj = ndarrays_to_parameters(initial_parameters)
    strategy = TrackedFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=init_params_obj,
        evaluate_metrics_aggregation_fn=_weighted_average,
        fit_metrics_aggregation_fn=_fit_metrics_aggregation,
    )
    return strategy