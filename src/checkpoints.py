"""Checkpoint coordination utilities for local, federated, and centralized models.

Provides helpers to load any checkpoint variant and produce consistent
evaluation metrics across all training regimes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.model import VehicleStateGRU
from src.preprocessing import load_rsu_dataset
from src.train_utils import (
    NormalizationStats,
    compute_per_horizon_rmse,
    encode_relative_position_windows,
    create_dataloader,
    load_checkpoint,
    measure_inference_latency,
)

logger = logging.getLogger(__name__)

RSU_IDS = ("rsu_1", "rsu_2", "rsu_3")


def evaluate_checkpoint_on_test(
    checkpoint_path: str | Path,
    data_dir: str = "data/rsu",
    batch_size: int = 256,
    rsu_id: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """Evaluate locally per RSU and aggregate only RMSE sums and counts.

    With ``rsu_id=None``, each RSU's test arrays are evaluated separately.
    Only per-feature squared-error totals and sample counts are combined; raw
    arrays are never concatenated across RSUs.
    """
    if rsu_id is None:
        local_results = [
            evaluate_checkpoint_on_test(
                checkpoint_path, data_dir, batch_size, rsu_id=rid, device=device
            )
            for rid in RSU_IDS
        ]
        total_samples = sum(item["num_test_samples"] for item in local_results)
        if total_samples == 0:
            raise ValueError("No test samples are available across RSUs")

        result = {
            "checkpoint_path": str(checkpoint_path),
            "training_mode": local_results[0]["training_mode"],
            "rsu_id": "all",
            "num_test_samples": total_samples,
            "metadata": local_results[0]["metadata"],
            "aggregation": "sample-weighted RMSE from local squared-error summaries",
        }
        for metric in ("position_rmse", "speed_rmse", "per_feature_rmse"):
            squared_error_total = sum(
                item["num_test_samples"] * np.square(np.asarray(item[metric]))
                for item in local_results
            )
            result[metric] = np.sqrt(squared_error_total / total_samples).tolist()
        return result

    model, norm_stats, metadata = load_checkpoint(checkpoint_path, device)
    X_test_raw, y_test_raw = load_rsu_dataset(rsu_id, "test", data_dir)
    X_test, y_test = encode_relative_position_windows(
        np.array(X_test_raw), np.array(y_test_raw)
    )

    X_test_norm = norm_stats.normalize_input(X_test)
    y_test_norm = norm_stats.normalize_target(y_test)
    test_loader = create_dataloader(
        X_test_norm, y_test_norm, batch_size=batch_size, shuffle=False
    )
    rmse_metrics = compute_per_horizon_rmse(model, test_loader, norm_stats, device)
    return {
        "checkpoint_path": str(checkpoint_path),
        "training_mode": metadata.get("training_mode", "unknown"),
        "rsu_id": rsu_id,
        "num_test_samples": len(X_test),
        "position_rmse": rmse_metrics["position_rmse"].tolist(),
        "speed_rmse": rmse_metrics["speed_rmse"].tolist(),
        "per_feature_rmse": rmse_metrics["per_feature_rmse"].tolist(),
        "metadata": metadata,
    }

def evaluate_all_regimes(
    local_checkpoint_dir: str = "models/local",
    federated_checkpoint_path: str = "models/fedavg_global.pt",
    centralized_checkpoint_path: str = "models/centralized_global.pt",
    data_dir: str = "data/rsu",
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    """Evaluate models without moving raw test arrays across RSU boundaries."""
    results = {}

    local_results = {}
    for rsu_id in RSU_IDS:
        ckpt_path = Path(local_checkpoint_dir) / f"local_{rsu_id}.pt"
        if ckpt_path.exists():
            local_results[rsu_id] = evaluate_checkpoint_on_test(
                ckpt_path, data_dir, batch_size, rsu_id=rsu_id, device=device
            )
        else:
            logger.warning("Local checkpoint not found: %s", ckpt_path)
    results["local_own_zone"] = local_results

    if Path(federated_checkpoint_path).exists():
        results["federated_per_rsu"] = {
            rsu_id: evaluate_checkpoint_on_test(
                federated_checkpoint_path, data_dir, batch_size,
                rsu_id=rsu_id, device=device,
            )
            for rsu_id in RSU_IDS
        }
        results["federated_all"] = evaluate_checkpoint_on_test(
            federated_checkpoint_path, data_dir, batch_size, device=device,
        )
        logger.info(
            "Federated ? position RMSE (aggregate of local metrics): %s",
            results["federated_all"]["position_rmse"],
        )
    else:
        logger.warning("Federated checkpoint not found: %s", federated_checkpoint_path)

    # Centralized is the explicitly privacy-unsafe reference model. Its test
    # scores are still calculated per RSU and aggregated from metric summaries.
    if Path(centralized_checkpoint_path).exists():
        results["centralized_per_rsu"] = {
            rsu_id: evaluate_checkpoint_on_test(
                centralized_checkpoint_path, data_dir, batch_size,
                rsu_id=rsu_id, device=device,
            )
            for rsu_id in RSU_IDS
        }
        results["centralized_all"] = evaluate_checkpoint_on_test(
            centralized_checkpoint_path, data_dir, batch_size, device=device,
        )
    else:
        logger.warning("Centralized checkpoint not found: %s", centralized_checkpoint_path)

    return results

def run_gap_experiment(
    checkpoint_path: str | Path,
    data_dir: str = "data/rsu",
    gap_steps: int = 5,
    batch_size: int = 256,
    device: str = "cpu",
    rsu_ids: Optional[tuple[str, ...]] = None,
) -> dict:
    """Compare frozen state and recursive predictions over up to 500 ms.

    Ground-truth steps 4 and 5 are joined from later windows belonging to the
    same recording and vehicle in the same RSU test partition. Windows without
    the required continuation are excluded from both methods.
    """
    from src.predictor import RSUStatePredictor
    from src.preprocessing import WINDOW_INDEX_DTYPE

    if gap_steps <= 0 or gap_steps > 5:
        raise ValueError("gap_steps must be between 1 and 5 (100-500 ms)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected_rsus = tuple(rsu_ids) if rsu_ids is not None else RSU_IDS
    if not selected_rsus or any(rid not in RSU_IDS for rid in selected_rsus):
        raise ValueError(f"rsu_ids must be a non-empty subset of {RSU_IDS}")

    predictor = RSUStatePredictor.from_checkpoint(checkpoint_path, device)
    pos_sse = np.zeros((2, gap_steps), dtype=np.float64)
    speed_sse = np.zeros((2, gap_steps), dtype=np.float64)
    total_samples = 0
    excluded_missing = 0

    for rsu_id in selected_rsus:
        X_raw, y_raw = load_rsu_dataset(rsu_id, "test", data_dir)
        index_path = Path(data_dir) / rsu_id / "test_window_index.npy"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"{index_path} is required for same-vehicle 500 ms gap evaluation; "
                "regenerate the RSU windows with the current preprocessing pipeline"
            )
        index = np.load(index_path, mmap_mode="r")
        if len(X_raw) != len(y_raw) or len(index) != len(X_raw):
            raise ValueError(f"X/y/index test lengths differ for {rsu_id}")
        if index.dtype != WINDOW_INDEX_DTYPE:
            raise ValueError(f"Unexpected window index dtype for {rsu_id}: {index.dtype}")

        lookup = {}
        for row, item in enumerate(index):
            track = (int(item["recording_code"]), int(item["vehicle_id"]))
            lookup.setdefault(track, {})[int(item["input_start_timestamp_ms"])] = row

        eligible = []
        for row, item in enumerate(index):
            track = (int(item["recording_code"]), int(item["vehicle_id"]))
            start_ms = int(item["input_start_timestamp_ms"])
            track_windows = lookup[track]
            continuation = []
            for step in range(4, gap_steps + 1):
                continuation_row = track_windows.get(start_ms + (step - 3) * 100)
                if continuation_row is None:
                    break
                continuation.append(continuation_row)
            if len(continuation) == gap_steps - 3:
                eligible.append((row, continuation))
            else:
                excluded_missing += 1

        for offset in range(0, len(eligible), batch_size):
            batch_rows = eligible[offset:offset + batch_size]
            rows = np.fromiter((item[0] for item in batch_rows), dtype=np.int64)
            X = np.asarray(X_raw[rows], dtype=np.float32)
            y_future = np.empty((len(rows), gap_steps, 4), dtype=np.float32)
            y_future[:, :min(3, gap_steps), :] = np.asarray(y_raw[rows, :min(3, gap_steps), :])
            if gap_steps > 3:
                continuation_rows = np.asarray([item[1] for item in batch_rows], dtype=np.int64)
                continuation_y = np.asarray(y_raw[continuation_rows, 2, :], dtype=np.float32)
                y_future[:, 3:, :] = continuation_y

            predicted = predictor.predict_recursive(X, total_steps=gap_steps)
            frozen = np.broadcast_to(X[:, -1:, :], y_future.shape)
            for method, estimate in enumerate((frozen, predicted)):
                position_error = estimate[:, :, :2] - y_future[:, :, :2]
                speed_error = estimate[:, :, 2] - y_future[:, :, 2]
                pos_sse[method] += np.square(position_error, dtype=np.float64).sum(axis=(0, 2))
                speed_sse[method] += np.square(speed_error, dtype=np.float64).sum(axis=0)
            total_samples += len(rows)
        del X_raw, y_raw, index

    if total_samples == 0:
        raise ValueError("No test samples have complete same-vehicle continuation")

    pos_rmse = np.sqrt(pos_sse / total_samples)
    speed_rmse = np.sqrt(speed_sse / total_samples)
    freeze_overall = float(np.sqrt(pos_sse[0].sum() / (total_samples * gap_steps)))
    model_overall = float(np.sqrt(pos_sse[1].sum() / (total_samples * gap_steps)))
    improvement = (freeze_overall - model_overall) / freeze_overall * 100 if freeze_overall > 0 else 0.0
    freeze_at_gap_end = float(pos_rmse[0, -1])
    model_at_gap_end = float(pos_rmse[1, -1])
    endpoint_improvement = (
        (freeze_at_gap_end - model_at_gap_end) / freeze_at_gap_end * 100
        if freeze_at_gap_end > 0 else 0.0
    )
    return {
        "gap_steps_evaluated": gap_steps,
        "gap_duration_ms": gap_steps * 100,
        "gap_steps_requested": gap_steps,
        "gap_duration_requested_ms": gap_steps * 100,
        "num_test_samples": total_samples,
        "num_windows_excluded_missing_continuation": excluded_missing,
        "rsu_ids": list(selected_rsus),
        "ground_truth_join": "same recording, vehicle, RSU and test split",
        "freeze_position_rmse_per_step": pos_rmse[0].tolist(),
        "model_position_rmse_per_step": pos_rmse[1].tolist(),
        "freeze_speed_rmse_per_step": speed_rmse[0].tolist(),
        "model_speed_rmse_per_step": speed_rmse[1].tolist(),
        "freeze_overall_position_rmse": freeze_overall,
        "model_overall_position_rmse": model_overall,
        "position_rmse_improvement_pct": round(improvement, 2),
        "position_rmse_improvement_at_gap_end_pct": round(endpoint_improvement, 2),
    }

def compute_weight_update_size(model_config: Optional[dict] = None) -> dict:
    """Compute the weight update size per round in kilobytes.

    Parameters
    ----------
    model_config : dict, optional
        Model configuration. Uses defaults if None.

    Returns
    -------
    dict
        Parameter count, bytes, and KB per update.
    """
    if model_config is None:
        model_config = {
            "input_size": 4,
            "hidden_size": 64,
            "prediction_steps": 3,
            "num_layers": 1,
        }
    model = VehicleStateGRU.from_config(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # In FedAvg, each client uploads its full model; server aggregates and
    # sends back the full model. So upload = download = total_bytes per round.
    return {
        "total_parameters": total_params,
        "bytes_per_update": total_bytes,
        "kb_per_update": round(total_bytes / 1024, 2),
        "upload_per_round_kb": round(total_bytes / 1024, 2),
        "download_per_round_kb": round(total_bytes / 1024, 2),
        "total_per_round_per_client_kb": round(2 * total_bytes / 1024, 2),
        "dtype": "float32",
        "model_config": model_config,
    }

