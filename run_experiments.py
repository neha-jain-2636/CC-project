"""Full experiment pipeline: runs all three training regimes and evaluation.

Usage (from repository root)::

    python run_experiments.py --data-dir data/rsu --epochs 30 --rounds 20

This script orchestrates:
1. Local-only training (3 independent RSU models)
2. Federated training (FedAvg with 3 RSU clients)
3. Centralized baseline (pooled data, privacy-unsafe)
4. Evaluation on test set for all three regimes
5. 500 ms gap experiment
6. Weight update size calculation
7. Inference latency measurement
8. Saving all results and generating plots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import VehicleStateGRU
from src.train_utils import (
    NormalizationStats,
    load_checkpoint,
    measure_inference_latency,
)
from src.local_training import run_local_training
from src.centralized_training import run_centralized_training
from src.federated_training import run_federated_training
from src.checkpoints import (
    compute_weight_update_size,
    evaluate_all_regimes,
    run_gap_experiment,
)

logger = logging.getLogger(__name__)


def run_full_pipeline(
    data_dir: str = "data/rsu",
    epochs: int = 30,
    rounds: int = 20,
    local_epochs_per_round: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    seed: int = 42,
    output_dir: str = "results",
    models_dir: str = "models",
) -> dict:
    """Run the full experiment pipeline."""

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)

    model_config = {
        "input_size": 4,
        "hidden_size": hidden_size,
        "prediction_steps": 3,
        "num_layers": 1,
    }

    all_results = {}

    # ================================================================
    # 1. LOCAL-ONLY TRAINING
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOCAL-ONLY BASELINE TRAINING")
    logger.info("=" * 70)
    local_start = time.time()
    local_results = run_local_training(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=str(models_dir_path / "local"),
        model_config=model_config,
        seed=seed,
        device="cpu",
    )
    local_elapsed = time.time() - local_start
    all_results["local_training"] = {
        rsu_id: {
            "final_train_loss": r["final_train_loss"],
            "final_val_loss": r["final_val_loss"],
            "elapsed_seconds": r["elapsed_seconds"],
        }
        for rsu_id, r in local_results.items()
    }
    logger.info("Local training completed in %.1f s", local_elapsed)

    # ================================================================
    # 2. FEDERATED TRAINING (FedAvg)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: FEDERATED TRAINING (FedAvg)")
    logger.info("=" * 70)
    fed_start = time.time()
    federated_checkpoint_path = str(models_dir_path / "fedavg_global.pt")
    federated_checkpoint = run_federated_training(
        data_dir=data_dir,
        rounds=rounds,
        local_epochs=local_epochs_per_round,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=federated_checkpoint_path,
        model_config=model_config,
        seed=seed,
        device="cpu",
    )
    _, _, fed_metadata = load_checkpoint(federated_checkpoint)
    fed_results = {"history": fed_metadata.get("history", {})}
    fed_elapsed = float(fed_metadata.get("elapsed_seconds", time.time() - fed_start))
    all_results["federated_training"] = {
        "elapsed_seconds": round(fed_elapsed, 1),
        "num_rounds": rounds,
        "round_train_loss": fed_results["history"]["round_train_loss"],
        "round_val_loss": fed_results["history"]["round_val_loss"],
    }
    logger.info("Federated training completed in %.1f s", fed_elapsed)

    # ================================================================
    # 3. CENTRALIZED BASELINE
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: CENTRALIZED BASELINE (PRIVACY-UNSAFE)")
    logger.info("=" * 70)
    cent_start = time.time()
    centralized_checkpoint_path = str(models_dir_path / "centralized_global.pt")
    cent_results = run_centralized_training(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=centralized_checkpoint_path,
        model_config=model_config,
        seed=seed,
        device="cpu",
    )
    cent_elapsed = time.time() - cent_start
    all_results["centralized_training"] = {
        "final_train_loss": cent_results["history"]["train_loss"][-1],
        "final_val_loss": cent_results["history"]["val_loss"][-1],
        "elapsed_seconds": cent_results["elapsed_seconds"],
        "total_train_samples": cent_results["total_train_samples"],
    }
    logger.info("Centralized training completed in %.1f s", cent_elapsed)

    # ================================================================
    # 4. EVALUATION ON TEST DATA
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: EVALUATION ON TEST DATA")
    logger.info("=" * 70)
    eval_results = evaluate_all_regimes(
        local_checkpoint_dir=str(models_dir_path / "local"),
        federated_checkpoint_path=federated_checkpoint_path,
        centralized_checkpoint_path=centralized_checkpoint_path,
        data_dir=data_dir,
        batch_size=batch_size,
    )
    all_results["evaluation"] = eval_results

    # ================================================================
    # 5. FULL 500 ms GAP EXPERIMENT
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: FULL 500 ms GAP EXPERIMENT")
    logger.info("=" * 70)
    # Each local checkpoint is evaluated only on its own RSU's test partition.
    gap_checkpoints = {
        "federated": (federated_checkpoint_path, ("rsu_1", "rsu_2", "rsu_3")),
        "centralized": (centralized_checkpoint_path, ("rsu_1", "rsu_2", "rsu_3")),
    }
    for rsu_id in ("rsu_1", "rsu_2", "rsu_3"):
        lpath = str(models_dir_path / "local" / f"local_{rsu_id}.pt")
        if Path(lpath).exists():
            gap_checkpoints[f"local_{rsu_id}"] = (lpath, (rsu_id,))

    gap_all = {}
    best_gap_name = None
    best_gap_improvement = -float("inf")
    for gap_name, (gap_ckpt, evaluation_rsus) in gap_checkpoints.items():
        if not Path(gap_ckpt).exists():
            continue
        gr = run_gap_experiment(
            gap_ckpt,
            data_dir=data_dir,
            gap_steps=5,
            batch_size=batch_size,
            rsu_ids=evaluation_rsus,
        )
        gap_all[gap_name] = gr
        logger.info(
            "Gap experiment [%s] — Freeze RMSE: %.4f m, Model RMSE: %.4f m, Improvement: %.2f%%",
            gap_name,
            gr["freeze_overall_position_rmse"],
            gr["model_overall_position_rmse"],
            gr["position_rmse_improvement_pct"],
        )
        if gr["position_rmse_improvement_pct"] > best_gap_improvement:
            best_gap_improvement = gr["position_rmse_improvement_pct"]
            best_gap_name = gap_name

    # FedAvg is the deployable privacy-preserving predictor. Centralized and
    # local checkpoints remain comparison references in gap_experiment_all_models.
    primary_gap = gap_all.get("federated", gap_all.get(best_gap_name, {}))
    all_results["gap_experiment"] = primary_gap
    all_results["gap_experiment_primary_model"] = "federated" if "federated" in gap_all else best_gap_name
    all_results["gap_experiment_all_models"] = gap_all
    all_results["gap_experiment_best_model"] = best_gap_name
    logger.info(
        "Best gap model: %s — Improvement: %.2f%%",
        best_gap_name, best_gap_improvement,
    )

    # ================================================================
    # 6. WEIGHT UPDATE SIZE
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: WEIGHT UPDATE SIZE")
    logger.info("=" * 70)
    weight_sizes = compute_weight_update_size(model_config)
    all_results["weight_update_size"] = weight_sizes
    logger.info(
        "Weight update: %d params, %.2f KB per update",
        weight_sizes["total_parameters"], weight_sizes["kb_per_update"],
    )

    # ================================================================
    # 7. INFERENCE LATENCY
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: INFERENCE LATENCY")
    logger.info("=" * 70)
    model, _, _ = load_checkpoint(federated_checkpoint_path)
    latency = measure_inference_latency(model, warmup_runs=50, timed_runs=200)
    all_results["inference_latency"] = latency
    logger.info(
        "Inference latency: %.3f ± %.3f ms (min=%.3f, max=%.3f)",
        latency["mean_ms"], latency["std_ms"],
        latency["min_ms"], latency["max_ms"],
    )

    # ================================================================
    # 8. SAVE ALL RESULTS
    # ================================================================
    results_path = output_dir_path / "experiment_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("All results saved to %s", results_path)

    # Also save convergence histories for plotting
    # Local loss histories
    for rsu_id, r in local_results.items():
        hist_path = output_dir_path / f"local_{rsu_id}_history.json"
        hist_path.write_text(
            json.dumps(r["history"], indent=2) + "\n", encoding="utf-8"
        )

    # Centralized loss history
    cent_hist_path = output_dir_path / "centralized_history.json"
    cent_hist_path.write_text(
        json.dumps(cent_results["history"], indent=2) + "\n", encoding="utf-8"
    )

    # Federated convergence history
    fed_hist_path = output_dir_path / "federated_history.json"
    fed_hist_path.write_text(
        json.dumps({
            "round_train_loss": fed_results["history"]["round_train_loss"],
            "round_val_loss": fed_results["history"]["round_val_loss"],
        }, indent=2) + "\n", encoding="utf-8"
    )

    # Print summary table
    _print_summary(all_results)

    return all_results


def _print_summary(results: dict) -> None:
    """Print a human-readable summary of experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Test RMSE comparison
    eval_data = results.get("evaluation", {})
    print("\n--- Position RMSE on Test Data (meters) ---")
    print(f"{'Model':<25} {'K=1':<10} {'K=2':<10} {'K=3':<10}")
    print("-" * 55)

    # Local (average across RSUs)
    local_own = eval_data.get("local_own_zone", {})
    if local_own:
        pos_rmses = [local_own[r]["position_rmse"] for r in local_own]
        avg_rmse = np.mean(pos_rmses, axis=0)
        print(f"{'Local (avg own zone)':<25} {avg_rmse[0]:<10.4f} {avg_rmse[1]:<10.4f} {avg_rmse[2]:<10.4f}")

    fed_all = eval_data.get("federated_all", {})
    if fed_all:
        r = fed_all["position_rmse"]
        print(f"{'FedAvg (all zones)':<25} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f}")

    cent_all = eval_data.get("centralized_all", {})
    if cent_all:
        r = cent_all["position_rmse"]
        print(f"{'Centralized (all zones)':<25} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f}")

    # Speed RMSE
    print("\n--- Speed RMSE on Test Data (m/s) ---")
    print(f"{'Model':<25} {'K=1':<10} {'K=2':<10} {'K=3':<10}")
    print("-" * 55)

    if local_own:
        spd_rmses = [local_own[r]["speed_rmse"] for r in local_own]
        avg_spd = np.mean(spd_rmses, axis=0)
        print(f"{'Local (avg own zone)':<25} {avg_spd[0]:<10.4f} {avg_spd[1]:<10.4f} {avg_spd[2]:<10.4f}")

    if fed_all:
        r = fed_all["speed_rmse"]
        print(f"{'FedAvg (all zones)':<25} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f}")

    if cent_all:
        r = cent_all["speed_rmse"]
        print(f"{'Centralized (all zones)':<25} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f}")

    # Gap experiment
    gap = results.get("gap_experiment", {})
    if gap:
        print(f"\n--- Gap Experiment: {gap.get('gap_duration_ms', 0)} ms evaluated of {gap.get('gap_duration_requested_ms', 0)} ms requested ({gap.get('num_test_samples', 0)} eligible windows) ---")
        primary_name = results.get("gap_experiment_primary_model", "unknown")
        best_name = results.get("gap_experiment_best_model", "unknown")
        print(f"Primary model: {primary_name}; best comparison: {best_name}")
        freeze_end = gap["freeze_position_rmse_per_step"][-1]
        model_end = gap["model_position_rmse_per_step"][-1]
        endpoint_improvement = gap.get(
            "position_rmse_improvement_at_gap_end_pct",
            (freeze_end - model_end) / freeze_end * 100,
        )
        print(f"K=5 / 500 ms freeze position RMSE: {freeze_end:.4f} m")
        print(f"K=5 / 500 ms model position RMSE: {model_end:.4f} m")
        print(f"K=5 / 500 ms improvement over freeze: {endpoint_improvement:.2f}%")
        print(f"Five-horizon aggregate RMSE: freeze={gap['freeze_overall_position_rmse']:.4f} m, "
              f"model={gap['model_overall_position_rmse']:.4f} m "
              f"(improvement={gap['position_rmse_improvement_pct']:.2f}%)")
        # Print all models if available
        gap_all = results.get("gap_experiment_all_models", {})
        if gap_all:
            print("\n  Per-model gap results:")
            for mname, gr in gap_all.items():
                freeze_end = gr["freeze_position_rmse_per_step"][-1]
                model_end = gr["model_position_rmse_per_step"][-1]
                endpoint_improvement = gr.get(
                    "position_rmse_improvement_at_gap_end_pct",
                    (freeze_end - model_end) / freeze_end * 100,
                )
                print(f"  {mname:20s}  K=5 freeze={freeze_end:.4f}m  "
                      f"model={model_end:.4f}m  improvement={endpoint_improvement:.2f}%")

    # Weight update size
    ws = results.get("weight_update_size", {})
    if ws:
        print(f"\n--- Weight Update Size ---")
        print(f"Parameters: {ws['total_parameters']}")
        print(f"Per update: {ws['kb_per_update']} KB")

    # Latency
    lat = results.get("inference_latency", {})
    if lat:
        print(f"\n--- Inference Latency ---")
        print(f"Mean: {lat['mean_ms']:.3f} ms ± {lat['std_ms']:.3f} ms")

    print("=" * 70)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiments: local, federated, centralized",
    )
    parser.add_argument("--data-dir", type=str, default="data/rsu")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs for local and centralized training")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of federated rounds")
    parser.add_argument("--local-epochs-per-round", type=int, default=1,
                        help="Local epochs per federated round")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_full_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        rounds=args.rounds,
        local_epochs_per_round=args.local_epochs_per_round,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )

