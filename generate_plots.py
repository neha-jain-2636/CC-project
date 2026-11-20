"""Generate all plots and tables from experiment results.

Usage::

    python generate_plots.py --results-dir results --output-dir results/plots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_all_plots(
    results_dir: str = "results",
    output_dir: str = "results/plots",
) -> None:
    """Generate all plots from saved experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load main results
    main_results_file = results_path / "experiment_results.json"
    if not main_results_file.exists():
        logger.error("No experiment results found at %s", main_results_file)
        return
    results = json.loads(main_results_file.read_text(encoding="utf-8"))

    # ================================================================
    # PLOT 1: Position RMSE at K=1,2,3 for all regimes
    # ================================================================
    _plot_rmse_comparison(results, output_path, "position_rmse",
                          "Position RMSE by Prediction Horizon",
                          "Position RMSE (meters)")

    # ================================================================
    # PLOT 2: Speed RMSE at K=1,2,3 for all regimes
    # ================================================================
    _plot_rmse_comparison(results, output_path, "speed_rmse",
                          "Speed RMSE by Prediction Horizon",
                          "Speed RMSE (m/s)")

    # ================================================================
    # PLOT 3: Training convergence curves
    # ================================================================
    _plot_convergence(results_path, output_path)

    # ================================================================
    # PLOT 4: Gap experiment comparison
    # ================================================================
    _plot_gap_experiment(results, output_path)

    # ================================================================
    # PLOT 5: Per-RSU comparison
    # ================================================================
    _plot_per_rsu_comparison(results, output_path)

    logger.info("All plots saved to %s", output_path)


def _plot_rmse_comparison(
    results: dict,
    output_path: Path,
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    """Bar chart of RMSE at K=1,2,3 for local, federated, centralized."""
    import matplotlib.pyplot as plt

    eval_data = results.get("evaluation", {})
    horizons = ["K=1\n(100ms)", "K=2\n(200ms)", "K=3\n(300ms)"]
    x = np.arange(len(horizons))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Local (average across RSUs)
    local_own = eval_data.get("local_own_zone", {})
    if local_own:
        vals = [local_own[r][metric_key] for r in local_own]
        avg = np.mean(vals, axis=0)
        bars1 = ax.bar(x - width, avg, width, label="Local-only (avg)",
                       color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars1, avg):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # Federated
    fed_all = eval_data.get("federated_all", {})
    if fed_all:
        vals = fed_all[metric_key]
        bars2 = ax.bar(x, vals, width, label="FedAvg",
                       color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars2, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # Centralized
    cent_all = eval_data.get("centralized_all", {})
    if cent_all:
        vals = cent_all[metric_key]
        bars3 = ax.bar(x + width, vals, width, label="Centralized (privacy-unsafe)",
                       color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars3, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Prediction Horizon", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / f"{metric_key}_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path / f"{metric_key}_comparison.png")


def _plot_convergence(results_path: Path, output_path: Path) -> None:
    """Plot training loss convergence for all regimes."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE convergence curves (sqrt of MSE loss)
    colors = {"rsu_1": "#e74c3c", "rsu_2": "#e67e22", "rsu_3": "#f39c12"}

    for rsu_id in ["rsu_1", "rsu_2", "rsu_3"]:
        hist_path = results_path / f"local_{rsu_id}_history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            val_rmse = [v**0.5 for v in hist["val_loss"]]
            train_rmse = [v**0.5 for v in hist["train_loss"]]
            ax1.plot(range(1, len(val_rmse) + 1), val_rmse,
                     label=f"Local {rsu_id}", color=colors[rsu_id],
                     linestyle="--", alpha=0.7)
            ax2.plot(range(1, len(train_rmse) + 1), train_rmse,
                     label=f"Local {rsu_id}", color=colors[rsu_id],
                     linestyle="--", alpha=0.7)

    cent_path = results_path / "centralized_history.json"
    if cent_path.exists():
        hist = json.loads(cent_path.read_text(encoding="utf-8"))
        val_rmse = [v**0.5 for v in hist["val_loss"]]
        train_rmse = [v**0.5 for v in hist["train_loss"]]
        ax1.plot(range(1, len(val_rmse) + 1), val_rmse,
                 label="Centralized", color="#2ecc71", linewidth=2)
        ax2.plot(range(1, len(train_rmse) + 1), train_rmse,
                 label="Centralized", color="#2ecc71", linewidth=2)

    fed_path = results_path / "federated_history.json"
    if fed_path.exists():
        hist = json.loads(fed_path.read_text(encoding="utf-8"))
        val_rmse = [v**0.5 for v in hist["round_val_loss"]]
        train_rmse = [v**0.5 for v in hist["round_train_loss"]]
        ax1.plot(range(1, len(val_rmse) + 1), val_rmse,
                 label="FedAvg", color="#3498db", linewidth=2, marker="o", markersize=4)
        ax2.plot(range(1, len(train_rmse) + 1), train_rmse,
                 label="FedAvg", color="#3498db", linewidth=2, marker="o", markersize=4)

    ax1.set_xlabel("Epoch / Round")
    ax1.set_ylabel("Validation RMSE")
    ax1.set_title("Validation RMSE Convergence", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch / Round")
    ax2.set_ylabel("Training RMSE")
    ax2.set_title("Training RMSE Convergence", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "convergence_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved convergence curves")


def _plot_gap_experiment(results: dict, output_path: Path) -> None:
    """Plot the gap experiment: freeze vs. prediction per step."""
    import matplotlib.pyplot as plt

    gap = results.get("gap_experiment", {})
    if not gap:
        return

    steps = list(range(1, gap["gap_steps_evaluated"] + 1))
    step_labels = [f"{s * 100}ms" for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Position RMSE
    ax1.plot(steps, gap["freeze_position_rmse_per_step"], "ro-",
             label="Freeze (last state)", linewidth=2, markersize=8)
    ax1.plot(steps, gap["model_position_rmse_per_step"], "b^-",
             label="GRU Prediction", linewidth=2, markersize=8)
    ax1.set_xlabel("Prediction Step")
    ax1.set_ylabel("Position RMSE (meters)")
    ax1.set_title("Telemetry Gap: Position RMSE", fontweight="bold")
    ax1.set_xticks(steps)
    ax1.set_xticklabels(step_labels)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Speed RMSE
    ax2.plot(steps, gap["freeze_speed_rmse_per_step"], "ro-",
             label="Freeze (last state)", linewidth=2, markersize=8)
    ax2.plot(steps, gap["model_speed_rmse_per_step"], "b^-",
             label="GRU Prediction", linewidth=2, markersize=8)
    ax2.set_xlabel("Prediction Step")
    ax2.set_ylabel("Speed RMSE (m/s)")
    ax2.set_title("Telemetry Gap: Speed RMSE", fontweight="bold")
    ax2.set_xticks(steps)
    ax2.set_xticklabels(step_labels)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    best_name = results.get("gap_experiment_best_model", "model")
    fig.suptitle(
        f"Gap Experiment ({best_name}) — Position improvement: {gap['position_rmse_improvement_pct']:.1f}%",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path / "gap_experiment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also plot a comparison across all models if available
    gap_all = results.get("gap_experiment_all_models", {})
    if gap_all and len(gap_all) > 1:
        fig2, ax = plt.subplots(figsize=(10, 6))
        model_names = []
        freeze_vals = []
        model_vals = []
        for mname, gr in gap_all.items():
            model_names.append(mname)
            freeze_vals.append(gr["freeze_overall_position_rmse"])
            model_vals.append(gr["model_overall_position_rmse"])

        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, freeze_vals, width, label="Freeze Baseline",
               color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + width/2, model_vals, width, label="GRU Prediction",
               color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Model Checkpoint", fontsize=12)
        ax.set_ylabel("Position RMSE (meters)", fontsize=12)
        ax.set_title("Gap Experiment: All Models Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(output_path / "gap_experiment_all_models.png", dpi=150)
        plt.close(fig2)
        logger.info("Saved gap experiment all-models comparison plot")

    logger.info("Saved gap experiment plot")


def _plot_per_rsu_comparison(results: dict, output_path: Path) -> None:
    """Plot per-RSU RMSE for local vs. federated vs. centralized."""
    import matplotlib.pyplot as plt

    eval_data = results.get("evaluation", {})
    rsu_ids = ["rsu_1", "rsu_2", "rsu_3"]
    x = np.arange(len(rsu_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Local (each RSU's model on its own zone)
    local_own = eval_data.get("local_own_zone", {})
    if local_own:
        vals = [local_own[r]["position_rmse"][2] for r in rsu_ids if r in local_own]
        ax.bar(x - width, vals, width, label="Local-only",
               color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Federated
    fed_per = eval_data.get("federated_per_rsu", {})
    if fed_per:
        vals = [fed_per[r]["position_rmse"][2] for r in rsu_ids if r in fed_per]
        ax.bar(x, vals, width, label="FedAvg",
               color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Centralized
    cent_per = eval_data.get("centralized_per_rsu", {})
    if cent_per:
        vals = [cent_per[r]["position_rmse"][2] for r in rsu_ids if r in cent_per]
        ax.bar(x + width, vals, width, label="Centralized",
               color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("RSU Zone", fontsize=12)
    ax.set_ylabel("Position RMSE at K=3 (meters)", fontsize=12)
    ax.set_title("Per-RSU Position RMSE at K=3 (300 ms)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["RSU 1\n(0–227m)", "RSU 2\n(227–454m)", "RSU 3\n(454–682m)"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "per_rsu_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved per-RSU comparison plot")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/plots")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    generate_all_plots(args.results_dir, args.output_dir)

