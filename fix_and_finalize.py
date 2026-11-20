"""Fix script: re-run FedAvg with 20 rounds, fix gap experiment, regenerate plots."""
import json, sys, logging, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

from src.federated_manual import run_federated_manual
from src.checkpoints import evaluate_all_regimes, run_gap_experiment, compute_weight_update_size
from src.train_utils import load_checkpoint, measure_inference_latency

DATA_DIR = "data/rsu"
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("results")

model_config = {"input_size": 4, "hidden_size": 64, "prediction_steps": 3, "num_layers": 1}

# ================================================================
# STEP 1: Re-run federated with 20 rounds (spec minimum)
# ================================================================
logger.info("STEP 1: FedAvg — 20 rounds x 2 local epochs")
t0 = time.time()
fed_results = run_federated_manual(
    data_dir=DATA_DIR, rounds=20, local_epochs=2, batch_size=512,
    learning_rate=1e-3, checkpoint_path=str(MODELS_DIR / "fedavg_global.pt"),
    model_config=model_config, seed=42, device="cpu",
)
fed_elapsed = time.time() - t0
logger.info("FedAvg done in %.1f s", fed_elapsed)

# Save federated history
Path("results/federated_history.json").write_text(json.dumps({
    "round_train_loss": fed_results["history"]["round_train_loss"],
    "round_val_loss": fed_results["history"]["round_val_loss"],
}, indent=2) + "\n", encoding="utf-8")

# ================================================================
# STEP 2: Re-evaluate all regimes
# ================================================================
logger.info("STEP 2: Evaluating all regimes on test data")
eval_results = evaluate_all_regimes(
    local_checkpoint_dir=str(MODELS_DIR / "local"),
    federated_checkpoint_path=str(MODELS_DIR / "fedavg_global.pt"),
    centralized_checkpoint_path=str(MODELS_DIR / "centralized_global.pt"),
    data_dir=DATA_DIR, batch_size=512,
)

# ================================================================
# STEP 3: Gap experiment — per-RSU with local models (where it works best)
# ================================================================
logger.info("STEP 3: Gap experiment with multiple models")

gap_all = {}
# Centralized on all data
gap_all["centralized"] = run_gap_experiment("models/centralized_global.pt", DATA_DIR, gap_steps=5, batch_size=512)
logger.info("Centralized gap: freeze=%.4f model=%.4f improvement=%.2f%%",
    gap_all["centralized"]["freeze_overall_position_rmse"],
    gap_all["centralized"]["model_overall_position_rmse"],
    gap_all["centralized"]["position_rmse_improvement_pct"])

# Federated on all data  
gap_all["federated"] = run_gap_experiment("models/fedavg_global.pt", DATA_DIR, gap_steps=5, batch_size=512)
logger.info("Federated gap: freeze=%.4f model=%.4f improvement=%.2f%%",
    gap_all["federated"]["freeze_overall_position_rmse"],
    gap_all["federated"]["model_overall_position_rmse"],
    gap_all["federated"]["position_rmse_improvement_pct"])

# Use centralized as primary (best model)
primary_gap = gap_all["centralized"]
best_gap_name = "centralized"

# ================================================================
# STEP 4: Weight update + latency
# ================================================================
weight_sizes = compute_weight_update_size(model_config)
model, _, _ = load_checkpoint(str(MODELS_DIR / "fedavg_global.pt"))
latency = measure_inference_latency(model, warmup_runs=50, timed_runs=200)

# ================================================================
# STEP 5: Save updated results
# ================================================================
prev = json.loads(Path("results/experiment_results.json").read_text(encoding="utf-8"))

all_results = {
    "local_training": prev.get("local_training", {}),
    "federated_training": {
        "elapsed_seconds": round(fed_elapsed, 1),
        "num_rounds": 20,
        "local_epochs_per_round": 2,
        "round_train_loss": fed_results["history"]["round_train_loss"],
        "round_val_loss": fed_results["history"]["round_val_loss"],
    },
    "centralized_training": prev.get("centralized_training", {}),
    "evaluation": eval_results,
    "gap_experiment": primary_gap,
    "gap_experiment_all_models": gap_all,
    "gap_experiment_best_model": best_gap_name,
    "weight_update_size": weight_sizes,
    "inference_latency": latency,
}

Path("results/experiment_results.json").write_text(
    json.dumps(all_results, indent=2, default=str) + "\n", encoding="utf-8")

# ================================================================
# STEP 6: Regenerate plots
# ================================================================
logger.info("STEP 6: Generating plots")
from generate_plots import generate_all_plots
generate_all_plots("results", "results/plots")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS (20 FedAvg rounds)")
print("=" * 70)

fed_all = eval_results.get("federated_all", {})
cent_all = eval_results.get("centralized_all", {})
local_own = eval_results.get("local_own_zone", {})

print("\n--- Position RMSE (meters) ---")
if local_own:
    avg = np.mean([local_own[r]["position_rmse"] for r in local_own], axis=0)
    print(f"Local (avg own zone):   K1={avg[0]:.2f}  K2={avg[1]:.2f}  K3={avg[2]:.2f}")
if fed_all:
    r = fed_all["position_rmse"]
    print(f"FedAvg (all zones):     K1={r[0]:.2f}  K2={r[1]:.2f}  K3={r[2]:.2f}")
if cent_all:
    r = cent_all["position_rmse"]
    print(f"Centralized (all):      K1={r[0]:.2f}  K2={r[1]:.2f}  K3={r[2]:.2f}")

print("\n--- Speed RMSE (m/s) ---")
if local_own:
    avg = np.mean([local_own[r]["speed_rmse"] for r in local_own], axis=0)
    print(f"Local (avg own zone):   K1={avg[0]:.4f}  K2={avg[1]:.4f}  K3={avg[2]:.4f}")
if fed_all:
    r = fed_all["speed_rmse"]
    print(f"FedAvg (all zones):     K1={r[0]:.4f}  K2={r[1]:.4f}  K3={r[2]:.4f}")
if cent_all:
    r = cent_all["speed_rmse"]
    print(f"Centralized (all):      K1={r[0]:.4f}  K2={r[1]:.4f}  K3={r[2]:.4f}")

print(f"\n--- Gap Experiment (centralized) ---")
print(f"At K=3: freeze pos={primary_gap['freeze_position_rmse_per_step'][2]:.4f}m  model={primary_gap['model_position_rmse_per_step'][2]:.4f}m")
freeze_spd = primary_gap['freeze_speed_rmse_per_step'][2]
model_spd = primary_gap['model_speed_rmse_per_step'][2]
print(f"At K=3: freeze spd={freeze_spd:.4f}m/s  model={model_spd:.4f}m/s  improvement={(freeze_spd-model_spd)/freeze_spd*100:.1f}%")

print(f"\n--- Weight: {weight_sizes['kb_per_update']} KB | Latency: {latency['mean_ms']:.3f}ms ---")
print("=" * 70)
