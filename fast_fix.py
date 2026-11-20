"""FAST fix: resume FedAvg from existing 10-round checkpoint, run 10 MORE rounds, then eval+plots."""
import json, sys, logging, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import OrderedDict
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

from src.model import VehicleStateGRU
from src.preprocessing import load_rsu_dataset
from src.train_utils import NormalizationStats, create_dataloader, train_one_epoch, evaluate, save_checkpoint, load_checkpoint, measure_inference_latency
from src.checkpoints import evaluate_all_regimes, run_gap_experiment, compute_weight_update_size

DATA_DIR = "data/rsu"
RSU_IDS = ("rsu_1", "rsu_2", "rsu_3")
CKPT = "models/fedavg_global.pt"
model_config = {"input_size": 4, "hidden_size": 64, "prediction_steps": 3, "num_layers": 1}

# Load existing checkpoint (10 rounds already done)
print("Loading existing 10-round FedAvg checkpoint...")
model, norm_stats, metadata = load_checkpoint(CKPT)

# Load existing history
prev_hist = json.loads(Path("results/federated_history.json").read_text(encoding="utf-8"))
prev_train = prev_hist["round_train_loss"]
prev_val = prev_hist["round_val_loss"]
print(f"Existing: {len(prev_train)} rounds, last val_loss={prev_val[-1]:.6f}")

# Prepare client data (normalized)
print("Loading client data...")
client_data = {}
for rsu_id in RSU_IDS:
    X_tr = norm_stats.normalize_input(np.array(load_rsu_dataset(rsu_id, "train", DATA_DIR)[0]))
    y_tr = norm_stats.normalize_target(np.array(load_rsu_dataset(rsu_id, "train", DATA_DIR)[1]))
    X_va = norm_stats.normalize_input(np.array(load_rsu_dataset(rsu_id, "validation", DATA_DIR)[0]))
    y_va = norm_stats.normalize_target(np.array(load_rsu_dataset(rsu_id, "validation", DATA_DIR)[1]))
    client_data[rsu_id] = {
        "train_loader": create_dataloader(X_tr, y_tr, batch_size=1024, shuffle=True),
        "val_loader": create_dataloader(X_va, y_va, batch_size=1024, shuffle=False),
        "num_train": len(X_tr),
    }
    print(f"  {rsu_id}: {len(X_tr)} train samples")

# Extract global params from loaded model
def get_params(m):
    return [v.cpu().detach().numpy().copy() for v in m.state_dict().values()]
def set_params(m, params):
    keys = list(m.state_dict().keys())
    m.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)}), strict=True)

global_params = get_params(model)

# Run 10 MORE rounds (rounds 11-20)
NEW_ROUNDS = 10
new_train_losses = []
new_val_losses = []

print(f"\nRunning {NEW_ROUNDS} more FedAvg rounds (11-20)...")
t0 = time.time()
for rnd in range(1, NEW_ROUNDS + 1):
    actual_round = len(prev_train) + rnd
    rnd_start = time.time()
    client_params_list = []
    client_weights = []
    client_tl = {}
    client_vl = {}
    
    for rsu_id in RSU_IDS:
        local_model = VehicleStateGRU.from_config(model_config)
        set_params(local_model, global_params)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        tl = train_one_epoch(local_model, client_data[rsu_id]["train_loader"], optimizer, criterion, "cpu")
        vl = evaluate(local_model, client_data[rsu_id]["val_loader"], criterion, "cpu")
        client_tl[rsu_id] = tl
        client_vl[rsu_id] = vl
        client_params_list.append(get_params(local_model))
        client_weights.append(client_data[rsu_id]["num_train"])
    
    # FedAvg aggregate
    total_w = sum(client_weights)
    num_layers = len(client_params_list[0])
    agg = []
    for li in range(num_layers):
        ws = np.zeros_like(client_params_list[0][li], dtype=np.float64)
        for ci, p in enumerate(client_params_list):
            ws += p[li].astype(np.float64) * client_weights[ci]
        agg.append((ws / total_w).astype(np.float32))
    global_params = agg
    
    avg_tl = sum(client_tl[r] * client_data[r]["num_train"] for r in RSU_IDS) / total_w
    avg_vl = sum(client_vl[r] * client_data[r]["num_train"] for r in RSU_IDS) / total_w
    new_train_losses.append(avg_tl)
    new_val_losses.append(avg_vl)
    
    elapsed = time.time() - rnd_start
    print(f"  Round {actual_round}/20  train={avg_tl:.6f}  val={avg_vl:.6f}  ({elapsed:.0f}s)")

fed_time = time.time() - t0
print(f"10 additional rounds done in {fed_time:.0f}s")

# Save updated model
set_params(model, global_params)
metadata["num_rounds"] = 20
metadata["elapsed_seconds"] = round(metadata.get("elapsed_seconds", 0) + fed_time, 1)
save_checkpoint(model, norm_stats, CKPT, metadata)

# Combine histories
all_train = prev_train + new_train_losses
all_val = prev_val + new_val_losses
Path("results/federated_history.json").write_text(json.dumps({
    "round_train_loss": all_train, "round_val_loss": all_val,
}, indent=2) + "\n", encoding="utf-8")
print(f"\nFull history: {len(all_train)} rounds saved")

# Re-evaluate
print("\nEvaluating all regimes...")
eval_results = evaluate_all_regimes(
    local_checkpoint_dir="models/local", federated_checkpoint_path=CKPT,
    centralized_checkpoint_path="models/centralized_global.pt",
    data_dir=DATA_DIR, batch_size=512)

# Gap experiment
print("\nGap experiment (centralized model)...")
gap_cent = run_gap_experiment("models/centralized_global.pt", DATA_DIR, gap_steps=5, batch_size=512)
print(f"  Centralized: freeze={gap_cent['freeze_overall_position_rmse']:.4f} model={gap_cent['model_overall_position_rmse']:.4f} imp={gap_cent['position_rmse_improvement_pct']:.2f}%")

gap_fed = run_gap_experiment(CKPT, DATA_DIR, gap_steps=5, batch_size=512)
print(f"  Federated:   freeze={gap_fed['freeze_overall_position_rmse']:.4f} model={gap_fed['model_overall_position_rmse']:.4f} imp={gap_fed['position_rmse_improvement_pct']:.2f}%")

# Weight + latency
weight_sizes = compute_weight_update_size(model_config)
latency = measure_inference_latency(model, warmup_runs=50, timed_runs=200)

# Save all results
prev = json.loads(Path("results/experiment_results.json").read_text(encoding="utf-8"))
all_results = {
    "local_training": prev.get("local_training", {}),
    "federated_training": {"elapsed_seconds": round(fed_time + 1243.3, 1), "num_rounds": 20, "local_epochs_per_round": 1, "round_train_loss": all_train, "round_val_loss": all_val},
    "centralized_training": prev.get("centralized_training", {}),
    "evaluation": eval_results,
    "gap_experiment": gap_cent,
    "gap_experiment_all_models": {"centralized": gap_cent, "federated": gap_fed},
    "gap_experiment_best_model": "centralized",
    "weight_update_size": weight_sizes,
    "inference_latency": latency,
}
Path("results/experiment_results.json").write_text(json.dumps(all_results, indent=2, default=str) + "\n", encoding="utf-8")

# Plots
print("\nGenerating plots...")
from generate_plots import generate_all_plots
generate_all_plots("results", "results/plots")

# Final summary
fed_all = eval_results.get("federated_all", {})
cent_all = eval_results.get("centralized_all", {})
local_own = eval_results.get("local_own_zone", {})
print("\n" + "=" * 60)
print("DONE! 20-round FedAvg results:")
if fed_all:
    r = fed_all["position_rmse"]
    print(f"  FedAvg position RMSE: K1={r[0]:.2f} K2={r[1]:.2f} K3={r[2]:.2f}")
    r = fed_all["speed_rmse"]
    print(f"  FedAvg speed RMSE:    K1={r[0]:.4f} K2={r[1]:.4f} K3={r[2]:.4f}")
print(f"  Gap (cent): {gap_cent['position_rmse_improvement_pct']:.2f}% position improvement")
for i in range(3):
    fs = gap_cent['freeze_speed_rmse_per_step'][i]
    ms = gap_cent['model_speed_rmse_per_step'][i]
    print(f"  Gap K={i+1}: speed freeze={fs:.4f} model={ms:.4f} improvement={(fs-ms)/fs*100:.1f}%")
print("=" * 60)
