# RSU-Federated Learning for Vehicle State Prediction

## Digital Twin-Assisted Vehicular Edge Computing

A complete implementation of RSU-federated vehicle state prediction using GRU neural networks, Federated Learning (FedAvg), and the NGSIM US-101 public trajectory dataset.

## Project Overview

In Digital Twin-assisted Vehicular Edge Computing (DT-VEC), each RSU maintains a virtual replica of vehicles in its zone. When telemetry gaps occur (due to wireless volatility or bandwidth constraints), the digital twin becomes stale. This project deploys a GRU-based prediction model at each RSU that forecasts vehicle states (position, speed, heading) during these gaps.

**Key Innovation**: Each RSU client trains on its own partition. The GRU predicts position displacement from the last observed state, reducing the effect of different absolute road coordinates across RSUs. Flower FedAvg aggregates model updates using fixed, public physical scales rather than pooled normalization statistics. Each simulated RSU runs in a separate child process, loads only its assigned partition, and returns model parameters and scalar metrics to the coordinator. The centralized baseline remains privacy-unsafe. This is a same-host process simulation, not hardened OS-level isolation or a deployment security guarantee.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Base Station (Aggregator)                 │
│                     FedAvg Strategy                         │
│              Aggregates weight updates only                 │
└───────┬──────────────┬──────────────┬───────────────────────┘
        │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │  RSU 1  │    │  RSU 2  │    │  RSU 3  │
   │ 0-227m  │    │ 227-454m│    │ 454-682m│
   │ GRU     │    │ GRU     │    │ GRU     │
   │ Model   │    │ Model   │    │ Model   │
   └─────────┘    └─────────┘    └─────────┘
   Local Data      Local Data     Local Data
   (private)       (private)      (private)
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset Setup

The project uses the NGSIM US-101 public trajectory dataset. Place the `processed data.zip` in the project root, then:

```bash
python -c "import zipfile; zipfile.ZipFile('processed data.zip').extractall('extracted_data')"
# Copy to expected location:
# Windows: xcopy /E /I "extracted_data\processed data\data" data
# Linux/Mac: cp -r "extracted_data/processed data/data" data/
```

### Run All Experiments

```bash
python run_experiments.py --data-dir data/rsu --epochs 10 --rounds 20 --batch-size 1024
```

### Run Individual Training Modes

```bash
# Local-only baseline (3 independent RSU models)
python -m src.local_training --data-dir data/rsu --epochs 10

# Federated training (FedAvg with 3 RSU clients)
python -m src.federated_training --data-dir data/rsu --rounds 20

# Centralized baseline (privacy-unsafe upper bound)
python -m src.centralized_training --data-dir data/rsu --epochs 10
```

### Generate Plots

```bash
python generate_plots.py --results-dir results --output-dir results/plots
```

### Run Tests

```bash
python -m unittest discover -s tests -v
```

## Project Structure

```
CC-project/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Person 1: NGSIM data pipeline, RSU partitioning
│   ├── model.py                # Person 2: GRU model architecture
│   ├── train_utils.py          # Person 2: Training/eval utilities, normalization
│   ├── predictor.py            # Person 2: Inference API (RSUStatePredictor)
│   ├── rsu_client.py           # Person 3: Flower RSU client wrapper
│   ├── base_station.py         # Person 3: FedAvg aggregation strategy
│   ├── federated_training.py   # Person 3: Federated training orchestration (Flower)
│   ├── federated_manual.py     # Integration: Manual FedAvg (no ray dependency)
│   ├── local_training.py       # Person 4: Local-only baseline training
│   ├── centralized_training.py # Person 4: Centralized baseline (privacy-unsafe)
│   └── checkpoints.py          # Person 4: Evaluation and comparison utilities
├── tests/
│   ├── test_preprocessing.py   # Person 1 tests
│   ├── test_model.py           # Person 2 tests
│   ├── test_federated.py       # Person 3 tests
│   └── test_local_centralized.py # Person 4 tests
├── docs/
│   └── DATASET.md              # Dataset documentation and provenance
├── run_experiments.py          # Full experiment pipeline
├── generate_plots.py           # Plot generation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Technical Details

### Model Architecture

- **Type**: GRU (Gated Recurrent Unit) with single-shot decoder
- **Input**: `[B, 5, 4]` — 5 timesteps × 4 features (x_m, y_m, speed_mps, heading_rad)
- **Output**: `[B, 3, 4]` — 3 predicted future timesteps × 4 features
- **Hidden size**: 64
- **Parameters**: 18,380 (lightweight for edge deployment)
- **Position encoding**: input and target positions are relative to the last observed position; the inference API restores absolute coordinates
- **Loss**: MSE on features normalized with fixed physical scales; scales are not computed by pooling RSU samples

### Dataset

- **Source**: NGSIM US-101 vehicle trajectories (public, CC BY-SA 3.0)
- **Vehicles**: 6,101 unique trajectories across 3 recording periods
- **Sampling**: 100 ms (10 Hz) intervals
- **RSU partitioning**: 3 longitudinal zones by y-coordinate
  - RSU 1: 0–227.2 m (~1M train windows)
  - RSU 2: 227.2–454.4 m (~1M train windows)
  - RSU 3: 454.4–681.6 m (~772K train windows)

### Training Regimes

| Regime | Description | Privacy |
|--------|-------------|---------|
| **Local-only** | Each RSU trains independently on its own data | ✅ Private |
| **FedAvg** | Each RSU child process loads only its partition and sends model updates to the Base Station | Raw arrays remain in each worker; same-host OS isolation is limited |
| **Centralized** | All data pooled at one server | ❌ Unsafe |

### Evaluation Metrics

1. **Position RMSE** (meters) at K=1 (100ms), K=2 (200ms), K=3 (300ms)
2. **Speed RMSE** (m/s) at each horizon
3. **Convergence curves** — loss vs. epoch/round
4. **Weight update size** per FL round (KB)
5. **500 ms gap experiment** — freeze baseline vs. model prediction
6. **Inference latency** on CPU

## Inference API

The `RSUStatePredictor` module is self-contained and importable with no dependencies beyond PyTorch:

```python
from src.predictor import RSUStatePredictor
import numpy as np

# Load trained model
predictor = RSUStatePredictor.from_checkpoint("models/fedavg_global.pt")

# Predict from 5-step history
history = np.array([
    [x1, y1, speed1, heading1],
    [x2, y2, speed2, heading2],
    [x3, y3, speed3, heading3],
    [x4, y4, speed4, heading4],
    [x5, y5, speed5, heading5],
], dtype=np.float32)  # [5, 4]

future = predictor.predict(history)  # [3, 4]
# Returns: 3 predicted states (x, y, speed, heading)

# Recursive prediction for longer gaps (e.g., 500 ms = 5 steps)
future_5 = predictor.predict_recursive(history, total_steps=5)  # [5, 4]
```

## Experimental Results

### Run configuration

The reported run uses NGSIM US-101, vehicle-disjoint train/validation/test partitions, a 64-unit GRU, relative-position encoding, and fixed public physical scales. Local-only and centralized models train for 10 epochs. FedAvg runs 20 rounds with one local epoch per RSU per round. Batch size is 1,024 and seed is 42. The centralized baseline uses all 2,813,941 training windows. Test RMSE is reported in physical units.

### Test prediction accuracy

#### Position RMSE (meters)

| Training regime | K=1 (100 ms) | K=2 (200 ms) | K=3 (300 ms) |
|-----------------|--------------:|--------------:|--------------:|
| Local-only (mean across own RSU zones) | 0.0199 | 0.0345 | 0.0599 |
| FedAvg (all RSUs) | 0.0141 | 0.0305 | 0.0526 |
| Centralized (pooled reference) | **0.0130** | **0.0289** | **0.0486** |

#### Speed RMSE (m/s)

| Training regime | K=1 (100 ms) | K=2 (200 ms) | K=3 (300 ms) |
|-----------------|--------------:|--------------:|--------------:|
| Local-only (mean across own RSU zones) | 0.0788 | 0.1838 | 0.3226 |
| FedAvg (all RSUs) | 0.0624 | 0.1562 | 0.2865 |
| Centralized (pooled reference) | **0.0572** | **0.1472** | **0.2740** |

The FedAvg values above are from the final process-isolated checkpoint. Each regime is evaluated on the same vehicle-disjoint test split. The position and speed comparison figures are in `results/plots/position_rmse_comparison.png` and `results/plots/speed_rmse_comparison.png`.

### Full 500 ms telemetry-gap evaluation

The predictor produces five future states recursively: it predicts the first three steps, then feeds predictions back to forecast steps four and five. Freeze means the RSU holds the last observed state at every future step. Ground truth is joined by recording, vehicle, timestamp, RSU, and test split. The FedAvg and centralized models each have 594,371 eligible test windows; each local model uses only its own RSU zone.

| Horizon | Freeze position RMSE | FedAvg position RMSE | Freeze speed RMSE | FedAvg speed RMSE |
|---------|---------------------:|---------------------:|------------------:|------------------:|
| 100 ms | 1.017 m | 0.014 m | 0.162 m/s | 0.062 m/s |
| 200 ms | 2.035 m | 0.030 m | 0.312 m/s | 0.156 m/s |
| 300 ms | 3.053 m | 0.052 m | 0.443 m/s | 0.287 m/s |
| 400 ms | 4.072 m | 0.088 m | 0.552 m/s | 0.429 m/s |
| 500 ms (K=5) | 5.092 m | 0.138 m | 0.642 m/s | 0.550 m/s |

At the 500 ms endpoint, FedAvg position RMSE is 0.138 m compared with 5.092 m for freeze, a **97.30% reduction**. The five-horizon aggregate RMSE is 0.078 m for FedAvg and 3.377 m for freeze, a **97.68% reduction**. The aggregate is calculated across all five forecast steps and is distinct from the K=5 endpoint.

| Model evaluated over its eligible RSU test windows | Eligible windows | Position RMSE at 500 ms | Aggregate position RMSE across five steps |
|-----------------------------------------------------|-----------------:|-------------------------:|-------------------------------------------:|
| FedAvg | 594,371 | 0.138 m | 0.078 m |
| Centralized | 594,371 | 0.133 m | 0.075 m |
| Local RSU 1 | 220,468 | 0.143 m | 0.083 m |
| Local RSU 2 | 212,016 | 0.141 m | 0.081 m |
| Local RSU 3 | 161,887 | 0.174 m | 0.100 m |

The per-model table reports both the actual 500 ms endpoint and the aggregate over the five horizons. The corresponding figures are `results/plots/gap_experiment.png` and `results/plots/gap_experiment_all_models.png`.

### Convergence and communication

The convergence figure reports normalized-feature RMSE (the square root of normalized MSE). Local and centralized curves use epochs; FedAvg uses rounds, with one local epoch per RSU per round. The run used 10 local/centralized epochs and 20 FedAvg rounds.

| Communication measure | Payload per round |
|------------------------|------------------:|
| One RSU upload | 71.8 KiB |
| Uploads from all three RSUs | 215.4 KiB |
| Three uploads plus three broadcasts | 430.8 KiB |

These values are the float32 model payload size (18,380 parameters), repeated each round. The plot is `results/plots/weight_update_size.png`.

### Inference latency

On CPU, one three-step prediction averages 0.505 ms (standard deviation 0.085 ms over 200 timed runs, after 50 warm-up runs).

## Discussion

Centralized training has the lowest position and speed RMSE at each tested horizon, and serves as the pooled, privacy-unsafe reference. FedAvg improves on the mean local-only results at all three horizons for both position and speed.

FedAvg also improves K=3 position RMSE over the local model in each RSU zone:

| RSU zone | Local-only position RMSE | FedAvg position RMSE | Centralized position RMSE |
|----------|-------------------------:|---------------------:|---------------------------:|
| RSU 1 | 0.0545 m | 0.0509 m | 0.0484 m |
| RSU 2 | 0.0548 m | 0.0477 m | 0.0436 m |
| RSU 3 | 0.0705 m | 0.0604 m | 0.0546 m |

Across the full gap, FedAvg reduces position error substantially compared with freezing the twin state. Centralized remains slightly more accurate than FedAvg, while FedAvg improves over each local-only model at the 500 ms endpoint. As a sanity reference, constant-velocity extrapolation on the same 599,807 test windows has position RMSE 0.0225 m, 0.0610 m, and 0.1123 m at K=1, 2, and 3; FedAvg is lower at each horizon. These results use smooth 100 ms NGSIM trajectories, so the low absolute errors should be interpreted in that dataset context.

The FedAvg coordinator receives model parameters and scalar metrics from three separate RSU child processes; raw training and validation arrays remain in each worker. The processes share one host and OS account, so this simulation does not provide hardened filesystem isolation. The centralized baseline is intentionally privacy-unsafe.

## Limitations

- Heading is estimated from position displacement (no sensor heading available)
- Heading is treated as a raw radian value (no wraparound handling at ±π)
- NGSIM data has known measurement noise in position values
- FedAvg is standard averaging without formal differential privacy guarantees
- Recursive prediction beyond 3 steps can accumulate error; report the measured five-step result with its evaluation setup
- RSU clients run in separate child processes, but share the host OS account and filesystem permissions; use separate machines or hardened containers and access controls for deployment isolation
- Evaluated on CPU only (no GPU benchmarks)

## Team Contributions

| Person | Responsibility | Key Files |
|--------|---------------|-----------|
| Person 1 | Dataset pipeline & RSU partitioning | `preprocessing.py`, `test_preprocessing.py` |
| Person 2 | GRU model, training utils, inference API | `model.py`, `train_utils.py`, `predictor.py` |
| Person 3 | Federated learning orchestration | `rsu_client.py`, `base_station.py`, `federated_training.py` |
| Person 4 | Baselines, evaluation & integration | `local_training.py`, `centralized_training.py`, `checkpoints.py` |

## References

1. U.S. DOT, *NGSIM Open Data*, https://catalog.data.gov/dataset/next-generation-simulation-ngsim-open-data
2. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
3. Beutel et al., *Flower: A Friendly Federated Learning Framework*, arXiv 2020
