"""Tests for Person 4's local-only and centralized baseline modules.

Uses small synthetic data. Does not require real NGSIM data, Flower,
Pandas, or Matplotlib.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.model import VehicleStateGRU
from src.train_utils import (
    NormalizationStats,
    load_checkpoint,
    save_checkpoint,
)
from src.predictor import RSUStatePredictor
from src.local_training import (
    DEFAULT_MODEL_CONFIG as LOCAL_DEFAULT_CONFIG,
    RSU_IDS,
    run_local_training,
    train_local_model,
    parse_args as local_parse_args,
)
from src.centralized_training import (
    run_centralized_training,
    parse_args as centralized_parse_args,
)
from src.checkpoints import (
    compute_weight_update_size,
    evaluate_checkpoint_on_test,
    run_gap_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rsu_data(
    root: Path,
    rsu_ids: tuple[str, ...] = RSU_IDS,
    n_train: int = 40,
    n_val: int = 16,
    n_test: int = 16,
    seed: int = 42,
) -> None:
    """Write small .npy files mimicking RSU partition layout."""
    rng = np.random.RandomState(seed)
    for rsu_id in rsu_ids:
        rsu_dir = root / rsu_id
        rsu_dir.mkdir(parents=True, exist_ok=True)
        for split, n in [("train", n_train), ("validation", n_val), ("test", n_test)]:
            X = (rng.randn(n, 5, 4) + 10.0).astype(np.float32)
            y = (rng.randn(n, 3, 4) + 10.0).astype(np.float32)
            np.save(rsu_dir / f"{split}_X.npy", X)
            np.save(rsu_dir / f"{split}_y.npy", y)


# ---------------------------------------------------------------------------
# Local training tests
# ---------------------------------------------------------------------------

class TestLocalTraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name) / "rsu"
        _write_rsu_data(self.data_dir, n_train=24, n_val=8, n_test=8)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_train_local_model_returns_model_and_history(self):
        """Train one RSU's local model and verify output."""
        X, y = [], []
        for rsu_id in RSU_IDS:
            Xr, yr = np.load(self.data_dir / rsu_id / "train_X.npy"), np.load(self.data_dir / rsu_id / "train_y.npy")
            X.append(Xr); y.append(yr)
        norm_stats = NormalizationStats.from_training_data(
            np.concatenate(X), np.concatenate(y)
        )
        model, history = train_local_model(
            rsu_id="rsu_1",
            norm_stats=norm_stats,
            data_dir=str(self.data_dir),
            epochs=2,
            batch_size=8,
            seed=42,
        )
        self.assertIsInstance(model, VehicleStateGRU)
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertTrue(all(np.isfinite(v) for v in history["train_loss"]))

    def test_run_local_training_produces_checkpoints(self):
        """Full local training produces 3 checkpoints and metrics."""
        ckpt_dir = Path(self.tmpdir.name) / "models" / "local"
        results = run_local_training(
            data_dir=str(self.data_dir),
            epochs=2,
            batch_size=8,
            checkpoint_dir=str(ckpt_dir),
            seed=42,
        )
        self.assertEqual(set(results.keys()), set(RSU_IDS))

        for rsu_id in RSU_IDS:
            ckpt_path = ckpt_dir / f"local_{rsu_id}.pt"
            self.assertTrue(ckpt_path.exists())
            model, stats, meta = load_checkpoint(ckpt_path)
            self.assertEqual(meta["training_mode"], "local")
            self.assertEqual(meta["rsu_id"], rsu_id)

        # Metrics JSON
        metrics_path = ckpt_dir / "local_metrics.json"
        self.assertTrue(metrics_path.exists())
        metrics = json.loads(metrics_path.read_text())
        self.assertEqual(set(metrics.keys()), set(RSU_IDS))

    def test_local_checkpoints_loadable_by_predictor(self):
        """RSUStatePredictor can load local checkpoints."""
        ckpt_dir = Path(self.tmpdir.name) / "models" / "local"
        run_local_training(
            data_dir=str(self.data_dir),
            epochs=1,
            batch_size=8,
            checkpoint_dir=str(ckpt_dir),
            seed=0,
        )
        for rsu_id in RSU_IDS:
            predictor = RSUStatePredictor.from_checkpoint(
                ckpt_dir / f"local_{rsu_id}.pt"
            )
            history = np.random.randn(5, 4).astype(np.float32)
            result = predictor.predict(history)
            self.assertEqual(result.shape, (3, 4))


# ---------------------------------------------------------------------------
# Centralized training tests
# ---------------------------------------------------------------------------

class TestCentralizedTraining(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name) / "rsu"
        _write_rsu_data(self.data_dir, n_train=24, n_val=8, n_test=8)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_centralized_training_produces_checkpoint(self):
        ckpt_path = Path(self.tmpdir.name) / "models" / "centralized.pt"
        result = run_centralized_training(
            data_dir=str(self.data_dir),
            epochs=2,
            batch_size=8,
            checkpoint_path=str(ckpt_path),
            seed=42,
        )
        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertEqual(result["total_train_samples"], 24 * 3)

        model, stats, meta = load_checkpoint(result["checkpoint_path"])
        self.assertEqual(meta["training_mode"], "centralized")
        self.assertIn("privacy_note", meta)

    def test_centralized_checkpoint_loadable_by_predictor(self):
        ckpt_path = Path(self.tmpdir.name) / "models" / "centralized.pt"
        run_centralized_training(
            data_dir=str(self.data_dir),
            epochs=1,
            batch_size=8,
            checkpoint_path=str(ckpt_path),
            seed=0,
        )
        predictor = RSUStatePredictor.from_checkpoint(ckpt_path)
        result = predictor.predict(np.random.randn(5, 4).astype(np.float32))
        self.assertEqual(result.shape, (3, 4))


# ---------------------------------------------------------------------------
# Checkpoint utilities tests
# ---------------------------------------------------------------------------

class TestCheckpointUtils(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name) / "rsu"
        _write_rsu_data(self.data_dir, n_train=24, n_val=8, n_test=16)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_evaluate_checkpoint_on_test(self):
        """Evaluate a checkpoint and get per-horizon RMSE."""
        ckpt_path = Path(self.tmpdir.name) / "test.pt"
        X_all, y_all = [], []
        for rsu_id in RSU_IDS:
            X, y = np.load(self.data_dir / rsu_id / "train_X.npy"), np.load(self.data_dir / rsu_id / "train_y.npy")
            X_all.append(X); y_all.append(y)
        norm_stats = NormalizationStats.from_training_data(
            np.concatenate(X_all), np.concatenate(y_all)
        )
        model = VehicleStateGRU()
        save_checkpoint(model, norm_stats, ckpt_path, {"training_mode": "test"})

        result = evaluate_checkpoint_on_test(
            ckpt_path, str(self.data_dir), batch_size=8
        )
        self.assertEqual(len(result["position_rmse"]), 3)
        self.assertEqual(len(result["speed_rmse"]), 3)
        self.assertTrue(all(v > 0 for v in result["position_rmse"]))

    def test_weight_update_size(self):
        sizes = compute_weight_update_size()
        self.assertGreater(sizes["total_parameters"], 0)
        self.assertGreater(sizes["kb_per_update"], 0)
        self.assertEqual(sizes["dtype"], "float32")

    def test_gap_experiment(self):
        """Run gap experiment on a dummy checkpoint."""
        ckpt_path = Path(self.tmpdir.name) / "gap_test.pt"
        X_all, y_all = [], []
        for rsu_id in RSU_IDS:
            X, y = np.load(self.data_dir / rsu_id / "train_X.npy"), np.load(self.data_dir / rsu_id / "train_y.npy")
            X_all.append(X); y_all.append(y)
        norm_stats = NormalizationStats.from_training_data(
            np.concatenate(X_all), np.concatenate(y_all)
        )
        model = VehicleStateGRU()
        save_checkpoint(model, norm_stats, ckpt_path, {"training_mode": "test"})

        result = run_gap_experiment(ckpt_path, str(self.data_dir), batch_size=8)
        self.assertEqual(result["gap_steps_evaluated"], 3)
        self.assertEqual(len(result["freeze_position_rmse_per_step"]), 3)
        self.assertEqual(len(result["model_position_rmse_per_step"]), 3)
        self.assertGreater(result["freeze_overall_position_rmse"], 0)


# ---------------------------------------------------------------------------
# CLI parsing tests
# ---------------------------------------------------------------------------

class TestCLIParsing(unittest.TestCase):
    def test_local_defaults(self):
        args = local_parse_args([])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.seed, 42)

    def test_centralized_defaults(self):
        args = centralized_parse_args([])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.seed, 42)


if __name__ == "__main__":
    unittest.main()

