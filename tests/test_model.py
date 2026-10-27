"""Focused tests for Person 2's model, training utilities, and inference API.

Uses small dummy tensors / synthetic data.  Does not require the real NGSIM
dataset, Flower, Pandas, or Matplotlib.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.model import VehicleStateGRU
from src.train_utils import (
    NormalizationStats,
    compute_per_horizon_rmse,
    create_dataloader,
    evaluate,
    load_checkpoint,
    measure_inference_latency,
    save_checkpoint,
    train_one_epoch,
)
from src.predictor import RSUStatePredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_data(
    n: int = 32,
    obs: int = 5,
    pred: int = 3,
    features: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small synthetic arrays for smoke tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, obs, features).astype(np.float32)
    y = rng.randn(n, pred, features).astype(np.float32)
    return X, y


def _dummy_stats() -> NormalizationStats:
    """Create normalization stats from dummy data."""
    X, y = _dummy_data(n=100)
    return NormalizationStats.from_training_data(X, y)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestVehicleStateGRU(unittest.TestCase):
    """Tests for the shared GRU model."""

    def test_output_shape_default(self):
        model = VehicleStateGRU()
        x = torch.randn(8, 5, 4)
        out = model(x)
        self.assertEqual(out.shape, (8, 3, 4))

    def test_output_shape_custom_hidden(self):
        model = VehicleStateGRU(hidden_size=32)
        x = torch.randn(4, 5, 4)
        out = model(x)
        self.assertEqual(out.shape, (4, 3, 4))

    def test_output_shape_custom_prediction_steps(self):
        model = VehicleStateGRU(prediction_steps=5)
        x = torch.randn(4, 5, 4)
        out = model(x)
        self.assertEqual(out.shape, (4, 5, 4))

    def test_single_sample_batch(self):
        model = VehicleStateGRU()
        x = torch.randn(1, 5, 4)
        out = model(x)
        self.assertEqual(out.shape, (1, 3, 4))

    def test_rejects_2d_input(self):
        model = VehicleStateGRU()
        x = torch.randn(5, 4)
        with self.assertRaisesRegex(ValueError, "3D"):
            model(x)

    def test_rejects_wrong_feature_count(self):
        model = VehicleStateGRU(input_size=4)
        x = torch.randn(2, 5, 6)
        with self.assertRaisesRegex(ValueError, "features"):
            model(x)

    def test_rejects_invalid_config(self):
        with self.assertRaises(ValueError):
            VehicleStateGRU(input_size=0)
        with self.assertRaises(ValueError):
            VehicleStateGRU(hidden_size=-1)
        with self.assertRaises(ValueError):
            VehicleStateGRU(prediction_steps=0)
        with self.assertRaises(ValueError):
            VehicleStateGRU(num_layers=0)

    def test_get_config_roundtrip(self):
        original = VehicleStateGRU(hidden_size=48, prediction_steps=5, num_layers=2)
        config = original.get_config()
        rebuilt = VehicleStateGRU.from_config(config)
        self.assertEqual(rebuilt.hidden_size, 48)
        self.assertEqual(rebuilt.prediction_steps, 5)
        self.assertEqual(rebuilt.num_layers, 2)

    def test_gradients_flow(self):
        model = VehicleStateGRU()
        x = torch.randn(4, 5, 4, requires_grad=False)
        y = torch.randn(4, 3, 4)
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.all(p.grad == 0))


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalizationStats(unittest.TestCase):
    """Tests for NormalizationStats."""

    def test_from_training_data_shapes(self):
        X, y = _dummy_data(n=50)
        stats = NormalizationStats.from_training_data(X, y)
        self.assertEqual(stats.input_mean.shape, (4,))
        self.assertEqual(stats.input_std.shape, (4,))
        self.assertEqual(stats.target_mean.shape, (4,))
        self.assertEqual(stats.target_std.shape, (4,))

    def test_normalize_and_inverse_roundtrip(self):
        X, y = _dummy_data(n=100)
        stats = NormalizationStats.from_training_data(X, y)
        y_norm = stats.normalize_target(y)
        y_reconstructed = stats.inverse_transform_target(y_norm)
        np.testing.assert_allclose(y_reconstructed, y, atol=1e-5)

    def test_normalize_input_output_shape(self):
        X, _ = _dummy_data()
        stats = _dummy_stats()
        X_norm = stats.normalize_input(X)
        self.assertEqual(X_norm.shape, X.shape)
        self.assertEqual(X_norm.dtype, np.float32)

    def test_serialization_roundtrip(self):
        stats = _dummy_stats()
        d = stats.to_dict()
        restored = NormalizationStats.from_dict(d)
        np.testing.assert_array_equal(stats.input_mean, restored.input_mean)
        np.testing.assert_array_equal(stats.input_std, restored.input_std)
        np.testing.assert_array_equal(stats.target_mean, restored.target_mean)
        np.testing.assert_array_equal(stats.target_std, restored.target_std)

    def test_rejects_near_zero_std(self):
        with self.assertRaisesRegex(ValueError, "near-zero"):
            NormalizationStats(
                input_mean=np.zeros(4, dtype=np.float32),
                input_std=np.array([1, 1, 0, 1], dtype=np.float32),
                target_mean=np.zeros(4, dtype=np.float32),
                target_std=np.ones(4, dtype=np.float32),
            )

    def test_inverse_transform_accepts_tensor(self):
        stats = _dummy_stats()
        y_norm = torch.randn(4, 3, 4)
        result = stats.inverse_transform_target(y_norm)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 3, 4))


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

class TestCreateDataloader(unittest.TestCase):
    def test_basic_creation(self):
        X, y = _dummy_data(n=20)
        dl = create_dataloader(X, y, batch_size=8, shuffle=False)
        batches = list(dl)
        self.assertEqual(len(batches), 3)  # 20 / 8 = 2.5 → 3 batches
        self.assertEqual(batches[0][0].shape, (8, 5, 4))
        self.assertEqual(batches[0][1].shape, (8, 3, 4))

    def test_rejects_mismatched_lengths(self):
        X = np.random.randn(10, 5, 4).astype(np.float32)
        y = np.random.randn(8, 3, 4).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "same number"):
            create_dataloader(X, y)

    def test_rejects_invalid_batch_size(self):
        X, y = _dummy_data()
        with self.assertRaisesRegex(ValueError, "positive"):
            create_dataloader(X, y, batch_size=0)


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------

class TestTraining(unittest.TestCase):
    def test_one_epoch_reduces_loss(self):
        """A small training step should produce a finite, positive loss."""
        X, y = _dummy_data(n=32)
        stats = NormalizationStats.from_training_data(X, y)
        X_norm = stats.normalize_input(X)
        y_norm = stats.normalize_target(y)

        model = VehicleStateGRU(hidden_size=32)
        dl = create_dataloader(X_norm, y_norm, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = train_one_epoch(model, dl, optimizer, criterion)
        self.assertGreater(loss, 0.0)
        self.assertTrue(np.isfinite(loss))

    def test_evaluate_returns_finite(self):
        X, y = _dummy_data(n=16)
        stats = NormalizationStats.from_training_data(X, y)
        model = VehicleStateGRU(hidden_size=32)
        dl = create_dataloader(
            stats.normalize_input(X),
            stats.normalize_target(y),
            batch_size=8,
            shuffle=False,
        )
        criterion = nn.MSELoss()
        loss = evaluate(model, dl, criterion)
        self.assertTrue(np.isfinite(loss))


# ---------------------------------------------------------------------------
# Per-horizon RMSE tests
# ---------------------------------------------------------------------------

class TestPerHorizonRMSE(unittest.TestCase):
    def test_rmse_shapes_and_values(self):
        X, y = _dummy_data(n=32)
        stats = NormalizationStats.from_training_data(X, y)
        model = VehicleStateGRU(hidden_size=32)
        dl = create_dataloader(
            stats.normalize_input(X),
            stats.normalize_target(y),
            batch_size=16,
            shuffle=False,
        )
        metrics = compute_per_horizon_rmse(model, dl, stats)
        self.assertEqual(metrics["position_rmse"].shape, (3,))
        self.assertEqual(metrics["speed_rmse"].shape, (3,))
        self.assertEqual(metrics["per_feature_rmse"].shape, (3, 4))
        self.assertTrue(np.all(metrics["position_rmse"] > 0))
        self.assertTrue(np.all(metrics["speed_rmse"] > 0))


# ---------------------------------------------------------------------------
# Checkpoint round-trip tests
# ---------------------------------------------------------------------------

class TestCheckpoint(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = VehicleStateGRU(hidden_size=48)
            stats = _dummy_stats()
            meta = {"training_mode": "local", "rsu_id": "rsu_1"}

            # Save
            path = save_checkpoint(model, stats, Path(tmpdir) / "test.pt", meta)
            self.assertTrue(path.exists())

            # Load
            loaded_model, loaded_stats, loaded_meta = load_checkpoint(path)
            self.assertEqual(loaded_model.hidden_size, 48)
            np.testing.assert_array_equal(
                loaded_stats.input_mean, stats.input_mean
            )
            self.assertEqual(loaded_meta["training_mode"], "local")

            # Verify forward pass produces same output
            x = torch.randn(2, 5, 4)
            model.eval()
            with torch.no_grad():
                out1 = model(x)
                out2 = loaded_model(x)
            torch.testing.assert_close(out1, out2)

    def test_load_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint("nonexistent_checkpoint.pt")


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------

class TestRSUStatePredictor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        # Create and save a checkpoint
        self.model = VehicleStateGRU(hidden_size=32)
        self.stats = _dummy_stats()
        self.ckpt_path = save_checkpoint(
            self.model, self.stats, self.tmp_path / "test.pt",
            {"training_mode": "test"},
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_from_checkpoint(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        self.assertEqual(predictor.model_config["hidden_size"], 32)
        self.assertEqual(predictor.metadata["training_mode"], "test")

    def test_predict_single_sample(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        history = np.random.randn(5, 4).astype(np.float32)
        result = predictor.predict(history)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, np.float32)

    def test_predict_batch(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        history = np.random.randn(8, 5, 4).astype(np.float32)
        result = predictor.predict(history)
        self.assertEqual(result.shape, (8, 3, 4))

    def test_predict_accepts_float64(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        history = np.random.randn(5, 4).astype(np.float64)
        result = predictor.predict(history)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, np.float32)

    def test_predict_rejects_wrong_shape(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        with self.assertRaisesRegex(ValueError, "5"):
            predictor.predict(np.random.randn(3, 4).astype(np.float32))
        with self.assertRaisesRegex(ValueError, "2D.*3D"):
            predictor.predict(np.random.randn(4).astype(np.float32))

    def test_predict_recursive_5_steps(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        history = np.random.randn(5, 4).astype(np.float32)
        result = predictor.predict_recursive(history, total_steps=5)
        self.assertEqual(result.shape, (5, 4))
        self.assertEqual(result.dtype, np.float32)

    def test_predict_recursive_rejects_bad_input(self):
        predictor = RSUStatePredictor.from_checkpoint(self.ckpt_path)
        with self.assertRaises(ValueError):
            predictor.predict_recursive(np.random.randn(3, 4).astype(np.float32))

    def test_from_checkpoint_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            RSUStatePredictor.from_checkpoint("no_such_file.pt")


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------

class TestInferenceLatency(unittest.TestCase):
    def test_latency_measurement(self):
        model = VehicleStateGRU(hidden_size=32)
        result = measure_inference_latency(
            model, warmup_runs=2, timed_runs=5
        )
        self.assertIn("mean_ms", result)
        self.assertIn("std_ms", result)
        self.assertGreater(result["mean_ms"], 0.0)
        self.assertEqual(result["warmup_runs"], 2)
        self.assertEqual(result["timed_runs"], 5)


if __name__ == "__main__":
    unittest.main()
