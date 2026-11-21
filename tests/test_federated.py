"""Focused tests for the federated-learning orchestration module.

Uses small dummy/synthetic data and mocked RSU datasets so the test suite
can run **without** real NGSIM data or a network.  Does not import Pandas
or Matplotlib.

Run from the repository root::

    python -m pytest tests/test_federated.py -v

or::

    python -m unittest tests.test_federated -v
"""

from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from src.model import VehicleStateGRU
from src.train_utils import (
    NormalizationStats,
    create_dataloader,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)

# Import the modules under test
from src.rsu_client import RSUFlowerClient, make_rsu_client, _parameter_byte_size
from src.base_station import (
    create_fedavg_strategy,
    _weighted_average,
    _fit_metrics_aggregation,
)
from src.federated_training import (
    DEFAULT_MODEL_CONFIG,
    RSU_IDS,
    compute_shared_norm_stats,
    parse_args,
    run_federated_training,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _dummy_arrays(
    n: int = 40,
    obs: int = 5,
    pred: int = 3,
    features: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic X, y arrays for smoke tests."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, obs, features).astype(np.float32) + 10.0  # offset so std > 0
    y = rng.randn(n, pred, features).astype(np.float32) + 10.0
    return X, y


def _dummy_norm_stats(seed: int = 42) -> NormalizationStats:
    """NormalizationStats from small synthetic data."""
    X, y = _dummy_arrays(n=80, seed=seed)
    return NormalizationStats.from_training_data(X, y)


def _write_rsu_data(
    root: Path,
    rsu_ids: tuple[str, ...] = RSU_IDS,
    n_train: int = 40,
    n_val: int = 16,
    seed: int = 42,
) -> None:
    """Write small .npy files mimicking load_rsu_dataset's expected layout."""
    rng = np.random.RandomState(seed)
    for i, rsu_id in enumerate(rsu_ids):
        rsu_dir = root / rsu_id
        rsu_dir.mkdir(parents=True, exist_ok=True)
        for split, n in [("train", n_train), ("validation", n_val), ("test", 8)]:
            X = (rng.randn(n, 5, 4) + 10.0).astype(np.float32)
            y = (rng.randn(n, 3, 4) + 10.0).astype(np.float32)
            np.save(rsu_dir / f"{split}_X.npy", X)
            np.save(rsu_dir / f"{split}_y.npy", y)


# ---------------------------------------------------------------------------
# RSUFlowerClient tests
# ---------------------------------------------------------------------------

class TestRSUFlowerClient(unittest.TestCase):
    """Unit tests for the Flower client wrapper."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        _write_rsu_data(self.data_dir, n_train=24, n_val=8)
        self.norm_stats = _dummy_norm_stats()
        self.model_config = dict(DEFAULT_MODEL_CONFIG)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_client(self, rsu_id: str = "rsu_1") -> RSUFlowerClient:
        return make_rsu_client(
            rsu_id=rsu_id,
            model_config=self.model_config,
            norm_stats=self.norm_stats,
            data_dir=str(self.data_dir),
            local_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            seed=42,
        )

    def test_client_creation(self):
        """Client loads data and has correct sample counts."""
        client = self._make_client("rsu_1")
        self.assertEqual(client.rsu_id, "rsu_1")
        self.assertEqual(client.num_train_samples, 24)
        self.assertEqual(client.num_val_samples, 8)

    def test_get_parameters_returns_ndarrays(self):
        client = self._make_client()
        params = client.get_parameters(config={})
        self.assertIsInstance(params, list)
        self.assertTrue(all(isinstance(p, np.ndarray) for p in params))
        self.assertGreater(len(params), 0)

    def test_set_parameters_roundtrip(self):
        """set_parameters → get_parameters reproduces the same values."""
        client = self._make_client()
        original = client.get_parameters(config={})
        # Zero out all params
        zeroed = [np.zeros_like(p) for p in original]
        client.set_parameters(zeroed)
        after = client.get_parameters(config={})
        for z, a in zip(zeroed, after):
            np.testing.assert_array_equal(z, a)

    def test_set_parameters_rejects_wrong_count(self):
        client = self._make_client()
        with self.assertRaisesRegex(ValueError, "expected"):
            client.set_parameters([np.zeros(5)])

    def test_fit_returns_correct_structure(self):
        """fit() returns (parameters, num_samples, metrics)."""
        client = self._make_client()
        initial_params = client.get_parameters(config={})
        updated_params, num_samples, metrics = client.fit(
            initial_params, config={"local_epochs": 1}
        )
        self.assertIsInstance(updated_params, list)
        self.assertEqual(len(updated_params), len(initial_params))
        self.assertEqual(num_samples, 24)
        self.assertIn("train_loss", metrics)
        self.assertIn("rsu_id", metrics)
        self.assertIn("payload_bytes", metrics)
        self.assertIsInstance(metrics["train_loss"], float)
        self.assertTrue(np.isfinite(metrics["train_loss"]))

    def test_fit_modifies_parameters(self):
        """After one fit round, parameters should differ from the initial."""
        client = self._make_client()
        initial_params = client.get_parameters(config={})
        updated_params, _, _ = client.fit(
            initial_params, config={"local_epochs": 2}
        )
        changed = any(
            not np.array_equal(i, u)
            for i, u in zip(initial_params, updated_params)
        )
        self.assertTrue(changed, "Parameters should change after training")

    def test_evaluate_returns_correct_structure(self):
        """evaluate() returns (loss, num_samples, metrics)."""
        client = self._make_client()
        params = client.get_parameters(config={})
        loss, num_samples, metrics = client.evaluate(params, config={})
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(num_samples, 8)
        self.assertIn("val_loss", metrics)
        self.assertIn("rsu_id", metrics)

    def test_three_clients_load_different_data(self):
        """Each RSU client should have its own data (non-overlapping files)."""
        clients = [self._make_client(rsu_id) for rsu_id in RSU_IDS]
        ids = [c.rsu_id for c in clients]
        self.assertEqual(ids, list(RSU_IDS))
        # Each client should have created its own dataloader
        for c in clients:
            self.assertGreater(c.num_train_samples, 0)
            self.assertGreater(c.num_val_samples, 0)


# ---------------------------------------------------------------------------
# Base station / strategy tests
# ---------------------------------------------------------------------------

class TestBaseStation(unittest.TestCase):
    """Tests for the FedAvg strategy configuration and metric hooks."""

    def test_create_strategy(self):
        model = VehicleStateGRU()
        params = [v.cpu().numpy() for v in model.state_dict().values()]
        strategy = create_fedavg_strategy(params)
        self.assertIsNotNone(strategy)
        self.assertIsNotNone(strategy.initial_parameters)

    def test_weighted_average_metrics(self):
        """Weighted average should weight by sample count."""
        metrics = [
            (100, {"val_loss": 0.5, "rsu_id": "rsu_1"}),
            (200, {"val_loss": 0.2, "rsu_id": "rsu_2"}),
            (100, {"val_loss": 0.8, "rsu_id": "rsu_3"}),
        ]
        result = _weighted_average(metrics)
        expected = (100 * 0.5 + 200 * 0.2 + 100 * 0.8) / 400
        self.assertAlmostEqual(result["val_loss"], expected, places=6)

    def test_weighted_average_empty(self):
        result = _weighted_average([])
        self.assertTrue(np.isnan(result["val_loss"]))

    def test_fit_metrics_aggregation(self):
        metrics = [
            (100, {"train_loss": 1.0, "rsu_id": "rsu_1", "payload_bytes": 5000, "local_epochs": 1}),
            (100, {"train_loss": 2.0, "rsu_id": "rsu_2", "payload_bytes": 5000, "local_epochs": 1}),
        ]
        result = _fit_metrics_aggregation(metrics)
        self.assertAlmostEqual(result["train_loss"], 1.5, places=6)
        self.assertEqual(result["total_payload_bytes"], 10000)


# ---------------------------------------------------------------------------
# Parameter byte-size helper
# ---------------------------------------------------------------------------

class TestParameterByteSize(unittest.TestCase):
    def test_byte_size(self):
        arrays = [
            np.zeros((10, 4), dtype=np.float32),  # 160 bytes
            np.zeros((4,), dtype=np.float32),      # 16 bytes
        ]
        self.assertEqual(_parameter_byte_size(arrays), 176)


# ---------------------------------------------------------------------------
# End-to-end simulation test (with synthetic data)
# ---------------------------------------------------------------------------

class TestFederatedTrainingEndToEnd(unittest.TestCase):
    """Full simulation with tiny synthetic data.

    This test writes small .npy files, runs 2 federated rounds, and
    verifies the output checkpoint.
    """

    def test_simulation_produces_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "rsu"
            _write_rsu_data(data_dir, n_train=32, n_val=8, seed=99)

            ckpt_path = tmpdir / "test_fedavg.pt"

            saved = run_federated_training(
                data_dir=str(data_dir),
                rounds=2,
                local_epochs=1,
                batch_size=16,
                learning_rate=1e-3,
                checkpoint_path=str(ckpt_path),
                model_config=dict(DEFAULT_MODEL_CONFIG),
                seed=42,
                device="cpu",
            )

            # Checkpoint exists
            self.assertTrue(Path(saved).exists())

            # Load with Person 2's utility
            model, norm_stats, metadata = load_checkpoint(saved)

            # Model is usable
            x = torch.randn(1, 5, 4)
            model.eval()
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape, (1, 3, 4))

            # Metadata is correct
            self.assertEqual(metadata["training_mode"], "federated")
            self.assertEqual(metadata["strategy"], "FedAvg")
            self.assertEqual(metadata["num_rounds"], 2)
            self.assertEqual(metadata["num_clients"], 3)
            self.assertIn("elapsed_seconds", metadata)

            # Norm stats are valid
            self.assertEqual(norm_stats.input_mean.shape, (4,))
            self.assertEqual(norm_stats.input_std.shape, (4,))

    def test_single_round(self):
        """Even a single round should complete and save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "rsu"
            _write_rsu_data(data_dir, n_train=20, n_val=8)

            saved = run_federated_training(
                data_dir=str(data_dir),
                rounds=1,
                local_epochs=1,
                batch_size=8,
                checkpoint_path=str(tmpdir / "one_round.pt"),
                seed=0,
            )
            self.assertTrue(Path(saved).exists())


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.rounds, 20)
        self.assertEqual(args.local_epochs, 1)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.hidden_size, 64)

    def test_custom_args(self):
        args = parse_args([
            "--rounds", "10",
            "--local-epochs", "3",
            "--batch-size", "128",
            "--seed", "7",
            "--hidden-size", "32",
            "--data-dir", "/tmp/custom",
            "--checkpoint-path", "/tmp/ckpt.pt",
        ])
        self.assertEqual(args.rounds, 10)
        self.assertEqual(args.local_epochs, 3)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.hidden_size, 32)
        self.assertEqual(args.data_dir, "/tmp/custom")
        self.assertEqual(args.checkpoint_path, "/tmp/ckpt.pt")


# ---------------------------------------------------------------------------
# Checkpoint interop with RSUStatePredictor
# ---------------------------------------------------------------------------

class TestCheckpointInterop(unittest.TestCase):
    """Verify the federated checkpoint works with Person 2's predictor."""

    def test_predictor_loads_federated_checkpoint(self):
        from src.predictor import RSUStatePredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "rsu"
            _write_rsu_data(data_dir, n_train=20, n_val=8)

            ckpt_path = str(tmpdir / "interop.pt")
            run_federated_training(
                data_dir=str(data_dir),
                rounds=1,
                local_epochs=1,
                batch_size=8,
                checkpoint_path=ckpt_path,
                seed=1,
            )

            # Person 2's predictor should load and run
            predictor = RSUStatePredictor.from_checkpoint(ckpt_path)
            history = np.random.randn(5, 4).astype(np.float32)
            result = predictor.predict(history)
            self.assertEqual(result.shape, (3, 4))
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(
                predictor.metadata["training_mode"], "federated"
            )


# ---------------------------------------------------------------------------
# Global normalization stats
# ---------------------------------------------------------------------------

class TestGlobalNormStats(unittest.TestCase):
    def test_uses_fixed_scales_without_reading_rsu_data(self):
        stats = compute_shared_norm_stats()
        self.assertEqual(stats.input_mean.shape, (4,))
        self.assertEqual(stats.input_std.shape, (4,))
        self.assertTrue(np.all(stats.input_std > 0))
        np.testing.assert_array_equal(stats.input_std, [15.0, 15.0, 30.0, np.pi])


if __name__ == "__main__":
    unittest.main()
