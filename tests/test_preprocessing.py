"""Focused tests for Person 1's trajectory window and RSU data pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import unittest

from src.preprocessing import (
    FEATURE_COLUMNS,
    generate_prediction_windows,
    load_rsu_dataset,
    partition_windows_by_rsu,
    split_trajectories,
)


def _split_file(path, y_positions):
    rows = []
    for vehicle_id, y_position in enumerate(y_positions, start=1):
        for step in range(8):
            rows.append(
                {
                    "recording_id": "0750am-0805am",
                    "vehicle_id": vehicle_id,
                    "timestamp_ms": 1_000 + step * 100,
                    "x_m": float(step),
                    "y_m": float(y_position),
                    "speed_mps": 10.0 + step,
                    "heading_rad": 1.5,
                    "lane_id": 1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_window_shapes_and_targets_stay_within_trajectory(self):
        split_path = _split_file(self.tmp_path / "train.csv", [100.0])

        result = generate_prediction_windows(split_path, self.tmp_path / "windows", "train")
        x = np.load(result["x_path"])
        y = np.load(result["y_path"])

        self.assertEqual(x.shape, (1, 5, 4))
        self.assertEqual(y.shape, (1, 3, 4))
        self.assertEqual(x.dtype, y.dtype, np.float32)
        np.testing.assert_array_equal(x[0, :, 0], np.arange(5, dtype=np.float32))
        np.testing.assert_array_equal(y[0, :, 0], np.arange(5, 8, dtype=np.float32))
        self.assertEqual(tuple(FEATURE_COLUMNS), ("x_m", "y_m", "speed_mps", "heading_rad"))

    def test_windows_reject_irregular_sampling(self):
        split_path = _split_file(self.tmp_path / "train.csv", [100.0])
        frame = pd.read_csv(split_path)
        frame.loc[3, "timestamp_ms"] += 25
        frame.to_csv(split_path, index=False)

        with self.assertRaisesRegex(ValueError, "irregular sampling"):
            generate_prediction_windows(split_path, self.tmp_path / "windows", "train")

    def test_trajectory_split_has_no_overlap_and_is_reproducible(self):
        source = self.tmp_path / "states.csv"
        rows = []
        for vehicle_id in range(1, 21):
            for step in range(8):
                rows.append({
                    "recording_id": "0750am-0805am",
                    "vehicle_id": vehicle_id,
                    "timestamp_ms": 1_000 + step * 100,
                    "x_m": float(step), "y_m": 100.0,
                    "speed_mps": 10.0, "heading_rad": 1.5, "lane_id": 1,
                })
        pd.DataFrame(rows).to_csv(source, index=False)

        split_trajectories([source], self.tmp_path / "split_a", seed="test-seed")
        split_trajectories([source], self.tmp_path / "split_b", seed="test-seed")
        trajectory_sets = {}
        for split in ("train", "validation", "test"):
            first = pd.read_csv(self.tmp_path / "split_a" / f"{split}.csv")
            second = pd.read_csv(self.tmp_path / "split_b" / f"{split}.csv")
            first_ids = set(zip(first.recording_id, first.vehicle_id))
            second_ids = set(zip(second.recording_id, second.vehicle_id))
            self.assertEqual(first_ids, second_ids)
            trajectory_sets[split] = first_ids
        self.assertFalse(trajectory_sets["train"] & trajectory_sets["validation"])
        self.assertFalse(trajectory_sets["train"] & trajectory_sets["test"])
        self.assertFalse(trajectory_sets["validation"] & trajectory_sets["test"])

    def test_rsu_partition_is_disjoint_and_loader_is_scoped(self):
        windows = self.tmp_path / "windows"
        windows.mkdir()
        for split in ("train", "validation", "test"):
            path = _split_file(self.tmp_path / f"{split}.csv", [100.0, 300.0, 500.0])
            generate_prediction_windows(path, windows, split)

        counts = partition_windows_by_rsu(windows, self.tmp_path / "rsu")
        self.assertEqual(counts["rsu_1"]["train"], 1)
        self.assertEqual(counts["rsu_2"]["train"], 1)
        self.assertEqual(counts["rsu_3"]["train"], 1)

        indexes = []
        for rsu_id, lower, upper in (
            ("rsu_1", 0.0, 227.2),
            ("rsu_2", 227.2, 454.4),
            ("rsu_3", 454.4, 681.6),
        ):
            x, y = load_rsu_dataset(rsu_id, "train", self.tmp_path / "rsu")
            self.assertEqual(x.shape, (1, 5, 4))
            self.assertEqual(y.shape, (1, 3, 4))
            self.assertTrue(np.all(x[:, :, 1] >= lower))
            if rsu_id != "rsu_3":
                self.assertTrue(np.all(x[:, :, 1] < upper))
            else:
                self.assertTrue(np.all(x[:, :, 1] <= upper))
            indexes.extend(
                np.load(self.tmp_path / "rsu" / rsu_id / "train_window_index.npy")["vehicle_id"].tolist()
            )
        self.assertEqual(len(indexes), len(set(indexes)))


    def test_loader_rejects_invalid_rsu_and_split(self):
        with self.assertRaisesRegex(ValueError, "Unknown RSU"):
            load_rsu_dataset("rsu_4", "train", self.tmp_path)
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            load_rsu_dataset("rsu_1", "dev", self.tmp_path)

    def test_loader_returns_shaped_arrays_for_empty_split(self):
        rsu_dir = self.tmp_path / "rsu_1"
        rsu_dir.mkdir()
        np.save(rsu_dir / "validation_X.npy", np.empty((0, 5, 4), dtype=np.float32))
        np.save(rsu_dir / "validation_y.npy", np.empty((0, 3, 4), dtype=np.float32))

        x, y = load_rsu_dataset("rsu_1", "validation", self.tmp_path)
        self.assertEqual(x.shape, (0, 5, 4))
        self.assertEqual(y.shape, (0, 3, 4))


if __name__ == "__main__":
    unittest.main()
