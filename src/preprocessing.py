"""Convert the headerless NGSIM US-101 source files into vehicle state tables.

This module handles only source parsing, unit conversion, and heading
derivation. Splitting, resampling, window generation, and RSU loading are later
pipeline steps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FT_TO_M = 0.3048
NGSIM_COLUMNS = (
    "vehicle_id",
    "frame_id",
    "total_frames",
    "timestamp_ms",
    "local_x_ft",
    "local_y_ft",
    "global_x_ft",
    "global_y_ft",
    "vehicle_length_ft",
    "vehicle_width_ft",
    "vehicle_class",
    "speed_ft_s",
    "acceleration_ft_s2",
    "lane_id",
    "preceding_vehicle_id",
    "following_vehicle_id",
    "space_headway_ft",
    "time_headway_s",
)

STATE_COLUMNS = (
    "recording_id",
    "vehicle_id",
    "timestamp_ms",
    "x_m",
    "y_m",
    "speed_mps",
    "heading_rad",
    "lane_id",
)
FEATURE_COLUMNS = ("x_m", "y_m", "speed_mps", "heading_rad")
RECORDING_CODE_BY_ID = {
    "0750am-0805am": 0,
    "0805am-0820am": 1,
    "0820am-0835am": 2,
}
WINDOW_INDEX_DTYPE = np.dtype(
    [
        ("recording_code", "u1"),
        ("vehicle_id", "<i4"),
        ("input_start_timestamp_ms", "<i8"),
    ],
    align=False,
)


def _derive_heading(frame: pd.DataFrame) -> pd.Series:
    """Estimate heading from local position changes, in radians.

    The angle is counterclockwise from the positive local-X axis. At the first
    row of a track, use its first forward displacement. Repeated positions
    inherit the nearest observed direction. A wholly stationary track uses
    pi/2, the positive local-Y direction of the US-101 section coordinates.
    """
    grouped = frame.groupby("vehicle_id", sort=False)
    dx_previous = grouped["local_x_ft"].diff()
    dy_previous = grouped["local_y_ft"].diff()
    dx_next = grouped["local_x_ft"].shift(-1) - frame["local_x_ft"]
    dy_next = grouped["local_y_ft"].shift(-1) - frame["local_y_ft"]

    first = grouped.cumcount().eq(0)
    dx = dx_previous.where(~first, dx_next)
    dy = dy_previous.where(~first, dy_next)
    moving = dx.notna() & dy.notna() & (dx.ne(0) | dy.ne(0))

    heading = pd.Series(np.nan, index=frame.index, dtype="float64")
    heading.loc[moving] = np.arctan2(dy.loc[moving], dx.loc[moving])
    heading = heading.groupby(frame["vehicle_id"], sort=False).ffill()
    heading = heading.groupby(frame["vehicle_id"], sort=False).bfill()
    return heading.fillna(np.pi / 2.0)


def convert_ngsim_file(source_path: str | Path, output_path: str | Path) -> int:
    """Write one NGSIM TXT recording as a canonical SI-unit state CSV.

    Returns the number of rows written. Invalid, missing, non-finite, negative
    speed, or duplicate-timestamp data is rejected with a clear error rather
    than silently removed or repaired. The raw source audit found none. The
    recording ID is taken from the filename so repeated vehicle IDs in
    different time periods remain distinct.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    dtypes = {
        "vehicle_id": "int32",
        "frame_id": "int32",
        "timestamp_ms": "int64",
        "local_x_ft": "float64",
        "local_y_ft": "float64",
        "speed_ft_s": "float64",
        "lane_id": "int16",
    }
    selected = [
        "vehicle_id",
        "frame_id",
        "timestamp_ms",
        "local_x_ft",
        "local_y_ft",
        "speed_ft_s",
        "lane_id",
    ]
    frame = pd.read_csv(
        source_path,
        sep=r"\s+",
        header=None,
        names=NGSIM_COLUMNS,
        usecols=selected,
        dtype=dtypes,
    )

    if frame[selected].isna().any().any():
        raise ValueError(f"Missing required values in {source_path.name}")
    if not np.isfinite(frame[["local_x_ft", "local_y_ft", "speed_ft_s"]]).all().all():
        raise ValueError(f"Non-finite state values in {source_path.name}")
    if (frame["speed_ft_s"] < 0).any():
        raise ValueError(f"Negative speed values in {source_path.name}")

    frame.sort_values(["vehicle_id", "timestamp_ms"], kind="stable", inplace=True)
    duplicate_times = frame.duplicated(["vehicle_id", "timestamp_ms"])
    if duplicate_times.any():
        raise ValueError(f"Duplicate vehicle timestamps in {source_path.name}")

    frame["heading_rad"] = _derive_heading(frame)
    frame["x_m"] = frame["local_x_ft"] * FT_TO_M
    frame["y_m"] = frame["local_y_ft"] * FT_TO_M
    frame["speed_mps"] = frame["speed_ft_s"] * FT_TO_M
    frame["recording_id"] = source_path.stem.removeprefix("trajectories-")

    states = frame.loc[:, list(STATE_COLUMNS)]
    if not np.isfinite(states[["x_m", "y_m", "speed_mps", "heading_rad"]]).all().all():
        raise ValueError(f"Non-finite converted states in {source_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(output_path, index=False, float_format="%.6f")
    return len(states)


def convert_ngsim_directory(
    raw_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, int]:
    """Convert each NGSIM TXT file in a directory and return row counts."""
    raw_directory = Path(raw_directory)
    output_directory = Path(output_directory)
    results = {}
    for source_path in sorted(raw_directory.glob("trajectories-*.txt")):
        output_path = output_directory / f"{source_path.stem}_states.csv"
        results[source_path.name] = convert_ngsim_file(source_path, output_path)
    if not results:
        raise FileNotFoundError(f"No trajectories-*.txt files in {raw_directory}")
    return results


def inspect_sampling_interval(
    state_path: str | Path,
    expected_interval_ms: int = 100,
) -> dict[str, int | None]:
    """Count per-vehicle intervals matching the requested sampling period.

    The state CSV must be sorted by recording, vehicle, then timestamp, as the
    files produced by :func:`convert_ngsim_file` are. This function does not
    interpolate or alter any observations.
    """
    frame = pd.read_csv(
        state_path,
        usecols=["recording_id", "vehicle_id", "timestamp_ms"],
        dtype={"recording_id": "string", "vehicle_id": "int32", "timestamp_ms": "int64"},
    )
    intervals = frame.groupby(["recording_id", "vehicle_id"], sort=False)[
        "timestamp_ms"
    ].diff().dropna()
    if (intervals <= 0).any():
        raise ValueError(f"Duplicate or backward timestamps in {state_path}")
    matched = intervals.eq(expected_interval_ms)
    return {
        "observations": len(frame),
        "intervals": len(intervals),
        "matching_intervals": int(matched.sum()),
        "irregular_intervals": int((~matched).sum()),
        "min_interval_ms": int(intervals.min()) if len(intervals) else None,
        "max_interval_ms": int(intervals.max()) if len(intervals) else None,
    }


def split_trajectories(
    state_paths: list[str | Path],
    output_directory: str | Path,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: str = "ngsim-us101-v1",
    chunksize: int = 200_000,
) -> dict[str, int]:
    """Write deterministic trajectory-level train/validation/test CSV splits.

    A trajectory is identified by ``(recording_id, vehicle_id)``. All rows
    for a trajectory are assigned to one split before any prediction windows
    are generated. A stable hash orders trajectories within each recording;
    exact split sizes are then assigned using the requested proportions.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    paths = [Path(path) for path in state_paths]
    if not paths:
        raise FileNotFoundError("No processed state files were supplied")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_directory / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    manifest_path = output_directory / "trajectory_split_manifest.csv"
    existing = [path for path in [*output_paths.values(), manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing split files: {existing}")

    trajectories = pd.concat(
        [
            pd.read_csv(path, usecols=["recording_id", "vehicle_id"])
            .drop_duplicates()
            for path in paths
        ],
        ignore_index=True,
    ).drop_duplicates()
    assignments: dict[tuple[str, int], str] = {}
    for recording_id, group in trajectories.groupby("recording_id", sort=True):
        vehicle_ids = group["vehicle_id"].astype(int).tolist()
        ordered = sorted(
            vehicle_ids,
            key=lambda vehicle_id: hashlib.sha256(
                f"{seed}:{recording_id}:{vehicle_id}".encode("utf-8")
            ).digest(),
        )
        train_end = int(len(ordered) * train_fraction)
        validation_end = train_end + int(len(ordered) * validation_fraction)
        for index, vehicle_id in enumerate(ordered):
            split = (
                "train"
                if index < train_end
                else "validation"
                if index < validation_end
                else "test"
            )
            assignments[(str(recording_id), vehicle_id)] = split

    manifest = pd.DataFrame(
        [
            (recording_id, vehicle_id, split)
            for (recording_id, vehicle_id), split in sorted(assignments.items())
        ],
        columns=["recording_id", "vehicle_id", "split"],
    )
    manifest.to_csv(manifest_path, index=False)

    for path in output_paths.values():
        pd.DataFrame(columns=STATE_COLUMNS).to_csv(path, index=False)
    row_counts = {name: 0 for name in output_paths}
    for state_path in paths:
        for chunk in pd.read_csv(state_path, chunksize=chunksize):
            chunk_assignments = [
                assignments[(str(recording_id), int(vehicle_id))]
                for recording_id, vehicle_id in zip(
                    chunk["recording_id"], chunk["vehicle_id"]
                )
            ]
            chunk["_split"] = chunk_assignments
            for split, output_path in output_paths.items():
                selected = chunk.loc[chunk["_split"].eq(split), list(STATE_COLUMNS)]
                if not selected.empty:
                    selected.to_csv(output_path, mode="a", header=False, index=False)
                    row_counts[split] += len(selected)
    return row_counts


def generate_prediction_windows(
    split_path: str | Path,
    output_directory: str | Path,
    split_name: str,
    observation_steps: int = 5,
    prediction_steps: int = 3,
    expected_interval_ms: int = 100,
) -> dict[str, object]:
    """Write history/target arrays, keeping every window inside one trajectory.

    The arrays use the feature order ``(x_m, y_m, speed_mps, heading_rad)``
    and float32 dtype. They are saved as memory-mapped-compatible NPY files so
    downstream model code can open them without reading both complete arrays
    into RAM.
    """
    if not split_name or Path(split_name).name != split_name:
        raise ValueError("split_name must be a simple name such as train")
    if observation_steps <= 0 or prediction_steps <= 0:
        raise ValueError("observation_steps and prediction_steps must be positive")
    if expected_interval_ms <= 0:
        raise ValueError("expected_interval_ms must be positive")

    split_path = Path(split_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    x_path = output_directory / f"{split_name}_X.npy"
    y_path = output_directory / f"{split_name}_y.npy"
    index_path = output_directory / f"{split_name}_window_index.npy"
    if x_path.exists() or y_path.exists() or index_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing arrays for {split_name}")

    dtypes = {
        "recording_id": "string",
        "vehicle_id": "int32",
        "timestamp_ms": "int64",
        "x_m": "float32",
        "y_m": "float32",
        "speed_mps": "float32",
        "heading_rad": "float32",
    }
    frame = pd.read_csv(split_path, dtype=dtypes)
    required = ["recording_id", "vehicle_id", "timestamp_ms", *FEATURE_COLUMNS]
    if any(column not in frame.columns for column in required):
        raise ValueError(f"{split_path} is missing required columns")
    if frame[required].isna().any().any():
        raise ValueError(f"Missing trajectory or feature values in {split_path}")
    if not np.isfinite(frame[list(FEATURE_COLUMNS)]).all().all():
        raise ValueError(f"Non-finite feature values in {split_path}")

    group_sizes = frame.groupby(["recording_id", "vehicle_id"], sort=False).size()
    window_length = observation_steps + prediction_steps
    window_count = int((group_sizes - window_length + 1).clip(lower=0).sum())
    if window_count == 0:
        raise ValueError(f"No complete prediction windows in {split_path}")

    x_array = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(window_count, observation_steps, len(FEATURE_COLUMNS)),
    )
    y_array = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float32,
        shape=(window_count, prediction_steps, len(FEATURE_COLUMNS)),
    )
    offset = 0
    for _, trajectory in frame.groupby(["recording_id", "vehicle_id"], sort=False):
        if len(trajectory) < window_length:
            continue
        timestamps = trajectory["timestamp_ms"].to_numpy()
        if not np.all(np.diff(timestamps) == expected_interval_ms):
            raise ValueError("A trajectory has an irregular sampling interval")
        states = trajectory.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(
            states, window_shape=window_length, axis=0
        ).transpose(0, 2, 1)
        count = len(windows)
        end = offset + count
        x_array[offset:end] = windows[:, :observation_steps, :]
        y_array[offset:end] = windows[:, observation_steps:, :]
        offset = end

    if offset != window_count:
        raise RuntimeError(f"Wrote {offset} windows; expected {window_count}")
    x_array.flush()
    y_array.flush()
    del x_array, y_array
    write_window_index(
        split_path,
        output_directory,
        split_name,
        observation_steps=observation_steps,
        prediction_steps=prediction_steps,
        expected_interval_ms=expected_interval_ms,
    )
    return {
        "split": split_name,
        "windows": window_count,
        "observation_shape": (window_count, observation_steps, len(FEATURE_COLUMNS)),
        "target_shape": (window_count, prediction_steps, len(FEATURE_COLUMNS)),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "window_index_path": str(index_path),
        "dtype": "float32",
    }


def write_window_index(
    split_path: str | Path,
    output_directory: str | Path,
    split_name: str,
    observation_steps: int = 5,
    prediction_steps: int = 3,
    expected_interval_ms: int = 100,
) -> dict[str, object]:
    """Write compact metadata aligned with each generated X/y window row.

    The structured index stores a one-byte recording code, vehicle ID, and
    input-window start timestamp. ``recording_codes.json`` maps codes back to
    recording IDs. Index rows follow the same within-trajectory ordering as
    :func:`generate_prediction_windows`.
    """
    if not split_name or Path(split_name).name != split_name:
        raise ValueError("split_name must be a simple name such as train")
    if observation_steps <= 0 or prediction_steps <= 0:
        raise ValueError("observation_steps and prediction_steps must be positive")
    if expected_interval_ms <= 0:
        raise ValueError("expected_interval_ms must be positive")

    split_path = Path(split_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    index_path = output_directory / f"{split_name}_window_index.npy"
    codes_path = output_directory / "recording_codes.json"
    if index_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing index {index_path}")

    frame = pd.read_csv(
        split_path,
        usecols=["recording_id", "vehicle_id", "timestamp_ms"],
        dtype={"recording_id": "string", "vehicle_id": "int32", "timestamp_ms": "int64"},
    )
    group_sizes = frame.groupby(["recording_id", "vehicle_id"], sort=False).size()
    window_length = observation_steps + prediction_steps
    index_count = int((group_sizes - window_length + 1).clip(lower=0).sum())
    if index_count == 0:
        raise ValueError(f"No complete prediction windows in {split_path}")

    codes = {str(code): recording_id for recording_id, code in RECORDING_CODE_BY_ID.items()}
    present_recordings = set(frame["recording_id"].astype(str).unique())
    unknown_recordings = present_recordings.difference(RECORDING_CODE_BY_ID)
    if unknown_recordings:
        raise ValueError(f"No recording codes configured for {sorted(unknown_recordings)}")
    if codes_path.exists():
        if json.loads(codes_path.read_text(encoding="utf-8")) != codes:
            raise ValueError(f"Recording code mapping in {codes_path} does not match")
    else:
        codes_path.write_text(json.dumps(codes, indent=2) + "\n", encoding="utf-8")

    index_array = np.lib.format.open_memmap(
        index_path,
        mode="w+",
        dtype=WINDOW_INDEX_DTYPE,
        shape=(index_count,),
    )
    offset = 0
    for (recording_id, vehicle_id), trajectory in frame.groupby(
        ["recording_id", "vehicle_id"], sort=False
    ):
        count = len(trajectory) - window_length + 1
        if count <= 0:
            continue
        timestamps = trajectory["timestamp_ms"].to_numpy()
        if not np.all(np.diff(timestamps) == expected_interval_ms):
            raise ValueError("A trajectory has an irregular sampling interval")
        end = offset + count
        index_array["recording_code"][offset:end] = RECORDING_CODE_BY_ID[
            str(recording_id)
        ]
        index_array["vehicle_id"][offset:end] = int(vehicle_id)
        index_array["input_start_timestamp_ms"][offset:end] = timestamps[:count]
        offset = end
    if offset != index_count:
        raise RuntimeError(f"Wrote {offset} window index rows; expected {index_count}")
    index_array.flush()
    del index_array
    return {
        "split": split_name,
        "windows": index_count,
        "dtype": str(WINDOW_INDEX_DTYPE),
        "index_path": str(index_path),
        "recording_codes_path": str(codes_path),
    }


def partition_windows_by_rsu(
    window_directory: str | Path,
    output_directory: str | Path,
    zone_boundaries_m: tuple[float, float, float, float] = (0.0, 227.2, 454.4, 681.6),
) -> dict[str, dict[str, int]]:
    """Write disjoint RSU window datasets using three longitudinal road zones.

    A window is assigned only when all five observed positions fall within one
    zone. Windows whose history crosses a zone boundary are omitted. The three
    zones are half-open except the final zone, which includes its upper edge.
    """
    if len(zone_boundaries_m) != 4 or not all(
        np.isfinite(zone_boundaries_m[i])
        and zone_boundaries_m[i] < zone_boundaries_m[i + 1]
        for i in range(3)
    ):
        raise ValueError("zone_boundaries_m must contain four increasing finite values")

    window_directory = Path(window_directory)
    output_directory = Path(output_directory)
    splits = ("train", "validation", "test")
    rsu_ids = ("rsu_1", "rsu_2", "rsu_3")
    result = {rsu_id: {split: 0 for split in splits} for rsu_id in rsu_ids}
    code_path = window_directory / "recording_codes.json"
    if not code_path.exists():
        raise FileNotFoundError(code_path)
    codes = json.loads(code_path.read_text(encoding="utf-8"))

    for split in splits:
        x_path = window_directory / f"{split}_X.npy"
        y_path = window_directory / f"{split}_y.npy"
        index_path = window_directory / f"{split}_window_index.npy"
        if not all(path.exists() for path in (x_path, y_path, index_path)):
            raise FileNotFoundError(f"Missing X/y/index files for {split}")
        x_source = np.load(x_path, mmap_mode="r")
        y_source = np.load(y_path, mmap_mode="r")
        index_source = np.load(index_path, mmap_mode="r")
        if not (len(x_source) == len(y_source) == len(index_source)):
            raise ValueError(f"X, y, and window index lengths differ for {split}")

        last_y = x_source[:, :, 1]
        masks = [
            np.all((last_y >= zone_boundaries_m[0]) & (last_y < zone_boundaries_m[1]), axis=1),
            np.all((last_y >= zone_boundaries_m[1]) & (last_y < zone_boundaries_m[2]), axis=1),
            np.all((last_y >= zone_boundaries_m[2]) & (last_y <= zone_boundaries_m[3]), axis=1),
        ]
        for rsu_id, mask in zip(rsu_ids, masks):
            count = int(mask.sum())
            rsu_dir = output_directory / rsu_id
            rsu_dir.mkdir(parents=True, exist_ok=True)
            for suffix, source in (("X", x_source), ("y", y_source), ("window_index", index_source)):
                destination = rsu_dir / f"{split}_{suffix}.npy"
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite {destination}")
                target = np.lib.format.open_memmap(
                    destination, mode="w+", dtype=source.dtype,
                    shape=(count, *source.shape[1:]),
                )
                target[:] = source[mask]
                target.flush()
                del target
            result[rsu_id][split] = count

    manifest = {
        "method": "all five observed input positions must lie inside one longitudinal zone",
        "coordinate": "NGSIM Local_Y converted to meters (y_m)",
        "zone_boundaries_m": list(zone_boundaries_m),
        "zones": {
            "rsu_1": [zone_boundaries_m[0], zone_boundaries_m[1]],
            "rsu_2": [zone_boundaries_m[1], zone_boundaries_m[2]],
            "rsu_3": [zone_boundaries_m[2], zone_boundaries_m[3]],
        },
        "recording_codes": codes,
        "window_counts": result,
        "boundary_crossing_windows": "excluded from RSU datasets",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "partition_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def load_rsu_dataset(
    rsu_id: str,
    split: str = "train",
    data_directory: str | Path = "data/rsu",
) -> tuple[np.ndarray, np.ndarray]:
    """Load one RSU's X/y arrays for a split, using read-only memory maps.

    Valid RSU IDs are ``rsu_1``, ``rsu_2``, and ``rsu_3``. Valid splits are
    ``train``, ``validation``, and ``test``. Returns arrays shaped ``[N, 5, 4]``
    and ``[N, 3, 4]`` without loading the complete files into RAM.
    """
    valid_rsus = {"rsu_1", "rsu_2", "rsu_3"}
    valid_splits = {"train", "validation", "test"}
    if rsu_id not in valid_rsus:
        raise ValueError(f"Unknown RSU {rsu_id!r}; expected one of {sorted(valid_rsus)}")
    if split not in valid_splits:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(valid_splits)}")

    rsu_directory = Path(data_directory) / rsu_id
    x_path = rsu_directory / f"{split}_X.npy"
    y_path = rsu_directory / f"{split}_y.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"Dataset for {rsu_id}/{split} is missing; run partition_windows_by_rsu() first"
        )
    x_array = np.load(x_path, mmap_mode="r")
    y_array = np.load(y_path, mmap_mode="r")
    if x_array.ndim != 3 or x_array.shape[1:] != (5, len(FEATURE_COLUMNS)):
        raise ValueError(f"Unexpected input array shape in {x_path}: {x_array.shape}")
    if y_array.ndim != 3 or y_array.shape[1:] != (3, len(FEATURE_COLUMNS)):
        raise ValueError(f"Unexpected target array shape in {y_path}: {y_array.shape}")
    if len(x_array) != len(y_array):
        raise ValueError(f"Input and target sample counts differ for {rsu_id}/{split}")
    return x_array, y_array
