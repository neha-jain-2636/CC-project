# Dataset selection and provenance

## Selected source: NGSIM US-101 vehicle trajectories

Use the public US-101 trajectory data from the U.S. Department of Transportation's NGSIM collection. The project brief specifically permits NGSIM or highD. NGSIM is the practical first choice because it is publicly accessible, its observations are recorded every 0.1 seconds (100 ms), and its trajectory records include vehicle IDs, timestamps, positions, and speed. This meets the required sampling rate without first downsampling a higher-rate source.

### Official access

- Dataset landing page: <https://data.transportation.gov/d/i5zb-xe34>
- Official U.S. DOT metadata and collection overview: <https://catalog.data.gov/dataset/next-generation-simulation-ngsim-open-data>
- US-101 trajectory metadata / data dictionary PDF: <https://data.transportation.gov/api/views/8ect-6jqj/files/ddb2c29d-2ef4-4b67-94ea-b55169229bd9?download=true&filename=1-%20US%20101%20Metadata%20Documentation.pdf>
- Official collection license shown in the catalog: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0).

The US-101 trajectory ZIP was obtained from the official DOT collection and its three original TXT trajectory files are in `data/raw/`. The files are whitespace-delimited and have no header row. Keep the source filenames and do not commit raw data to a future shared repository until the team checks the dataset license and repository policy. The ZIP contains corresponding CSV conversions too; the original TXT files are being retained as the source inputs.

## Fields relevant to this project

The official US-101 dictionary describes these fields. The raw TXT files contain 18 whitespace-separated fields in this order (there is no header row):

`Vehicle_ID, Frame_ID, Total_Frames, Global_Time, Local_X, Local_Y, Global_X, Global_Y, v_Length, v_Width, v_Class, v_Vel, v_Acc, Lane_ID, Preceding, Following, Space_Headway, Time_Headway`

| Source field | Meaning / unit | Planned use |
| --- | --- | --- |
| `Vehicle_ID` | Vehicle identifier; IDs may repeat between sections | Group rows into tracks, using recording/section ID together with vehicle ID |
| `Frame_ID` | Frame number, increasing with collection time | Ordering and sampling checks |
| `Global_Time` | Elapsed time in milliseconds since 1970-01-01 | Timestamp; use to verify 100 ms intervals |
| `Local_X`, `Local_Y` | Local position of vehicle front center, in feet | Position, converted to meters for consistent SI units |
| `Global_X`, `Global_Y` | State Plane coordinates of vehicle front center, in feet | Alternative position fields; use one coordinate frame consistently |
| `v_Vel` | Instantaneous speed, in feet per second | Speed, converted to meters per second |
| `Lane_ID` | Lane identifier | Optional road-segment/RSU partition metadata |

The remaining columns (`v_Acc`, `v_Width`, `v_Class`, `Preceding`, `Following`, `Space_Headway`, `Time_Headway`) describe acceleration, vehicle attributes, neighboring vehicles, and headway. Preserve the raw source; preprocessing can select the fields it needs.

NGSIM does not provide a direct heading field in the standard trajectory table. Heading will need to be estimated from consecutive positions during preprocessing, after ordering by vehicle and timestamp. A heading estimate must not bridge a long tracking gap; the detailed cleaning and gap policy belongs to the preprocessing step.

## Candidate comparison

highD is a viable alternative with higher quality drone-extracted trajectories and 25 Hz frame rate, but it requires a manual non-commercial access request. Its official terms also prohibit sharing the raw dataset or modified versions. The model task requires 100 ms intervals, so highD would need controlled resampling from 40 ms samples. NGSIM is therefore the simpler source for this project, provided its known measurement noise is handled and documented.

Official highD information: <https://levelxdata.com/highd-dataset/>. highD format reference: <https://levelxdata.com/wp-content/uploads/2023/10/highD-Format.pdf>.

## Important caveats

- NGSIM position and speed units are feet and feet/second. Convert them to meters and meters/second before combining them with data expressed in SI units.
- The standard NGSIM position coordinates track the front center of a vehicle. Keep this reference point consistent throughout the pipeline.
- The specified position-and-speed targets are explicit in the project brief; heading is derived, not directly recorded in the standard NGSIM table.
- NGSIM trajectory measurements can be noisy. We will inspect the downloaded file and validate sampling and values before defining cleaning thresholds.
- Do not use a synthetic fallback unless official data access fails; label it clearly if needed.

## Local source files

- `data/raw/trajectories-0750am-0805am.txt`
- `data/raw/trajectories-0805am-0820am.txt`
- `data/raw/trajectories-0820am-0835am.txt`

The full-file inspection completed on 2026-10-25 found:

| File | Rows | Unique vehicle IDs in file | Rows with issues found |
| --- | ---: | ---: | ---: |
| `trajectories-0750am-0805am.txt` | 1,180,598 | 2,169 | 0 |
| `trajectories-0805am-0820am.txt` | 1,403,095 | 2,017 | 0 |
| `trajectories-0820am-0835am.txt` | 1,515,240 | 1,915 | 0 |

The scan found 18 fields on every row, no parse errors or non-finite numeric values, no repeated or backward timestamps for a vehicle within a file, and no consecutive per-vehicle time step greater than 100 ms. The `v_Vel` values range from 0 to 95.3 ft/s (no negative values); zero speed is a valid stopped state and should not be treated as missing. Vehicle IDs can repeat across recording periods, so track identity must include the source file/recording period. The converter fails with a clear `ValueError` if required values are missing/non-finite, speed is negative, or duplicate timestamps are found; it does not silently drop records. This fail-fast cleaning policy preserves auditability, and the inspected source has zero such cases.

These checks establish that the raw records parse consistently. They do not rule out measurement noise in position or prove that each observed value is physically accurate.

## Step 3 state extraction

`src/preprocessing.py` now converts each raw recording into a state CSV under `data/processed/`. The selected fields are local X/Y position, `v_Vel`, timestamp, vehicle ID, and lane ID. X, Y, and speed are converted from feet / feet per second to meters / meters per second. Timestamp remains milliseconds. The output keeps recording ID with vehicle ID because vehicle IDs can repeat between recordings.

Heading is estimated in radians from consecutive `Local_X`/`Local_Y` displacement, measured counterclockwise from positive local X. A first row uses the next displacement; repeated positions carry the nearest observed direction. If a vehicle never changes position in its recording, heading defaults to positive local Y (pi/2 radians), following the section's longitudinal coordinate direction. This is an estimate from noisy position data, not a sensor-provided heading. No observations were removed; zero speed is preserved.

Generated state files and row counts:

- `data/processed/trajectories-0750am-0805am_states.csv` — 1,180,598 rows
- `data/processed/trajectories-0805am-0820am_states.csv` — 1,403,095 rows
- `data/processed/trajectories-0820am-0835am_states.csv` — 1,515,240 rows

Each output has eight columns: `recording_id`, `vehicle_id`, `timestamp_ms`, `x_m`, `y_m`, `speed_mps`, `heading_rad`, and `lane_id`. Conversion checks verified finite numeric state values and unchanged row counts.

## Step 4 sampling validation

`inspect_sampling_interval()` in `src/preprocessing.py` checks the timestamp difference between consecutive observations per recording and vehicle. All **4,092,832** per-vehicle intervals in the three processed files were exactly 100 ms (no irregular intervals, duplicates, or backward timestamps). Therefore the data already matches the requested 10 Hz rate; Step 4 does not resample or invent observations. All observations and values are retained; rows are ordered by recording, vehicle, then timestamp.

## Step 5 trajectory-level splits

`split_trajectories()` assigns each complete `(recording_id, vehicle_id)` trajectory to one split before window generation. A SHA-256 ordering with seed `ngsim-us101-v1` makes assignment reproducible. Split sizes are 70% train, 15% validation, and 15% test per recording (integer rounding assigns remaining trajectories to test).

| Split | Trajectories | Rows |
| --- | ---: | ---: |
| Train | 4,269 | 2,877,382 |
| Validation | 914 | 608,114 |
| Test | 918 | 613,437 |

The split row counts sum to all 4,098,933 state rows. The three trajectory ID sets have zero pairwise overlap. Per-recording trajectory counts are train/validation/test: 1,518/325/326 (07:50), 1,411/302/304 (08:05), and 1,340/287/288 (08:20). Sampling validation on the split files also found every per-trajectory interval remained exactly 100 ms.

Outputs are `data/splits/train.csv`, `data/splits/validation.csv`, `data/splits/test.csv`, and `data/splits/trajectory_split_manifest.csv`. Generate prediction windows from each split separately so no overlapping window can cross split boundaries.

## Step 6 prediction windows

`generate_prediction_windows()` creates a window of eight consecutive 100 ms observations within each individual trajectory, using the first five states as input and the following three as targets. Features are ordered as `(x_m, y_m, speed_mps, heading_rad)`. Each split is processed independently, and windows never cross vehicle or recording boundaries. Arrays are float32 NPY files that can be opened with `numpy.load(..., mmap_mode="r")`.

| Split | Windows | X shape | y shape |
| --- | ---: | --- | --- |
| Train | 2,847,499 | `[N, 5, 4]` | `[N, 3, 4]` |
| Validation | 601,716 | `[N, 5, 4]` | `[N, 3, 4]` |
| Test | 607,011 | `[N, 5, 4]` | `[N, 3, 4]` |

The output arrays have finite values, their shapes and dtypes were checked, and the first generated window in each split was compared with its source rows. Files are `data/windows/train_X.npy`, `train_y.npy`, `validation_X.npy`, `validation_y.npy`, `test_X.npy`, and `test_y.npy`.

Each array row also has a corresponding compact metadata row in `data/windows/{split}_window_index.npy`. Its fields are `recording_code`, `vehicle_id`, and `input_start_timestamp_ms`; the shared `data/windows/recording_codes.json` maps each recording code to its recording ID. The index uses 13 bytes per window and the same trajectory/window ordering as X and y, so a window can be traced back to its source without adding identifiers to the model feature tensors. Index lengths have been checked against both X and y for all three splits.

## Step 7 simulated RSU zones

`partition_windows_by_rsu()` assigns each prediction window to one of three contiguous longitudinal zones using `y_m` (NGSIM `Local_Y`, converted to meters). The current fixed boundaries are 0.0, 227.2, 454.4, and 681.6 m, covering the observed position range. A window is retained for an RSU only when all five input positions lie inside its zone; histories that cross a zone boundary are excluded. This makes the input datasets disjoint and keeps each RSU's observed history within its simulated coverage area. The rule is a reproducible simulation assumption, not a claim about real RSU locations or radio coverage.

Outputs are written separately under `data/rsu/rsu_1/`, `data/rsu/rsu_2/`, and `data/rsu/rsu_3/`, with X, y, and window-index NPY files per split. `data/rsu/partition_manifest.json` records the boundaries, assignment method, recording-code mapping, and counts. Current assigned window counts (train / validation / test): RSU 1: 1,034,586 / 219,014 / 222,304; RSU 2: 1,007,220 / 212,261 / 213,852; RSU 3: 772,135 / 163,288 / 163,651. Some windows are intentionally omitted at zone boundaries and outside the stated coverage extent.

`load_rsu_dataset(rsu_id, split="train")` reads only the requested RSU's files and returns its X/y arrays as read-only memory maps. It accepts RSU IDs `rsu_1`, `rsu_2`, `rsu_3` and splits `train`, `validation`, `test`; invalid values raise `ValueError`. An empty split is valid when both arrays exist with shapes `[0,5,4]` and `[0,3,4]`; the loader returns those empty arrays, while a missing file raises `FileNotFoundError`. The loader was checked against all nine RSU/split combinations and rejects invalid RSU/split names.

## Step 9 validation

`tests/test_preprocessing.py` uses Python's built-in `unittest` runner and small temporary synthetic trajectories; it does not modify the real NGSIM outputs. Tests cover window shape and target ordering, rejection of irregular timestamps, reproducible trajectory-level split separation, disjoint RSU assignment and zone bounds, invalid loader arguments, and correctly shaped empty splits. Run them with `python -m unittest discover -s tests -v`. Separately, the full real-data audit confirmed no trajectory overlap between train/validation/test and aligned the per-window index back to source rows for sampled windows in each split.

## Integration status

The converter fails clearly on invalid source rows instead of silently dropping them; the downloaded NGSIM files were inspected and contained no invalid rows. Repository integration and agreement on shared interfaces remain pending until the team repository is available.

## Source references

1. U.S. Department of Transportation, *Next Generation Simulation (NGSIM) Open Data*, <https://catalog.data.gov/dataset/next-generation-simulation-ngsim-open-data>.
2. U.S. Department of Transportation, *Next Generation Simulation (NGSIM) US Route 101 Dataset Metadata Documentation*, linked above.
3. leveLXData / RWTH Aachen, *highD Dataset*, <https://levelxdata.com/highd-dataset/>.
