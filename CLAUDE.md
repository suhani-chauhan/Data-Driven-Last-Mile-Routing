# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is a fresh scaffold: there are no commits yet (`git log` is empty) and no source code has
been written. Only two things currently exist on disk:

- `data/` — the raw dataset for the project (see below). Untracked.
- `venv/` — a pre-built Python 3.10.7 virtual environment. Untracked.

There is no `.gitignore` yet. Before making the first commit, add one that excludes `venv/` and `data/`
(the dataset alone is several GB and should not go into git history).

Because no application code exists, there are no build/lint/test commands to run yet. When code is added,
update this file with the actual commands (e.g. `pytest`, linter invocation) rather than assuming defaults.

## Dataset

`data/raw/` contains the Amazon Last Mile Routing Research Challenge (almrrc2021) dataset, licensed under
CC BY-NC 4.0 (see `data/raw/License.txt`). Layout:

```
data/raw/almrrc2021-data-training/
  model_build_inputs/   # route_data.json, package_data.json, travel_times.json,
                         # actual_sequences.json, invalid_sequence_scores.json (~2.2 GB total)
  model_build_outputs/
  model_apply_inputs/
  model_apply_outputs/
  model_score_inputs/
  model_score_outputs/
  model_score_timings/
data/raw/almrrc2021-data-evaluation/
  model_apply_inputs/
  model_score_inputs/
```

The naming (`model_build_*`, `model_apply_*`, `model_score_*`) follows Amazon's original challenge
pipeline convention: build a model/heuristic from training routes, apply it to generate predicted stop
sequences, then score predicted sequences against actual driver sequences. `route_data.json`,
`package_data.json`, and `travel_times.json` are the primary per-route inputs; `actual_sequences.json`
holds the ground-truth stop order used for scoring.

### Schema of `model_build_inputs/*.json`

All four files are top-level JSON objects keyed by `RouteID_<uuid>`. Within a route, the join key across
files is the 2-letter **stop code** (e.g. `"AD"`, `"AF"`) defined in that route's `route_data.json` →
`stops`. A route can be fully reconstructed by cross-referencing these files on `RouteID` + stop code.

- **`route_data.json`** — one record per route:
  ```json
  {
    "station_code": "DLA3",
    "date_YYYY_MM_DD": "2018-07-27",
    "departure_time_utc": "16:02:10",
    "executor_capacity_cm3": 3313071.0,
    "route_score": "High",
    "stops": {
      "AD": { "lat": 34.099611, "lng": -118.283062, "type": "Dropoff", "zone_id": "P-12.3C" }
    }
  }
  ```

- **`package_data.json`** — nested `route → stop_code → PackageID_<uuid> → package fields`. A stop can
  have multiple packages:
  ```json
  {
    "AD": {
      "PackageID_9d7fdd03-...": {
        "scan_status": "DELIVERED",
        "time_window": { "start_time_utc": NaN, "end_time_utc": NaN },
        "planned_service_time_seconds": 59.3,
        "dimensions": { "depth_cm": 25.4, "height_cm": 7.6, "width_cm": 17.8 }
      }
    }
  }
  ```
  `time_window` fields are literal JSON `NaN` (non-standard JSON, but Python's `json` module parses it)
  when no delivery window applies.

- **`travel_times.json`** — nested `route → stop_code → stop_code → seconds`, i.e. a full pairwise
  travel-time matrix per route (diagonal is `0.0`). This N×N-per-route matrix is why the file is ~1.8 GB:
  ```json
  {
    "AD": { "AD": 0.0, "AF": 198.3, "AG": 264.9 },
    "AF": { "AD": 209.8, "AF": 0.0, "AG": 348.3 }
  }
  ```

- **`actual_sequences.json`** — one record per route, single `"actual"` key mapping `stop_code → visit
  order (int)`, i.e. the ground-truth driver sequence:
  ```json
  { "actual": { "AD": 105, "AF": 47, "AG": 4, "BA": 33 } }
  ```

## Python environment

`venv/` is a Python 3.10.7 virtualenv with packages already installed, indicating the intended stack for
this project:

- **Data handling**: pandas, numpy, scipy, pyarrow
- **ML**: scikit-learn, xgboost
- **Routing / optimization**: ortools, networkx
- **Visualization**: matplotlib, plotly
- **Notebooks**: jupyter / jupyterlab / ipykernel
- **AWS**: boto3, awscli (likely for pulling/pushing data to S3)

Activate it on Windows with:

```
venv\Scripts\activate
```

There is no `requirements.txt` or `pyproject.toml` yet — once dependencies stabilize, freeze them into one
of these so the environment is reproducible from source rather than only existing as a local venv.
