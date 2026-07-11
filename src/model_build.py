"""Train an XGBoost model to predict a per-stop "difficulty score".

This is milestone 3 of the project's build order (see docs/project_brief.md's
Project Status Update): the ML half of the two-level hierarchical design,
producing the raw material for a future `alpha * P_ij` penalty term once folded
into the OR-Tools solver. That folding-in is NOT done here -- this script only
trains and serializes the difficulty model.

Difficulty label (target): there is no ground-truth "difficulty" anywhere in
the raw data, so it's operationalized directly from the project's own stated
premise (human drivers deviate from pure-geometric routing because of local
knowledge a travel-time matrix can't see -- see "Project Overview" in the
brief). For each stop:
  1. Build a naive greedy nearest-neighbor route from the depot using only
     travel_times.parquet (T_ij) -- this is the "pure geometric" baseline the
     brief explicitly contrasts against.
  2. Compare that stop's position in the ACTUAL driver sequence
     (stops.stop_sequence_order) to its position in the greedy route.
  3. difficulty = |actual_rank - greedy_rank| / (route_stop_count - 1), so it's
     in [0, 1] regardless of route size.
A stop the driver visited far from where pure geometry would place it is a
stop where something -- access constraints, a time window, a local shortcut --
overrode the naive shortest-path answer. That's the signal this model learns
to anticipate for stops it hasn't seen.

Features are restricted to fields available identically at train AND apply
time. scan_status and route_score are deliberately excluded even though
they're informative on the training split -- they're outcome fields absent
from the evaluation split (data_pipeline.py's split handling), so a model
trained on them would silently degrade at inference time on eval/apply data.

Trained on a random sample of training-split routes, not all 6,112 -- read
in per-chunk batches with each chunk's raw stops/travel_times discarded
immediately after feature extraction, the same memory-bounded streaming
pattern used throughout this project (see data_pipeline.py), rather than one
bulk read of the full training set's travel_times rows.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_N_ROUTES = 600
DEFAULT_CHUNK_SIZE = 50
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 0

FEATURE_COLS = [
    "package_count",
    "total_volume_cm3",
    "total_planned_service_time_seconds",
    "has_time_window",
    "distance_from_prev_greedy_seconds",
    "local_stop_density_seconds",
    "distance_from_depot_seconds",
    "route_stop_count",
    "departure_hour",
]
TARGET_COL = "difficulty"
LOCAL_DENSITY_K = 5  # mean travel time to the K nearest other stops in the route


def greedy_nn_order(dist_matrix: list[list[float]], depot_idx: int) -> list[int]:
    """Naive "pure geometric" route: always go to the nearest unvisited stop."""
    n = len(dist_matrix)
    visited = [False] * n
    order = [depot_idx]
    visited[depot_idx] = True
    current = depot_idx
    for _ in range(n - 1):
        best, best_d = None, None
        for j in range(n):
            if not visited[j]:
                d = dist_matrix[current][j]
                if best_d is None or d < best_d:
                    best, best_d = j, d
        order.append(best)
        visited[best] = True
        current = best
    return order


def route_stop_rows(route_id: str, stops_group: pd.DataFrame, travel_group: pd.DataFrame) -> list[dict]:
    stops_group = stops_group.set_index("stop_code")
    depot_rows = stops_group[stops_group.stop_type == "Station"]
    if len(depot_rows) != 1:
        return []  # malformed route (shouldn't happen on this dataset, but don't assume)
    depot_code = depot_rows.index[0]

    codes = [depot_code] + sorted(c for c in stops_group.index if c != depot_code)
    n = len(codes)
    idx = {c: i for i, c in enumerate(codes)}

    dist = [[0.0] * n for _ in range(n)]
    for row in travel_group.itertuples(index=False):
        if row.from_stop in idx and row.to_stop in idx:
            dist[idx[row.from_stop]][idx[row.to_stop]] = row.travel_time_seconds

    greedy_order = greedy_nn_order(dist, 0)
    greedy_rank = [0] * n
    for rank, node in enumerate(greedy_order):
        greedy_rank[node] = rank
    depot_dist = dist[0]

    # Predecessor-in-the-greedy-route distance, NOT predecessor-in-the-actual-route:
    # the greedy order depends only on the distance matrix, so it's computable at
    # model_apply.py time for a route that hasn't been solved yet. The actual-route
    # predecessor would only exist once the real sequence is already known, which
    # defeats the purpose of predicting difficulty before routing.
    prev_in_greedy = [0] * n
    for rank in range(1, len(greedy_order)):
        prev_in_greedy[greedy_order[rank]] = greedy_order[rank - 1]

    local_density = [0.0] * n
    for i in range(n):
        others = sorted(dist[i][j] for j in range(n) if j != i)
        k = min(LOCAL_DENSITY_K, len(others))
        local_density[i] = sum(others[:k]) / k if k else 0.0

    rows = []
    for code in codes[1:]:  # skip depot -- not a delivery difficulty target
        srow = stops_group.loc[code]
        actual_rank = srow.stop_sequence_order
        if pd.isna(actual_rank):
            continue
        actual_rank = int(actual_rank)
        i = idx[code]
        difficulty = abs(actual_rank - greedy_rank[i]) / (n - 1)
        rows.append(
            {
                "route_id": route_id,
                "stop_code": code,
                "package_count": srow.package_count,
                "total_volume_cm3": srow.total_volume_cm3,
                "total_planned_service_time_seconds": srow.total_planned_service_time_seconds,
                "has_time_window": int(srow.has_time_window),
                "distance_from_prev_greedy_seconds": dist[prev_in_greedy[i]][i],
                "local_stop_density_seconds": local_density[i],
                "distance_from_depot_seconds": depot_dist[i],
                TARGET_COL: difficulty,
            }
        )
    return rows


def build_stop_table(processed_dir: Path, route_ids: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> pd.DataFrame:
    all_rows: list[dict] = []
    for start in range(0, len(route_ids), chunk_size):
        chunk_ids = route_ids[start : start + chunk_size]
        stops_chunk = pd.read_parquet(processed_dir / "stops.parquet", filters=[("route_id", "in", chunk_ids)])
        travel_chunk = pd.read_parquet(processed_dir / "travel_times.parquet", filters=[("route_id", "in", chunk_ids)])
        travel_groups = dict(list(travel_chunk.groupby("route_id")))
        for route_id, stops_group in stops_chunk.groupby("route_id"):
            travel_group = travel_groups.get(route_id)
            if travel_group is None:
                continue
            all_rows.extend(route_stop_rows(route_id, stops_group, travel_group))
        print(f"  processed {min(start + chunk_size, len(route_ids))}/{len(route_ids)} routes, {len(all_rows)} stop rows so far", flush=True)
    return pd.DataFrame(all_rows)


def add_route_level_features(stop_table: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    route_features = routes_df.set_index("route_id")[["stop_count", "departure_time_utc"]].rename(
        columns={"stop_count": "route_stop_count"}
    )
    route_features["departure_hour"] = route_features["departure_time_utc"].apply(lambda t: t.hour)
    return stop_table.join(route_features[["route_stop_count", "departure_hour"]], on="route_id")


def train_and_evaluate(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[xgb.XGBRegressor, dict]:
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=DEFAULT_SEED,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    metrics = {
        "rmse": float(mean_squared_error(y_val, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_val, pred)),
        "r2": float(r2_score(y_val, pred)),
        "n_train_stops": len(train_df),
        "n_val_stops": len(val_df),
        "n_train_routes": train_df.route_id.nunique(),
        "n_val_routes": val_df.route_id.nunique(),
        "target_mean": float(y_train.mean()),
        "target_std": float(y_train.std()),
    }
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--n-routes", type=int, default=DEFAULT_N_ROUTES)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    t0 = time.time()
    routes_df = pd.read_parquet(args.processed_dir / "routes.parquet")
    train_routes = routes_df[routes_df.split == "train"]
    sample = train_routes.sample(n=min(args.n_routes, len(train_routes)), random_state=args.seed)
    train_ids, val_ids = train_test_split(
        sample.route_id.tolist(), test_size=args.val_fraction, random_state=args.seed
    )
    print(f"sampled {len(sample)} training routes: {len(train_ids)} train / {len(val_ids)} val (split by route, not by stop)")

    print("building training stop table...")
    train_stops = build_stop_table(args.processed_dir, train_ids)
    print("building validation stop table...")
    val_stops = build_stop_table(args.processed_dir, val_ids)

    train_df = add_route_level_features(train_stops, routes_df)
    val_df = add_route_level_features(val_stops, routes_df)

    model, metrics = train_and_evaluate(train_df, val_df)
    print(f"validation RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")
    print(f"target distribution (train): mean={metrics['target_mean']:.4f}  std={metrics['target_std']:.4f}")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "difficulty_xgb.json"
    meta_path = args.models_dir / "difficulty_meta.json"
    model.save_model(model_path)
    with open(meta_path, "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "target_col": TARGET_COL, "metrics": metrics}, f, indent=2)
    print(f"saved model to {model_path}")
    print(f"saved metadata to {meta_path}")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
