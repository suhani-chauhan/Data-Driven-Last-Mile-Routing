"""Solve a single delivery route as an Asymmetric TSP with OR-Tools.

Cost function follows the brief: C_ij = T_ij + alpha * scale * P_ij.
- T_ij: raw travel_times.parquet travel time in seconds, unmodified.
- P_ij: zone-transition-frequency penalty from zone_penalty.py (1 - empirical
  probability of a driver transitioning from stop i's zone to stop j's zone,
  observed across every training route's actual sequence). This is the
  fallback for the brief's ML-derived P_ij term -- the XGBoost per-stop
  difficulty model in model_build.py never got a usable validation R^2 (see
  docs/project_brief.md's Known Limitations), so it isn't used here.
- scale: this route's own mean T_ij, so alpha stays a roughly route-size-
  independent multiplier instead of needing separate retuning depending on
  whether a route's stops are 60s or 600s apart (P_ij alone is in [0, 1] and
  would otherwise be numerically negligible next to raw travel times).
- alpha: tunable weight, default 1.0.

Only the arc-cost objective (SetArcCostEvaluatorOfAllVehicles) gets the P_ij
term. The Time dimension enforcing service_time + hard time-window
constraints is untouched -- those are physical constraints, not preferences,
and must stay grounded in real elapsed time regardless of alpha.

Usage:
    python src/model_apply.py [--route-id ROUTE_ID] [--processed-dir data/processed]
                               [--time-limit-seconds 30] [--alpha 1.0]

If --route-id is omitted, the smallest (by stop count) training route is picked
automatically -- training routes carry actual_sequence ground truth
(stops.stop_sequence_order), which this script uses only to report a rank
correlation for a sanity check, never as a constraint.
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.stats import kendalltau

from zone_penalty import ZonePenaltyTable, build_pij_table, zone_of

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_TIME_LIMIT_SECONDS = 30
DEFAULT_ALPHA = 1.0


@dataclass
class RouteData:
    route_id: str
    node_codes: list[str]  # index -> stop_code; index 0 is always the depot
    depot_idx: int
    distance_matrix: list[list[int]]  # seconds, T_ij straight from travel_times.parquet
    service_time: list[int]  # seconds
    time_windows: list[tuple[int, int]]  # seconds relative to departure
    horizon: int
    departure_dt: dt.datetime
    actual_sequence: dict[str, int | None]  # ground truth stop_sequence_order, for reporting only
    zones: list[str]  # index -> zone label (DEPOT/UNKNOWN sentinels included), for the P_ij lookup
    mean_travel_time: float  # this route's own mean off-diagonal T_ij, the "scale" in alpha * scale * P_ij


def pick_default_route(processed_dir: Path) -> str:
    routes = pd.read_parquet(processed_dir / "routes.parquet", columns=["route_id", "split", "stop_count"])
    train_routes = routes[routes.split == "train"]
    if train_routes.empty:
        raise ValueError("no split='train' routes found in routes.parquet")
    return train_routes.sort_values("stop_count").iloc[0].route_id


def load_route(processed_dir: Path, route_id: str) -> RouteData:
    routes = pd.read_parquet(processed_dir / "routes.parquet", filters=[("route_id", "==", route_id)])
    if len(routes) != 1:
        raise ValueError(f"expected exactly one routes.parquet row for {route_id!r}, got {len(routes)}")
    route = routes.iloc[0]

    stops = pd.read_parquet(processed_dir / "stops.parquet", filters=[("route_id", "==", route_id)])
    if len(stops) != route.stop_count:
        raise ValueError(
            f"stops.parquet has {len(stops)} rows for {route_id!r}, routes.parquet says stop_count={route.stop_count}"
        )

    travel = pd.read_parquet(processed_dir / "travel_times.parquet", filters=[("route_id", "==", route_id)])
    expected_pairs = route.stop_count * route.stop_count
    if len(travel) != expected_pairs:
        raise ValueError(
            f"travel_times.parquet has {len(travel)} rows for {route_id!r}, expected {expected_pairs} (full NxN matrix)"
        )

    depot_rows = stops[stops.stop_type == "Station"]
    if len(depot_rows) != 1:
        raise ValueError(f"expected exactly one Station stop for {route_id!r}, found {len(depot_rows)}")
    depot_code = depot_rows.iloc[0].stop_code

    stops = stops.set_index("stop_code")
    other_codes = sorted(c for c in stops.index if c != depot_code)
    node_codes = [depot_code] + other_codes
    n = len(node_codes)
    code_to_idx = {c: i for i, c in enumerate(node_codes)}

    dist = [[0] * n for _ in range(n)]
    for row in travel.itertuples(index=False):
        dist[code_to_idx[row.from_stop]][code_to_idx[row.to_stop]] = round(row.travel_time_seconds)

    departure_dt = dt.datetime.combine(route.route_date, route.departure_time_utc)

    service_time = [0] * n
    time_windows: list[tuple[int, int] | None] = [None] * n
    for code, srow in stops.iterrows():
        idx = code_to_idx[code]
        service_time[idx] = round(srow.total_planned_service_time_seconds)
        if srow.has_time_window:
            start = max(0, round((srow.window_start_utc - departure_dt).total_seconds()))
            end = round((srow.window_end_utc - departure_dt).total_seconds())
            end = max(end, start)  # window may have already closed relative to departure; keep it satisfiable
            time_windows[idx] = (start, end)
    service_time[code_to_idx[depot_code]] = 0

    horizon_candidates = [w[1] for w in time_windows if w is not None]
    fallback_horizon = sum(max(row) for row in dist) + sum(service_time)
    horizon = max(horizon_candidates + [fallback_horizon]) + 3600
    for i in range(n):
        if time_windows[i] is None:
            time_windows[i] = (0, horizon)

    actual_sequence = stops["stop_sequence_order"].to_dict()

    zones = [zone_of(stops.loc[code].stop_type, stops.loc[code].zone_id) for code in node_codes]
    off_diagonal = [dist[i][j] for i in range(n) for j in range(n) if i != j]
    mean_travel_time = sum(off_diagonal) / len(off_diagonal) if off_diagonal else 0.0

    return RouteData(
        route_id=route_id,
        node_codes=node_codes,
        depot_idx=code_to_idx[depot_code],
        distance_matrix=dist,
        service_time=service_time,
        time_windows=time_windows,  # type: ignore[arg-type]
        horizon=horizon,
        departure_dt=departure_dt,
        actual_sequence=actual_sequence,
        zones=zones,
        mean_travel_time=mean_travel_time,
    )


def solve(
    route: RouteData,
    time_limit_seconds: int,
    pij_table: ZonePenaltyTable | None = None,
    alpha: float = 0.0,
) -> list[int] | None:
    """pij_table=None (the default) reproduces the original pure-T_ij baseline --
    model_score.py's calls rely on that to stay unaffected by this cost-function
    change. model_apply.py's CLI always passes a real table and alpha=1.0 by
    default; see the module docstring for the C_ij = T_ij + alpha * scale * P_ij
    formula."""
    n = len(route.node_codes)
    manager = pywrapcp.RoutingIndexManager(n, 1, route.depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == route.depot_idx:
            return 0  # open route: the closing arc back to the depot is free and dropped from the reported order
        t_ij = route.distance_matrix[from_node][to_node]
        if pij_table is None:
            return t_ij
        p_ij = pij_table.get(route.zones[from_node], route.zones[to_node])
        return round(t_ij + alpha * route.mean_travel_time * p_ij)

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == route.depot_idx:
            return 0
        return route.distance_matrix[from_node][to_node] + route.service_time[from_node]

    time_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_idx, route.horizon, route.horizon, True, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")
    for node in range(n):
        index = manager.NodeToIndex(node)
        start, end = route.time_windows[node]
        time_dimension.CumulVar(index).SetRange(start, end)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # PATH_CHEAPEST_ARC's greedy nearest-neighbor construction routinely fails to find
    # *any* first solution once enough hard time windows are in play -- it has no
    # lookahead for windows it's about to violate. PARALLEL_CHEAPEST_INSERTION builds
    # the route by inserting each stop at its cheapest feasible position and is the
    # strategy OR-Tools' own VRPTW guidance recommends; confirmed it solves routes that
    # PATH_CHEAPEST_ARC reported infeasible on (same data, same time budget).
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None

    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return order


def validate_and_report(
    route: RouteData, order: list[int], pij_table: ZonePenaltyTable | None = None, alpha: float = 0.0
) -> None:
    n = len(route.node_codes)
    assert len(order) == n, f"solution has {len(order)} stops, expected {n}"
    assert len(set(order)) == n, "solution revisits a stop"
    assert order[0] == route.depot_idx, "solution doesn't start at the depot"

    # Independently recompute service-start times rather than trust the solver's
    # internal state, so this check would actually catch a modeling bug, not just
    # echo it. A vehicle arriving before a window opens waits (free) rather than
    # violating the window -- only arriving *after* a window closes is a real
    # violation, since no amount of waiting fixes that.
    t = 0
    violations = []
    for i in range(1, n):
        prev, cur = order[i - 1], order[i]
        arrival = t + route.service_time[prev] + route.distance_matrix[prev][cur]
        start, end = route.time_windows[cur]
        if arrival > end:
            violations.append((route.node_codes[cur], arrival, start, end))
        t = max(arrival, start)
    total_real_travel = sum(route.distance_matrix[order[i - 1]][order[i]] for i in range(1, n))

    print(f"route: {route.route_id}")
    print(f"stops visited: {n} (depot + {n - 1} delivery stops), all distinct: yes")
    print(f"total travel time along solved route: {total_real_travel}s ({total_real_travel / 60:.1f} min)")
    if pij_table is not None:
        p_ijs = [pij_table.get(route.zones[order[i - 1]], route.zones[order[i]]) for i in range(1, n)]
        mean_p_ij = sum(p_ijs) / len(p_ijs)
        print(
            f"zone-transition penalty (alpha={alpha}, scale={route.mean_travel_time:.1f}s): "
            f"mean P_ij along solved route = {mean_p_ij:.4f}"
        )
    if violations:
        print(f"TIME WINDOW VIOLATIONS ({len(violations)}):")
        for code, arrival, start, end in violations:
            print(f"  {code}: arrival {arrival}s not in [{start}, {end}]")
    else:
        print("time windows: all satisfied")

    real_order = [v for v in (route.actual_sequence.get(route.node_codes[i]) for i in order) if pd.notna(v)]
    if len(real_order) == n:
        tau, _ = kendalltau(list(range(n)), real_order)
        print(f"Kendall tau vs. actual driver sequence: {tau:.3f} (informational only, not a constraint)")

    print()
    print("solved sequence (stop_code):")
    print(" -> ".join(route.node_codes[i] for i in order))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route-id", default=None)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args()

    route_id = args.route_id or pick_default_route(args.processed_dir)
    route = load_route(args.processed_dir, route_id)
    pij_table = build_pij_table(args.processed_dir)
    order = solve(route, args.time_limit_seconds, pij_table=pij_table, alpha=args.alpha)
    if order is None:
        raise SystemExit(f"no feasible solution found for {route_id!r} within {args.time_limit_seconds}s")
    validate_and_report(route, order, pij_table=pij_table, alpha=args.alpha)


if __name__ == "__main__":
    main()
