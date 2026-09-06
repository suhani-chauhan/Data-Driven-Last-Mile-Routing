"""Solve a delivery route as a multi-vehicle Capacitated VRP with Time
Windows (CVRPTW) -- splits one route across a fleet of vehicles instead of
model_apply.py's single-vehicle Asymmetric TSP. See fleet_solver.py's module
docstring for the cost function, capacity model, and the important note
that there is no official Amazon score for a multi-vehicle split.

Usage:
    python src/model_apply_fleet.py [--route-id ROUTE_ID] [--processed-dir data/processed]
                                     [--num-vehicles 3] [--vehicle-capacity-cm3 250000]
                                     [--time-limit-seconds 30] [--alpha 1.0]

If --route-id is omitted, the same default-route logic as model_apply.py is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fleet_solver import DEFAULT_NUM_VEHICLES, default_capacity_cm3, fleet_summary, load_demand, solve_fleet
from model_apply import DEFAULT_PROCESSED_DIR, DEFAULT_TIME_LIMIT_SECONDS, load_route, pick_default_route
from zone_penalty import build_pij_table

DEFAULT_ALPHA = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route-id", default=None)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--num-vehicles", type=int, default=DEFAULT_NUM_VEHICLES)
    ap.add_argument(
        "--vehicle-capacity-cm3",
        type=float,
        default=None,
        help="defaults to an even split of this route's total package volume, plus 40%% slack",
    )
    ap.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args()

    route_id = args.route_id or pick_default_route(args.processed_dir)
    route = load_route(args.processed_dir, route_id)
    demand = load_demand(args.processed_dir, route)
    pij_table = build_pij_table(args.processed_dir)
    capacity = args.vehicle_capacity_cm3
    if capacity is None:
        capacity = default_capacity_cm3(demand, args.num_vehicles)
        print(f"(no --vehicle-capacity-cm3 given; defaulting to {round(capacity):,} cm3/vehicle)")

    vehicle_orders = solve_fleet(
        route,
        demand,
        num_vehicles=args.num_vehicles,
        vehicle_capacity_cm3=capacity,
        time_limit_seconds=args.time_limit_seconds,
        pij_table=pij_table,
        alpha=args.alpha,
    )
    if vehicle_orders is None:
        raise SystemExit(
            f"no feasible {args.num_vehicles}-vehicle split found for {route_id!r} within "
            f"{args.time_limit_seconds}s (try more vehicles, more capacity, or a longer time limit)"
        )
    fleet_summary(route, vehicle_orders, demand, capacity, pij_table=pij_table, alpha=args.alpha)


if __name__ == "__main__":
    main()
