"""Multi-vehicle Capacitated VRP with Time Windows (CVRPTW) -- an extension
of model_apply.py's single-vehicle Asymmetric TSP to a fleet of vehicles
splitting one route.

Reuses model_apply.py's exact hybrid-cost math (make_distance_callback,
make_time_callback) so a fleet solve uses the identical
C_ij = T_ij + alpha * scale * P_ij objective as the single-vehicle solver --
nothing about the cost function is reimplemented here, only the vehicle
count and a capacity constraint are added.

Demand / capacity: the dataset has no package weight field, so each stop's
demand is the summed package volume_cm3 of everything delivered there (the
one physical quantity available that a real van's cargo space is limited
by); vehicle_capacity_cm3 is the same figure applied to every vehicle.

Important scope note: there is no official Amazon score for a fleet solve.
The dataset's ground truth (actual_sequences.json) is one sequence per
route, driven by one real driver -- it has no notion of splitting a route
across multiple vehicles, so model_score.py's ERP-based metric does not
apply here. fleet_summary() reports total travel time and capacity/time-
window feasibility instead, not a sequence-deviation score.

Usage as a library:
    from fleet_solver import solve_fleet, fleet_summary, load_demand
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from model_apply import RouteData, make_distance_callback, make_time_callback
from zone_penalty import ZonePenaltyTable

DEFAULT_NUM_VEHICLES = 3
CAPACITY_SLACK = 1.4  # default capacity = an even split of the route's total demand, plus 40% slack for imbalance


def default_capacity_cm3(demand: list[int], num_vehicles: int, slack: float = CAPACITY_SLACK) -> float:
    """A route-aware default vehicle capacity: routes vary hugely in total
    package volume (a 20-stop route and a 180-stop route need very different
    per-vehicle capacity), so a single fixed constant across all routes would
    be infeasible for some and meaninglessly slack for others. The slack
    factor accounts for stops not splitting perfectly evenly across
    vehicles."""
    total_demand = sum(demand)
    if total_demand == 0 or num_vehicles == 0:
        return 0.0
    return (total_demand / num_vehicles) * slack


def load_demand(processed_dir: Path, route: RouteData) -> list[int]:
    """Per-node demand for the Capacity dimension: total package volume_cm3
    at that stop (0 at the depot, which carries no delivery demand itself)."""
    packages = pd.read_parquet(
        processed_dir / "packages.parquet",
        columns=["route_id", "stop_code", "volume_cm3"],
        filters=[("route_id", "==", route.route_id)],
    )
    volume_by_stop = packages.groupby("stop_code")["volume_cm3"].sum()
    demand = [round(volume_by_stop.get(code, 0.0)) for code in route.node_codes]
    demand[route.depot_idx] = 0
    return demand


def solve_fleet(
    route: RouteData,
    demand: list[int],
    num_vehicles: int,
    vehicle_capacity_cm3: float,
    time_limit_seconds: int,
    pij_table: ZonePenaltyTable | None = None,
    alpha: float = 0.0,
) -> list[list[int]] | None:
    """Returns one stop-order list per vehicle (each starting with the depot
    node, delivery stops only after that -- mirrors model_apply.solve()'s
    "open route" convention of dropping the free closing arc back to the
    depot). Returns None if no feasible split exists within the time budget
    (e.g. too few vehicles/too little capacity for the route's total
    demand)."""
    n = len(route.node_codes)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, route.depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    transit_idx = routing.RegisterTransitCallback(make_distance_callback(route, manager, pij_table, alpha))
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    time_idx = routing.RegisterTransitCallback(make_time_callback(route, manager))
    routing.AddDimension(time_idx, route.horizon, route.horizon, True, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")
    for node in range(n):
        index = manager.NodeToIndex(node)
        start, end = route.time_windows[node]
        time_dimension.CumulVar(index).SetRange(start, end)

    def demand_callback(from_index: int) -> int:
        return demand[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [round(vehicle_capacity_cm3)] * num_vehicles, True, "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None

    vehicle_orders: list[list[int]] = []
    for v in range(num_vehicles):
        order = []
        index = routing.Start(v)
        while not routing.IsEnd(index):
            order.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        vehicle_orders.append(order)
    return vehicle_orders


def compute_vehicle_stats(
    route: RouteData,
    vehicle_orders: list[list[int]],
    demand: list[int],
    pij_table: ZonePenaltyTable | None = None,
) -> list[dict]:
    """One dict per vehicle -- shared by fleet_summary's CLI printout and the
    Fleet Planner dashboard page's stats table, so the two never drift.
    Keys: vehicle, stops, order (node indices), total_travel_seconds,
    used_capacity_cm3, mean_p_ij (None if pij_table/route is empty),
    violations (list of (stop_code, arrival, start, end))."""
    stats = []
    for v, order in enumerate(vehicle_orders):
        stops = len(order) - 1
        if stops == 0:
            stats.append({
                "vehicle": v, "stops": 0, "order": order, "total_travel_seconds": 0,
                "used_capacity_cm3": 0, "mean_p_ij": None, "violations": [],
            })
            continue

        t = 0
        violations = []
        for i in range(1, len(order)):
            prev, cur = order[i - 1], order[i]
            arrival = t + route.service_time[prev] + route.distance_matrix[prev][cur]
            start, end = route.time_windows[cur]
            if arrival > end:
                violations.append((route.node_codes[cur], arrival, start, end))
            t = max(arrival, start)
        total_travel = sum(route.distance_matrix[order[i - 1]][order[i]] for i in range(1, len(order)))
        used_capacity = sum(demand[node] for node in order if node != route.depot_idx)

        mean_p_ij = None
        if pij_table is not None:
            p_ijs = [pij_table.get(route.zones[order[i - 1]], route.zones[order[i]]) for i in range(1, len(order))]
            mean_p_ij = sum(p_ijs) / len(p_ijs)

        stats.append({
            "vehicle": v, "stops": stops, "order": order, "total_travel_seconds": total_travel,
            "used_capacity_cm3": used_capacity, "mean_p_ij": mean_p_ij, "violations": violations,
        })
    return stats


def fleet_summary(
    route: RouteData,
    vehicle_orders: list[list[int]],
    demand: list[int],
    vehicle_capacity_cm3: float,
    pij_table: ZonePenaltyTable | None = None,
    alpha: float = 0.0,
) -> None:
    n = len(route.node_codes)
    all_visited = [node for order in vehicle_orders for node in order if node != route.depot_idx]
    assert len(all_visited) == n - 1, f"expected {n - 1} delivery stops covered, got {len(all_visited)}"
    assert len(set(all_visited)) == len(all_visited), "a stop was visited by more than one vehicle"

    print(f"route: {route.route_id}")
    print(f"fleet: {len(vehicle_orders)} vehicles, capacity {round(vehicle_capacity_cm3):,} cm3 each")
    print("NOTE: no official Amazon score applies to a multi-vehicle split -- ground truth is one")
    print("      driver's single sequence. Reported below: travel time and feasibility per vehicle.\n")

    for s in compute_vehicle_stats(route, vehicle_orders, demand, pij_table):
        if s["stops"] == 0:
            print(f"vehicle {s['vehicle']}: 0 stops (unused)")
            continue
        line = (
            f"vehicle {s['vehicle']}: {s['stops']} stops, {s['total_travel_seconds']}s "
            f"({s['total_travel_seconds'] / 60:.1f} min) travel, "
            f"capacity {s['used_capacity_cm3']:,}/{round(vehicle_capacity_cm3):,} cm3"
        )
        if s["mean_p_ij"] is not None:
            line += f", mean P_ij {s['mean_p_ij']:.4f}"
        print(line)
        if s["violations"]:
            print(f"  TIME WINDOW VIOLATIONS ({len(s['violations'])}):")
            for code, arrival, start, end in s["violations"]:
                print(f"    {code}: arrival {arrival}s not in [{start}, {end}]")
        print("  " + " -> ".join(route.node_codes[i] for i in s["order"]))

    print()
    print(f"stops covered: {len(all_visited)}/{n - 1} (all distinct: {'yes' if len(set(all_visited)) == len(all_visited) else 'NO'})")
