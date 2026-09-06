"""Fleet Planner: split one of the sample routes across a fleet of vehicles
instead of a single driver -- a multi-vehicle Capacitated VRP with Time
Windows (CVRPTW), reusing the exact same hybrid cost function
(C_ij = T_ij + alpha * scale * P_ij) as the single-vehicle demo on the Home
page. All solving logic lives in fleet_solver.py; this file only assembles a
UI from what it produces.

There is no official Amazon score for a multi-vehicle split (the dataset's
ground truth is one driver's single sequence per route) -- this page reports
total travel time and capacity/time-window feasibility per vehicle instead.

Launch:
    streamlit run src/app.py
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from dashboard_common import FLEET_COLORS, PROCESSED_DIR, ROUTES, build_route_map, get_pij_table
from fleet_solver import compute_vehicle_stats, default_capacity_cm3, load_demand, solve_fleet
from model_apply import load_route

TIME_LIMIT_SECONDS = 30
ALPHA = 1.0

st.title("Fleet Planner")
st.caption(
    "Split one route across several vehicles instead of a single driver -- a capacitated, "
    "time-windowed vehicle routing problem (VRP), using the same learned zone-habit penalty "
    "as the Home page's single-vehicle solver."
)

label = st.selectbox("Route", list(ROUTES.keys()))
route_id = ROUTES[label]

col1, col2 = st.columns(2)
with col1:
    num_vehicles = st.slider("Number of vehicles", min_value=1, max_value=6, value=3)


@st.cache_data
def _route_and_demand(route_id: str):
    route = load_route(PROCESSED_DIR, route_id)
    demand = load_demand(PROCESSED_DIR, route)
    return route, demand


route, demand = _route_and_demand(route_id)
suggested_capacity = round(default_capacity_cm3(demand, num_vehicles))

with col2:
    # Keying on (route_id, num_vehicles) means the slider's default jumps to a
    # freshly computed, feasible-ish value whenever either changes, rather
    # than silently keeping a stale value sized for a different route/fleet.
    capacity = st.slider(
        "Vehicle capacity (cm³)",
        min_value=max(1, suggested_capacity // 4),
        max_value=suggested_capacity * 3,
        value=suggested_capacity,
        step=max(1, suggested_capacity // 100),
        key=f"capacity_{route_id}_{num_vehicles}",
        help="Defaults to an even split of this route's total package volume, plus 40% slack.",
    )

st.caption(f"This route's total package volume: {sum(demand):,} cm³ across {len(demand) - 1} stops.")

if "fleet_result" not in st.session_state:
    st.session_state.fleet_result = None

if st.button("Solve Fleet"):
    pij_table = get_pij_table()
    with st.spinner(f"Solving for {num_vehicles} vehicles (up to {TIME_LIMIT_SECONDS}s)..."):
        t0 = time.time()
        vehicle_orders = solve_fleet(
            route, demand, num_vehicles=num_vehicles, vehicle_capacity_cm3=capacity,
            time_limit_seconds=TIME_LIMIT_SECONDS, pij_table=pij_table, alpha=ALPHA,
        )
        solve_time = time.time() - t0

    if vehicle_orders is None:
        st.error(
            f"No feasible {num_vehicles}-vehicle split found within {TIME_LIMIT_SECONDS}s at this capacity. "
            "Try more vehicles, more capacity, or fewer vehicles with more capacity each."
        )
        st.session_state.fleet_result = None
    else:
        stops_df = pd.read_parquet(PROCESSED_DIR / "stops.parquet", filters=[("route_id", "==", route_id)])
        coords = stops_df.set_index("stop_code")[["lat", "lng"]]
        stats = compute_vehicle_stats(route, vehicle_orders, demand, pij_table)
        st.session_state.fleet_result = {
            "route_id": route_id, "coords": coords, "depot_code": route.node_codes[route.depot_idx],
            "vehicle_orders": vehicle_orders, "stats": stats, "capacity": capacity, "solve_time": solve_time,
        }

if st.session_state.fleet_result is not None and st.session_state.fleet_result["route_id"] == route_id:
    r = st.session_state.fleet_result
    st.success(f"Solved in {r['solve_time']:.1f}s.")
    st.info(
        "No official Amazon score applies here -- the dataset's ground truth is one driver's single "
        "sequence, not a multi-vehicle split. Numbers below are travel time and feasibility per vehicle."
    )

    layers = []
    table_rows = []
    for s in r["stats"]:
        color = FLEET_COLORS[s["vehicle"] % len(FLEET_COLORS)]
        if s["stops"] > 0:
            codes = [route.node_codes[i] for i in s["order"]]
            layers.append((f"Vehicle {s['vehicle'] + 1} ({s['stops']} stops)", codes, color, None))
        table_rows.append({
            "Vehicle": s["vehicle"] + 1,
            "Stops": s["stops"],
            "Travel time (min)": round(s["total_travel_seconds"] / 60, 1),
            "Capacity used (cm³)": f"{s['used_capacity_cm3']:,} / {r['capacity']:,}",
            "Mean P_ij": f"{s['mean_p_ij']:.3f}" if s["mean_p_ij"] is not None else "—",
            "Time-window violations": len(s["violations"]),
        })

    m = build_route_map(r["coords"], r["depot_code"], layers)
    from streamlit_folium import st_folium
    st_folium(m, width=None, height=520, returned_objects=[])

    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    total_violations = sum(row["Time-window violations"] for row in table_rows)
    if total_violations == 0:
        st.caption("All time windows satisfied across every vehicle.")
    else:
        st.warning(f"{total_violations} time-window violation(s) across the fleet -- see the table above.")
