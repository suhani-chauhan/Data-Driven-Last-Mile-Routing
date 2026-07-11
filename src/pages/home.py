"""Dashboard: (1) pick one of the routes already tested this session, solve
it both as the pure-T_ij baseline and the zone-penalty hybrid, and show an
interactive real-street map plus both official scores side by side -- the
original demo, unchanged; or (2) build your own route by adding stops (by
address or manual lat/lng), optimize it with the baseline distance solver
only, and walk through it stop by stop as a delivery-progress tracker. Mode
(1)'s comparison view also gets the same delivery-progress tracker, using
its hybrid-solved order.

Reuses model_apply.py (RouteData, load_route, solve) and model_score.py
(scoring) directly -- no solving or scoring logic is reimplemented here. The
map is built with folium (OpenStreetMap tiles, no API key) via
streamlit-folium; custom-route geocoding uses geopy's Nominatim (free, no
API key). This file only assembles a UI from what those modules produce.

Launch:
    streamlit run src/app.py
"""
from __future__ import annotations

import datetime as dt
import math
import time
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

from model_apply import RouteData, load_route, solve
from model_score import actual_sequence_list, cost_matrix_dict, isinvalid, score, submitted_sequence_list
from zone_penalty import build_pij_table

PROCESSED_DIR = Path("data/processed")
TIME_LIMIT_SECONDS = 60
ALPHA = 1.0
CUSTOM_ROUTE_TIME_LIMIT_SECONDS = 30  # custom routes are hand-entered, so small; no need for the full 60s
ASSUMED_SPEED_KMH = 30.0  # urban delivery-driving speed estimate for custom-route T_ij -- see build_custom_route_data
NOMINATIM_USER_AGENT = "data-driven-last-mile-routing-demo/1.0 (local Streamlit demo)"

ROUTES = {
    "33 stops -- 100% time-windowed stress test (RouteID_64cb7ba5)": "RouteID_64cb7ba5-342d-46db-9e04-962248c6f667",
    "59 stops (RouteID_00575ca4)": "RouteID_00575ca4-8a63-49d2-96c8-9b347be5ba6c",
    "119 stops (RouteID_00143bdd)": "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77",
    "19 stops -- eval split (RouteID_92a18d61)": "RouteID_92a18d61-1944-432e-a560-bedc863d6766",
}

# Fixed by entity, not by draw order -- Okabe-Ito colorblind-safe.
BASELINE_COLOR = "#0072B2"  # blue
HYBRID_COLOR = "#E69F00"  # orange
ACTUAL_COLOR = "#404040"  # neutral dark gray, dashed to read as "reference" not "solved"
CUSTOM_COLOR = "#009E73"  # green, distinct from all three sample-mode colors


@st.cache_resource
def get_pij_table():
    return build_pij_table(PROCESSED_DIR)


@st.cache_resource
def get_geolocator() -> Nominatim:
    # Nominatim's usage policy (operations.osmfoundation.org/policies/nominatim)
    # requires a descriptive User-Agent identifying the application -- the
    # geopy default is explicitly disallowed. One geocode per manual "Add stop"
    # click (a human clicking a button, not bulk geocoding) stays comfortably
    # under the 1 req/sec limit without needing an explicit RateLimiter.
    return Nominatim(user_agent=NOMINATIM_USER_AGENT)


def geocode_address(address: str) -> tuple[float, float] | None:
    try:
        location = get_geolocator().geocode(address, timeout=10)
    except (GeocoderTimedOut, GeocoderServiceError):
        return None
    if location is None:
        return None
    return location.latitude, location.longitude


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_custom_route_data(stops: list[dict]) -> RouteData:
    """stops: list of {"label": str, "lat": float, "lng": float}; stops[0] is
    the depot. T_ij here is a straight-line-distance estimate (haversine /
    ASSUMED_SPEED_KMH), not a real road-network travel time -- there's no free
    routing API for arbitrary addresses wired in. Time windows are fully open
    and service time is 0 for every stop, since no package/window data exists
    for custom stops (same reasoning the task itself calls out)."""
    n = len(stops)
    node_codes = [f"S{i}" for i in range(n)]
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                km = haversine_km(stops[i]["lat"], stops[i]["lng"], stops[j]["lat"], stops[j]["lng"])
                dist[i][j] = round(km / ASSUMED_SPEED_KMH * 3600)
    service_time = [0] * n
    off_diagonal = [dist[i][j] for i in range(n) for j in range(n) if i != j]
    horizon = (max(off_diagonal) * n if off_diagonal else 3600) + 3600
    time_windows = [(0, horizon)] * n
    zones = ["UNKNOWN"] * n
    mean_travel_time = sum(off_diagonal) / len(off_diagonal) if off_diagonal else 0.0

    return RouteData(
        route_id="custom-route",
        node_codes=node_codes,
        depot_idx=0,
        distance_matrix=dist,
        service_time=service_time,
        time_windows=time_windows,  # type: ignore[arg-type]
        horizon=horizon,
        departure_dt=dt.datetime.now(),
        actual_sequence={},
        zones=zones,
        mean_travel_time=mean_travel_time,
    )


def build_route_map(
    coords: pd.DataFrame,
    depot_code: str,
    layers: list[tuple[str, list[str], str, str | None]],
    code_labels: dict[str, str] | None = None,
) -> folium.Map:
    """layers: list of (label, ordered_stop_codes, color, dash_array) tuples, one
    per toggleable route condition. Numbered markers reflect each layer's own
    visit order, so the same physical stop can show a different number per layer
    -- only one layer's markers are visible at a time via LayerControl, avoiding
    three overlapping numberings on screen at once. code_labels optionally maps
    a stop_code to a human-readable label for popups (used by custom-route mode
    to show the real address instead of a synthetic "S3" code); sample mode
    leaves it None and popups show the stop_code as before."""

    def label_for(code: str) -> str:
        return code_labels[code] if code_labels else code

    depot_lat, depot_lng = coords.loc[depot_code].lat, coords.loc[depot_code].lng
    m = folium.Map(location=[depot_lat, depot_lng], zoom_start=13, tiles="OpenStreetMap")
    m.fit_bounds([[coords.lat.min(), coords.lng.min()], [coords.lat.max(), coords.lng.max()]])

    folium.Marker(
        location=[depot_lat, depot_lng],
        popup=f"Depot: {label_for(depot_code)}",
        tooltip="Depot",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for i, (label, ordered_codes, color, dash) in enumerate(layers):
        fg = folium.FeatureGroup(name=label, show=(i == 0))
        locations = [[coords.loc[c].lat, coords.loc[c].lng] for c in ordered_codes]
        folium.PolyLine(locations=locations, color=color, weight=4, opacity=0.9, dash_array=dash).add_to(fg)

        # Depot and the delivery cluster are often far apart (a single long arc
        # into a tight group of stops), so fit_bounds zooms out enough that
        # 15-30 numbered circles overlap into an unreadable smear -- MarkerCluster
        # collapses them into a single numbered bubble at that zoom, which expands
        # to the individual numbered stops as soon as you zoom in or click it.
        cluster = MarkerCluster(disable_clustering_at_zoom=16, max_cluster_radius=45).add_to(fg)

        position = 0
        for code in ordered_codes:
            if code == depot_code:
                continue
            position += 1
            folium.Marker(
                location=[coords.loc[code].lat, coords.loc[code].lng],
                icon=folium.DivIcon(html=f"""
                    <div style="background:{color};color:white;border-radius:50%;
                                width:24px;height:24px;line-height:22px;text-align:center;
                                font-size:12px;font-weight:bold;border:2px solid white;
                                box-shadow:0 0 3px rgba(0,0,0,0.6);">{position}</div>"""),
                popup=folium.Popup(f"Stop: {label_for(code)}<br>Position: {position}", max_width=200),
            ).add_to(cluster)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# CSS pulse animation for the "next stop" marker in the tracking map, injected once
# per map via folium's raw-HTML escape hatch (folium has no built-in animated icon).
_PULSE_CSS = """
<style>
@keyframes pulse-anim {
    0% { box-shadow: 0 0 0 0 rgba(213, 94, 0, 0.7); }
    70% { box-shadow: 0 0 0 14px rgba(213, 94, 0, 0); }
    100% { box-shadow: 0 0 0 0 rgba(213, 94, 0, 0); }
}
.pulse-marker { animation: pulse-anim 1.5s infinite; }
</style>
"""


def build_tracking_map(coords: pd.DataFrame, depot_code: str, remaining_codes: list[str], color: str) -> folium.Map:
    """Single-route "where do I go next" map for the Delivery Progress tracker --
    deliberately simpler than build_route_map (no layer toggle, only one route).
    remaining_codes: NOT-YET-DELIVERED stop codes in solved order (depot and
    already-delivered stops excluded). remaining_codes[0] is the next stop and
    gets a distinct, larger, pulsing marker kept outside the cluster so it's
    never hidden behind a cluster bubble -- the driver should always be able to
    find it at a glance regardless of zoom level."""
    depot_lat, depot_lng = coords.loc[depot_code].lat, coords.loc[depot_code].lng
    m = folium.Map(location=[depot_lat, depot_lng], zoom_start=13, tiles="OpenStreetMap")
    m.get_root().html.add_child(folium.Element(_PULSE_CSS))

    all_lat = [depot_lat] + [coords.loc[c].lat for c in remaining_codes]
    all_lng = [depot_lng] + [coords.loc[c].lng for c in remaining_codes]
    m.fit_bounds([[min(all_lat), min(all_lng)], [max(all_lat), max(all_lng)]])

    folium.Marker(
        location=[depot_lat, depot_lng],
        popup=f"Depot: {depot_code}",
        tooltip="Depot",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    if not remaining_codes:
        return m

    locations = [[depot_lat, depot_lng]] + [[coords.loc[c].lat, coords.loc[c].lng] for c in remaining_codes]
    folium.PolyLine(locations=locations, color=color, weight=4, opacity=0.9).add_to(m)

    next_code = remaining_codes[0]
    folium.Marker(
        location=[coords.loc[next_code].lat, coords.loc[next_code].lng],
        icon=folium.DivIcon(html="""
            <div class="pulse-marker" style="background:#D55E00;color:white;border-radius:50%;
                        width:30px;height:30px;line-height:27px;text-align:center;
                        font-size:14px;font-weight:bold;border:3px solid white;
                        box-shadow:0 0 4px rgba(0,0,0,0.7);">1</div>"""),
        popup=folium.Popup("NEXT STOP -- position 1", max_width=200),
        tooltip="Next stop",
    ).add_to(m)

    if len(remaining_codes) > 1:
        cluster = MarkerCluster(disable_clustering_at_zoom=16, max_cluster_radius=45).add_to(m)
        for position, code in enumerate(remaining_codes[1:], start=2):
            folium.Marker(
                location=[coords.loc[code].lat, coords.loc[code].lng],
                icon=folium.DivIcon(html=f"""
                    <div style="background:{color};color:white;border-radius:50%;
                                width:24px;height:24px;line-height:22px;text-align:center;
                                font-size:12px;font-weight:bold;border:2px solid white;
                                box-shadow:0 0 3px rgba(0,0,0,0.6);">{position}</div>"""),
                popup=folium.Popup(f"Position: {position}", max_width=150),
            ).add_to(cluster)

    return m


def render_delivery_progress(
    ordered_codes: list[str],
    code_to_label: dict[str, str],
    coords: pd.DataFrame,
    depot_code: str,
    color: str,
    total_seconds: float,
    session_key: str,
) -> None:
    """ordered_codes: ALL delivery stop codes in solved order (depot already
    excluded). code_to_label: stop_code -> human-readable display text (the
    stop_code itself for sample routes, the real address for custom routes).
    session_key: distinct per mode so sample-mode and custom-mode progress never
    collide or leak into each other.

    Delivered stops are tracked as a *set* of stop codes, not just a count, so
    the map can filter by membership rather than assume strict positional
    order -- the current UI only ever marks the next stop delivered (in order),
    but the underlying state doesn't hard-code that assumption."""
    delivered_key = f"{session_key}_delivered"
    if delivered_key not in st.session_state:
        st.session_state[delivered_key] = set()
    delivered: set[str] = st.session_state[delivered_key]

    remaining_codes = [c for c in ordered_codes if c not in delivered]
    n_total = len(ordered_codes)
    n_done = n_total - len(remaining_codes)

    st.subheader("Delivery Progress")
    st.progress(n_done / n_total if n_total else 0.0, text=f"{n_done} / {n_total} stops delivered")

    if remaining_codes:
        next_code = remaining_codes[0]
        st.markdown(f"#### Next stop: {code_to_label[next_code]}  _(Stop {n_done + 1} of {n_total})_")

        tracking_map = build_tracking_map(coords, depot_code, remaining_codes, color)
        # key includes n_done: st_folium keeps a fixed-key component's PREVIOUS
        # rendered state across reruns (that's the point of a key -- it's what
        # lets the map keep its pan/zoom instead of resetting on every unrelated
        # rerun), so a static key here would never show the post-delivery map.
        # Varying the key with the thing that actually changed forces a fresh
        # mount exactly when the route data changes, and only then.
        st_folium(tracking_map, width=1200, height=500, returned_objects=[], key=f"{session_key}_tracking_map_{n_done}")

        if st.button("Mark as Delivered", key=f"{session_key}_deliver"):
            st.session_state[delivered_key] = delivered | {next_code}
            st.rerun()
    else:
        st.success(
            f"🎉 Route Complete! All {n_total} stops delivered. Total route travel time: "
            f"{total_seconds:.0f}s ({total_seconds / 60:.1f} min)."
        )

    with st.expander("Full stop list", expanded=False):
        for code in ordered_codes:
            if code in delivered:
                mark = "✅ "
            elif remaining_codes and code == remaining_codes[0]:
                mark = "➡️ "
            else:
                mark = "⬜ "
            st.write(mark + code_to_label[code])

    if delivered and st.button("Reset Route", key=f"{session_key}_reset"):
        st.session_state[delivered_key] = set()
        st.rerun()


st.title("Baseline vs. Zone-Penalty Hybrid Routing Demo")

mode = st.radio(
    "Mode",
    ["Use sample route from dataset", "Build your own route"],
    horizontal=True,
)

# ---------------------------------------------------------------------------
# Mode 1: sample route from the dataset (the original demo -- unchanged logic)
# ---------------------------------------------------------------------------
if mode == "Use sample route from dataset":
    label = st.selectbox("Route", list(ROUTES.keys()))
    route_id = ROUTES[label]

    if "results" not in st.session_state:
        st.session_state.results = None

    if st.button("Solve"):
        pij_table = get_pij_table()
        route = load_route(PROCESSED_DIR, route_id)
        stops_df = pd.read_parquet(PROCESSED_DIR / "stops.parquet", filters=[("route_id", "==", route_id)])
        travel_df = pd.read_parquet(PROCESSED_DIR / "travel_times.parquet", filters=[("route_id", "==", route_id)])
        coords = stops_df.set_index("stop_code")[["lat", "lng"]]
        actual = actual_sequence_list(stops_df)
        cost_mat = cost_matrix_dict(travel_df)

        conditions = [
            {"key": "baseline", "friendly": "Straight-line route", "technical": "Baseline (T_ij only)", "pij": None, "alpha": 0.0},
            {"key": "hybrid", "friendly": "Smart route (learns from real drivers)", "technical": f"Zone-Penalty Hybrid (alpha={ALPHA})", "pij": pij_table, "alpha": ALPHA},
        ]
        panels = []
        solve_times = {}
        for cond in conditions:
            with st.spinner(f"Solving: {cond['friendly']} (up to {TIME_LIMIT_SECONDS}s)..."):
                t0 = time.time()
                order = solve(route, TIME_LIMIT_SECONDS, pij_table=cond["pij"], alpha=cond["alpha"])
                solve_times[cond["key"]] = time.time() - t0
            if order is None:
                st.error(f"{cond['friendly']}: no feasible solution within {TIME_LIMIT_SECONDS}s")
                st.stop()
            solved_codes = [route.node_codes[i] for i in order]
            sub = submitted_sequence_list(solved_codes)
            if isinvalid(actual, sub):
                st.error(f"{cond['friendly']}: solved route flagged invalid by the scorer")
                st.stop()
            s = score(actual, sub, cost_mat)
            total_travel = sum(route.distance_matrix[order[k - 1]][order[k]] for k in range(1, len(order)))
            panels.append({
                "key": cond["key"],
                "friendly": cond["friendly"],
                "technical": cond["technical"],
                "codes": solved_codes,
                "score": s,
                "travel": total_travel,
            })

        # Cached in session_state, not just a local variable: streamlit-folium's map
        # reruns the script on interaction (pan/click), and without this the results
        # would vanish on the next rerun since `st.button` only reads True once.
        st.session_state.results = {
            "route_id": route_id,
            "stop_count": len(route.node_codes),
            "depot_code": route.node_codes[0],
            "coords": coords,
            "actual": actual,
            "panels": panels,
            "solve_times": solve_times,
        }
        st.session_state["delivery_index_sample"] = 0  # fresh route -> fresh progress

    if st.session_state.results is not None:
        r = st.session_state.results
        baseline, hybrid = r["panels"][0], r["panels"][1]

        layers = [
            (baseline["friendly"], baseline["codes"], BASELINE_COLOR, None),
            (hybrid["friendly"], hybrid["codes"], HYBRID_COLOR, None),
            ("Actual driver route", r["actual"], ACTUAL_COLOR, "6,8"),
        ]
        st.markdown(
            f"⬛ **Depot** (home icon) &nbsp;|&nbsp; 🔵 **{baseline['friendly']}** &nbsp;|&nbsp; "
            f"🟠 **{hybrid['friendly']}** &nbsp;|&nbsp; ⬤ **Actual driver route** (dashed)\n\n"
            "Only one route's markers/line show at a time by default -- use the layer "
            "toggle box (top-right corner of the map) to switch, or check more than one "
            "to compare. Delivery stops start clustered into a single numbered bubble "
            "when the depot is far from them; **click a cluster or zoom in** to expand "
            "it into individual numbered stops, then click a numbered marker for its "
            "stop code and position."
        )
        m = build_route_map(r["coords"], r["depot_code"], layers)
        st_folium(m, width=1200, height=650, returned_objects=[], key="sample_map")

        pct_improvement = (baseline["score"] - hybrid["score"]) / baseline["score"] * 100 if baseline["score"] else float("nan")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(baseline["friendly"])
            st.metric("Difference from real driver (lower = better match)", f"{baseline['score']:.4f}")
            st.write(f"Solve time: {r['solve_times']['baseline']:.1f}s")
        with col2:
            st.subheader(hybrid["friendly"])
            st.metric(
                "Difference from real driver (lower = better match)",
                f"{hybrid['score']:.4f}",
                delta=f"{hybrid['score'] - baseline['score']:+.4f}",
                delta_color="inverse",
            )
            st.write(f"Solve time: {r['solve_times']['hybrid']:.1f}s")

        st.write(
            f"**The {hybrid['friendly'].lower()} is {pct_improvement:+.2f}% closer to what the real "
            f"driver did than the {baseline['friendly'].lower()}.**"
        )

        with st.expander("Technical details"):
            st.markdown(
                f"- **{baseline['friendly']}** = `{baseline['technical']}`: routes solved using only "
                "raw point-to-point travel time (T_ij).\n"
                f"- **{hybrid['friendly']}** = `{hybrid['technical']}`: cost function "
                "`C_ij = T_ij + alpha * scale * P_ij`, where P_ij is a zone-transition-frequency "
                "penalty learned from historical driver sequences and scale is each route's own "
                "mean travel time.\n"
                "- **Difference from real driver (lower = better match)** = the official Amazon "
                "Last-Mile Routing Research Challenge scoring metric (ERP-based sequence deviation); "
                "0.0 = identical to the actual driver route.\n\n"
                "This page wires together `model_apply.py` (routing/solving) and `model_score.py` "
                "(scoring) directly -- no solving/scoring logic lives here."
            )

        # Delivery-progress tracker walks the hybrid-solved order (the system's
        # recommended route), depot excluded. Sample routes have no separate
        # human-readable label -- the stop_code itself is the display text.
        render_delivery_progress(
            hybrid["codes"][1:],
            {c: c for c in hybrid["codes"]},
            r["coords"],
            r["depot_code"],
            HYBRID_COLOR,
            hybrid["travel"],
            "delivery_index_sample",
        )

# ---------------------------------------------------------------------------
# Mode 2: build your own route
# ---------------------------------------------------------------------------
else:
    st.info(
        "Custom routes only use straight-line-distance optimization -- the "
        "'Smart route' feature that learns from real drivers needs historical "
        "data specific to routes already in our dataset, which custom "
        "addresses don't have. Distances here are also straight-line "
        f"estimates at an assumed {ASSUMED_SPEED_KMH:.0f} km/h, not real road "
        "travel times -- no routing API is wired in here."
    )

    if "custom_stops" not in st.session_state:
        st.session_state.custom_stops = []  # list of {"label", "lat", "lng"}
    if "custom_result" not in st.session_state:
        st.session_state.custom_result = None

    st.subheader("Add a stop")
    input_mode = st.radio("Enter location by:", ["Address", "Manual lat/lng"], horizontal=True)

    if input_mode == "Address":
        address = st.text_input("Address")
        if st.button("Add stop", key="add_stop_address"):
            if not address.strip():
                st.warning("Enter an address first.")
            else:
                geocoded = geocode_address(address)
                if geocoded is None:
                    st.error(f"Could not geocode {address!r} -- try a more specific address, or switch to 'Manual lat/lng'.")
                else:
                    lat, lng = geocoded
                    st.session_state.custom_stops.append({"label": address, "lat": lat, "lng": lng})
                    st.session_state.custom_result = None  # stale after the stop list changes
                    st.rerun()
    else:
        c1, c2, c3 = st.columns(3)
        manual_lat = c1.number_input("Latitude", value=0.0, format="%.6f")
        manual_lng = c2.number_input("Longitude", value=0.0, format="%.6f")
        manual_label = c3.text_input("Label (optional)")
        if st.button("Add stop", key="add_stop_manual"):
            label_text = manual_label.strip() or f"({manual_lat:.4f}, {manual_lng:.4f})"
            st.session_state.custom_stops.append({"label": label_text, "lat": manual_lat, "lng": manual_lng})
            st.session_state.custom_result = None
            st.rerun()

    n_stops = len(st.session_state.custom_stops)
    if n_stops:
        st.subheader(f"Stops added ({n_stops}) -- #1 is treated as the depot")
        for i, s in enumerate(st.session_state.custom_stops):
            c1, c2 = st.columns([5, 1])
            tag = "[DEPOT] " if i == 0 else ""
            c1.write(f"{i + 1}. {tag}{s['label']}  ({s['lat']:.5f}, {s['lng']:.5f})")
            if c2.button("Remove", key=f"remove_stop_{i}"):
                st.session_state.custom_stops.pop(i)
                st.session_state.custom_result = None
                st.rerun()

    if n_stops < 3:
        st.caption(f"Add at least 3 stops to optimize a route ({n_stops}/3 so far).")
    elif st.button("Optimize Route"):
        custom_route = build_custom_route_data(st.session_state.custom_stops)
        with st.spinner(f"Solving (baseline only, up to {CUSTOM_ROUTE_TIME_LIMIT_SECONDS}s)..."):
            t0 = time.time()
            order = solve(custom_route, CUSTOM_ROUTE_TIME_LIMIT_SECONDS, pij_table=None, alpha=0.0)
            solve_time = time.time() - t0
        if order is None:
            st.error(f"No feasible solution found within {CUSTOM_ROUTE_TIME_LIMIT_SECONDS}s.")
        else:
            total_travel = sum(custom_route.distance_matrix[order[k - 1]][order[k]] for k in range(1, len(order)))
            st.session_state.custom_result = {
                "stops": st.session_state.custom_stops,
                "order": order,
                "solve_time": solve_time,
                "total_travel": total_travel,
            }
            st.session_state["delivery_index_custom"] = 0  # fresh route -> fresh progress
            st.rerun()

    if st.session_state.custom_result is not None:
        cr = st.session_state.custom_result
        stops = cr["stops"]
        order = cr["order"]
        code_labels = {f"S{i}": s["label"] for i, s in enumerate(stops)}

        coords_df = pd.DataFrame(
            {"lat": [s["lat"] for s in stops], "lng": [s["lng"] for s in stops]},
            index=[f"S{i}" for i in range(len(stops))],
        )
        node_codes_in_order = [f"S{i}" for i in order]
        layers = [("Straight-line route", node_codes_in_order, CUSTOM_COLOR, None)]
        m = build_route_map(coords_df, "S0", layers, code_labels=code_labels)
        st_folium(m, width=1200, height=650, returned_objects=[], key="custom_map")

        st.write(
            f"Total travel time: {cr['total_travel']}s ({cr['total_travel'] / 60:.1f} min)  |  "
            f"Solve time: {cr['solve_time']:.1f}s"
        )

        with st.expander("Technical details"):
            st.markdown(
                "This route is solved with the same baseline distance-optimization solver as "
                "the sample-route mode (`model_apply.py`, raw T_ij only, no `alpha * P_ij` "
                "penalty term). T_ij here is a straight-line (haversine) distance divided by an "
                f"assumed {ASSUMED_SPEED_KMH:.0f} km/h, not a real road-network travel time -- "
                "there's no routing API wired in for arbitrary custom addresses."
            )

        render_delivery_progress(
            node_codes_in_order[1:],
            code_labels,
            coords_df,
            "S0",
            CUSTOM_COLOR,
            cr["total_travel"],
            "delivery_index_custom",
        )
