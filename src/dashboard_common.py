"""Shared constants and map-building helpers used by more than one dashboard
page (pages/home.py, pages/fleet_planner.py). Kept as plain importable
functions/constants -- never top-level Streamlit UI code -- so importing
this module has no side effects on whichever page imports it.
"""
from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster

from zone_penalty import build_pij_table

# Full data/processed/ is gitignored (multi-hundred-MB, one file over GitHub's
# 100MB limit) so it never exists on Streamlit Community Cloud. data/deploy/
# is a small, git-committed subset covering just the routes in ROUTES below --
# see build_deploy_dataset.py. Local dev machines have the full
# data/processed/ and keep using it; only a from-git-clone deploy falls back
# to data/deploy/.
PROCESSED_DIR = Path("data/processed") if Path("data/processed").exists() else Path("data/deploy")

ROUTES = {
    "33 stops -- 100% time-windowed stress test (RouteID_64cb7ba5)": "RouteID_64cb7ba5-342d-46db-9e04-962248c6f667",
    "59 stops (RouteID_00575ca4)": "RouteID_00575ca4-8a63-49d2-96c8-9b347be5ba6c",
    "119 stops (RouteID_00143bdd)": "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77",
    "19 stops -- eval split (RouteID_92a18d61)": "RouteID_92a18d61-1944-432e-a560-bedc863d6766",
}

# Okabe-Ito colorblind-safe palette. Fixed by entity, not by draw order.
BASELINE_COLOR = "#0072B2"  # blue
HYBRID_COLOR = "#E69F00"  # orange
ACTUAL_COLOR = "#404040"  # neutral dark gray, dashed to read as "reference" not "solved"
CUSTOM_COLOR = "#009E73"  # green, distinct from all three sample-mode colors

# One color per fleet vehicle on the Fleet Planner page -- cycles if a route
# is split across more vehicles than colors (the dashboard caps num_vehicles
# well below this length in practice).
FLEET_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


@st.cache_resource
def get_pij_table():
    return build_pij_table(PROCESSED_DIR)


def build_route_map(
    coords: pd.DataFrame,
    depot_code: str,
    layers: list[tuple[str, list[str], str, str | None]],
    code_labels: dict[str, str] | None = None,
) -> folium.Map:
    """layers: list of (label, ordered_stop_codes, color, dash_array) tuples, one
    per toggleable route condition (or, on the Fleet Planner page, one per
    vehicle). Numbered markers reflect each layer's own visit order, so the
    same physical stop can show a different number per layer -- only one
    layer's markers are visible at a time via LayerControl, avoiding
    overlapping numberings on screen at once. code_labels optionally maps a
    stop_code to a human-readable label for popups (used by custom-route mode
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
