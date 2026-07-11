"""Portfolio visualization from the brief: side-by-side plots of the baseline
(pure T_ij) and zone-penalty hybrid (alpha=1.0) solved routes over real stop
coordinates, each overlaid with the actual driver's path as a dotted
reference line, so a viewer can see which solved route tracks the real
driver's path more closely.

Both conditions are solved at the same --time-limit-seconds (default 60,
matching compare_baseline_vs_hybrid.py's methodology -- a shorter budget was
found to make OR-Tools' time-limited search inconsistent between otherwise-
identical runs, so a mismatched-budget comparison wouldn't be trustworthy
here either).

Colors are fixed by entity, not by panel: "solved route" is always the same
blue in both subplots, "actual driver route" is always the same dark dotted
line in both subplots -- only the panel position and title (with each
route's official score) distinguish baseline from hybrid. Colors are Okabe-Ito
(colorblind-safe): blue #0072B2 for the solved route, vermillion #D55E00 star
for the depot/start.

Usage:
    python src/plot_route_comparison.py [--route-id ROUTE_ID] [--processed-dir data/processed]
                                         [--time-limit-seconds 60] [--alpha 1.0]
                                         [--output results/route_comparison.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model_apply import load_route, solve
from model_score import actual_sequence_list, cost_matrix_dict, score, submitted_sequence_list
from zone_penalty import build_pij_table

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT = Path("results/route_comparison.png")
DEFAULT_ROUTE_ID = "RouteID_64cb7ba5-342d-46db-9e04-962248c6f667"
DEFAULT_TIME_LIMIT_SECONDS = 60
DEFAULT_ALPHA = 1.0

SOLVED_COLOR = "#0072B2"  # Okabe-Ito blue
ACTUAL_COLOR = "#404040"  # neutral dark gray, distinct from any categorical hue
DEPOT_COLOR = "#D55E00"  # Okabe-Ito vermillion


def build_comparison_figure(route_id: str, stop_count: int, coords: pd.DataFrame, actual_codes: list[str], panels: list[tuple[str, list[str], float]]):
    """panels: list of (label, solved_codes, official_score) tuples, one per subplot.
    coords: DataFrame indexed by stop_code with lat/lng columns. Pure plotting --
    takes already-solved/scored data in, returns a Figure, no solving/scoring/IO."""
    actual_lat = [coords.loc[code].lat for code in actual_codes]
    actual_lng = [coords.loc[code].lng for code in actual_codes]

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7), sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, solved_codes, s) in zip(axes, panels):
        solved_lat = [coords.loc[code].lat for code in solved_codes]
        solved_lng = [coords.loc[code].lng for code in solved_codes]

        ax.plot(actual_lng, actual_lat, linestyle=":", linewidth=2, color=ACTUAL_COLOR, label="Actual driver route", zorder=2)
        ax.plot(solved_lng, solved_lat, linestyle="-", linewidth=2, color=SOLVED_COLOR, marker="o", markersize=4, label="Solved route", zorder=3)
        ax.scatter([coords.loc[solved_codes[0]].lng], [coords.loc[solved_codes[0]].lat], s=120, marker="*", color=DEPOT_COLOR, label="Depot (start)", zorder=4)

        ax.set_title(f"{label}\nofficial score: {s:.4f}  (0.0 = identical to actual, lower is better)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Route {route_id}  ({stop_count} stops)", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route-id", default=DEFAULT_ROUTE_ID)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    route = load_route(args.processed_dir, args.route_id)
    stops_df = pd.read_parquet(args.processed_dir / "stops.parquet", filters=[("route_id", "==", args.route_id)])
    travel_df = pd.read_parquet(args.processed_dir / "travel_times.parquet", filters=[("route_id", "==", args.route_id)])
    coords = stops_df.set_index("stop_code")[["lat", "lng"]]

    actual = actual_sequence_list(stops_df)
    cost_mat = cost_matrix_dict(travel_df)

    print("building zone-transition penalty table...")
    pij_table = build_pij_table(args.processed_dir)

    panels = []
    for label, pij, alpha in [("Baseline (T_ij only)", None, 0.0), (f"Zone-Penalty Hybrid (alpha={args.alpha})", pij_table, args.alpha)]:
        print(f"solving {label}...")
        order = solve(route, args.time_limit_seconds, pij_table=pij, alpha=alpha)
        if order is None:
            raise SystemExit(f"{label}: no feasible solution within {args.time_limit_seconds}s")
        solved_codes = [route.node_codes[i] for i in order]
        sub = submitted_sequence_list(solved_codes)
        s = score(actual, sub, cost_mat)
        print(f"  {label}: official score = {s:.6f}")
        panels.append((label, solved_codes, s))

    fig = build_comparison_figure(args.route_id, len(route.node_codes), coords, actual, panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved figure to {args.output}")


if __name__ == "__main__":
    main()
