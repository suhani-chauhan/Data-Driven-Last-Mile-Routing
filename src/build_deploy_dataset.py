"""Builds data/deploy/ -- a small subset of data/processed/ committed to git
so the Streamlit app has something to read on Streamlit Community Cloud,
where data/processed/ (gitignored, several hundred MB, and travel_times.parquet
alone exceeds GitHub's 100MB per-file limit) is never present.

The app only ever solves the routes hardcoded in pages/home.py's ROUTES dict
plus arbitrary custom routes (which need no historical data at all), so:
  - routes.parquet, stops.parquet: copied in full. stops.parquet has to stay
    full (not just the demo routes) because zone_penalty.build_pij_table()
    computes its transition-probability table from every TRAINING route's
    stop_sequence_order, not just the demo ones. Both files are already well
    under GitHub's size limits (routes.parquet ~0.4MB, stops.parquet ~42MB).
  - travel_times.parquet: filtered down to only the demo route_ids. This is
    the one file that MUST be filtered -- unfiltered it's a ~424MB single
    file (a full NxN matrix per route, for all 9,164 routes), over GitHub's
    100MB hard limit. Filtered to 4 routes it's a few hundred KB.
  - packages.parquet: not copied at all -- nothing downstream of the app
    (load_route, build_pij_table) reads it.

Run manually after regenerating data/processed/ (not part of the regular
pipeline / CI):
    python src/build_deploy_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Kept in sync with pages/home.py's ROUTES dict values -- not imported from
# there directly since home.py runs Streamlit UI calls at module scope and
# isn't meant to be imported outside a `streamlit run` context.
DEMO_ROUTE_IDS = [
    "RouteID_64cb7ba5-342d-46db-9e04-962248c6f667",  # 33 stops -- 100% time-windowed stress test
    "RouteID_00575ca4-8a63-49d2-96c8-9b347be5ba6c",  # 59 stops
    "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77",  # 119 stops
    "RouteID_92a18d61-1944-432e-a560-bedc863d6766",  # 19 stops -- eval split
]

SRC_DIR = Path("data/processed")
DST_DIR = Path("data/deploy")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    demo_route_ids = sorted(DEMO_ROUTE_IDS)
    print(f"Demo route IDs ({len(demo_route_ids)}): {demo_route_ids}")

    for name in ["routes.parquet", "stops.parquet"]:
        src = SRC_DIR / name
        dst = DST_DIR / name
        pq.write_table(pq.read_table(src), dst)
        print(f"{name}: copied in full, {dst.stat().st_size / 1e6:.2f} MB")

    travel = pd.read_parquet(
        SRC_DIR / "travel_times.parquet",
        filters=[("route_id", "in", demo_route_ids)],
    )
    travel.to_parquet(DST_DIR / "travel_times.parquet", index=False)
    print(f"travel_times.parquet: filtered to {len(travel)} rows ({len(demo_route_ids)} routes), "
          f"{(DST_DIR / 'travel_times.parquet').stat().st_size / 1e6:.2f} MB")

    total_mb = sum((DST_DIR / f).stat().st_size for f in
                    ["routes.parquet", "stops.parquet", "travel_times.parquet"]) / 1e6
    print(f"\ndata/deploy/ total: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
