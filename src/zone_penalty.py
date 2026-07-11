"""Zone-transition-frequency penalty -- the fallback for the brief's P_ij term
in C_ij = T_ij + alpha * P_ij, used in place of the XGBoost per-stop difficulty
model. That model's validation R^2 stayed negative even after a feature swap
(see docs/project_brief.md's Known Limitations), so per an explicit stop
condition it wasn't wired into the live cost matrix. model_build.py and its
serialized artifacts stay in the repo as documented exploration; this module
is what model_apply.py actually uses.

P_ij = 1 - P(zone(j) | zone(i)), the empirical transition probability from
zone i to zone j observed across every consecutive pair of stops in every
TRAINING route's actual driver sequence (stops.stop_sequence_order). This is
a direct frequency-count implementation of the brief's own definition
("1 - Probability of Transition") -- the lightweight stand-in for the PPM
Markov idea that was dropped from the core deliverable.

Every stop needs a zone label to be looked up, including ones the raw data
doesn't give one to: the depot's own zone_id is always null (it's not a
delivery zone), so it gets the sentinel "DEPOT"; the ~0.76% of Dropoff stops
missing a real zone_id get "UNKNOWN". This means model_apply.py never needs
to special-case a stop for having no zone.

Zone pairs never observed in training -- including any zone that never
appears as an origin at all -- fall back to the training-set-wide average
penalty (the weighted mean of 1 - P(j|i) over every transition actually
observed), not a flat constant: "we have no data on this transition" is a
different situation from "we have data showing drivers avoid it," and 1.0
(max penalty) or 0.0 (no penalty) would silently conflate the two.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

DEPOT_ZONE = "DEPOT"
UNKNOWN_ZONE = "UNKNOWN"


class ZonePenaltyTable:
    def __init__(self, counts: dict[str, dict[str, int]], fallback: float):
        self._counts = counts
        self._totals = {zi: sum(dests.values()) for zi, dests in counts.items()}
        self.fallback = fallback

    def get(self, zone_i: str, zone_j: str) -> float:
        total = self._totals.get(zone_i)
        if not total:
            return self.fallback
        count = self._counts[zone_i].get(zone_j, 0)
        return 1 - count / total


def zone_of(stop_type: str, zone_id) -> str:
    if stop_type == "Station":
        return DEPOT_ZONE
    if pd.isna(zone_id):
        return UNKNOWN_ZONE
    return zone_id


def build_pij_table(processed_dir: Path) -> ZonePenaltyTable:
    stops = pd.read_parquet(
        processed_dir / "stops.parquet",
        columns=["route_id", "stop_type", "zone_id", "stop_sequence_order"],
        filters=[("split", "==", "train")],
    )
    stops = stops.dropna(subset=["stop_sequence_order"])
    stops["zone"] = [zone_of(t, z) for t, z in zip(stops.stop_type, stops.zone_id)]
    stops = stops.sort_values(["route_id", "stop_sequence_order"])

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_transitions = 0
    for _, group in stops.groupby("route_id", sort=False):
        zones = group["zone"].tolist()
        for a, b in zip(zones, zones[1:]):
            counts[a][b] += 1
            total_transitions += 1

    # Training-set-wide average of the same penalty formula applied to every
    # observed transition, weighted by how often it occurred -- used as the
    # fallback for pairs/origins never seen in training.
    weighted_sum = 0.0
    for dests in counts.values():
        total_i = sum(dests.values())
        for c in dests.values():
            weighted_sum += c * (1 - c / total_i)
    fallback = weighted_sum / total_transitions if total_transitions else 0.5

    return ZonePenaltyTable({k: dict(v) for k, v in counts.items()}, fallback)
