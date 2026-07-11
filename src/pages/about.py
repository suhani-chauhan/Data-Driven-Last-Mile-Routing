import streamlit as st

st.markdown(
    """## About This Project

Delivery drivers often take a different route than the "shortest path" a computer would calculate — because real streets, parking, and shortcuts matter more than straight-line distance.

This tool learns from thousands of real Amazon delivery drivers' actual historical routes, and tries to suggest delivery orders that match how real drivers actually work — not just the mathematically shortest path.

**What it's built on:** real route and package data from Amazon's public Last-Mile Routing Research Challenge dataset, a route-solving engine (Google OR-Tools), and patterns learned from historical driver behavior.

**What it isn't:** a claim that this finds the single "best" route. It's a comparison tool — showing how a route planned with real-driver patterns differs from one planned with straight-line math alone.

---

### How It Works

Every route is planned two different ways, so you can compare them side by side:

- **Straight-line route:** the "obvious" approach. For every pair of stops, the tool looks only at the raw travel time between them and finds the order that minimizes total driving time. This is what most basic route-planning software does.
- **Smart route:** starts from the same straight-line travel times, but adds a penalty based on how often real drivers historically moved between the same pair of delivery zones. If drivers almost never go straight from zone A to zone B — even though it looks efficient on a map — the tool treats that jump as "expensive" and looks for an order that avoids it, closer to how an experienced driver would actually work the area.

Both versions still have to respect the practical constraints of the route: how long the driver spends at each stop (service time), and any delivery time windows a package requires.

### The Dataset

This tool is built on the **2021 Amazon Last-Mile Routing Research Challenge** dataset — real, anonymized delivery data released publicly by Amazon for research purposes. It covers:

- **9,164 real delivery routes** (6,112 used to learn from, 3,052 held out for evaluation) across **5 major U.S. metro areas**.
- Real stop sequences, package details, delivery time windows, and the actual GPS-derived path each driver took.
- A full point-to-point travel-time estimate between every pair of stops on every route.

### The Technology

- **Google OR-Tools** — an open-source combinatorial optimization engine — actually solves each route as an Asymmetric Traveling Salesperson Problem (a route where the trip doesn't have to be symmetric or return to the start), respecting time windows and per-stop service time as hard constraints.
- **A zone-transition penalty**, learned by counting how often real drivers moved between each pair of delivery zones across the whole training dataset, powers the "smart route" behavior described above.
- **Scoring** uses the official metric from Amazon's own research challenge — a sequence-comparison algorithm (not a simple straight-line distance) that measures how closely a suggested order matches the real driver's actual order. A score of 0 means an exact match; there's no upper limit, but real-world routing approaches typically land somewhere between 0.03 (very close to real driver behavior) and 0.12 (a fairly literal shortest-path approach).

### Why This Matters

A mathematically "optimal" route on paper can still be a bad route in practice — if it ignores the fact that a driver already knows which loading dock is easiest to reach, which street reliably has parking, or which turn to avoid at a certain time of day. By learning directly from thousands of real routes instead of only optimizing distance, the goal is to suggest delivery orders that are not just theoretically efficient, but practically usable by an actual driver on an actual street."""
)
