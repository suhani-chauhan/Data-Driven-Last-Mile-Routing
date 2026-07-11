# Hybrid Machine Learning & Combinatorial Optimization for Data-Driven Last-Mile Routing

## Project Overview & Real-World Context

**The Core Problem:** Classic logistics systems use pure geometric math to find
the shortest route (the Traveling Salesperson Problem). However, human drivers
routinely deviate from these paths because they possess unwritten regional
context — such as knowing where it is impossible to park a delivery van, or
which alleyways allow quicker walkability.

**The Objective:** Build a dual-stage system that uses Machine Learning to
capture human driving preferences from historical data and converts those
patterns into mathematical "cost penalties." A Combinatorial Optimization
Solver then ingests these customized constraints to generate highly
efficient, driver-approved delivery sequences.

---

## Dataset Specifications

Official dataset: the **2021 Amazon Last Mile Routing Research Challenge
Dataset** (hosted on the AWS Open Data Registry).

- **Scale:** 9,184 historical routes (6,112 training, 3,072 evaluation)
  executed by actual Amazon drivers across 5 major U.S. metropolitan areas.
  > **Verified against the actual downloaded data (see Status note below):
  > the real eval split contains 3,052 routes, not 3,072 — 9,164 total, not
  > 9,184.** Confirmed directly against all four raw eval files independently
  > agreeing on route count. Treat 9,164 / 3,052 as ground truth going
  > forward; the numbers above are the original spec as written.

- **Data Structure:** Messy, nested JSON files split across key operational
  parameters:
  - `route_data.json` — date of route, departure time, vehicle dimensions,
    starting depot station ID.
  - `package_data.json` — package-level details (dimensions, weights,
    delivery time windows, exact latitude/longitude coordinates).
  - `travel_times.json` — a pre-computed, **asymmetric** point-to-point
    transit time matrix between all stop pairs on a given route.
  - `actual_sequences.json` — the ground-truth, step-by-step path sequence
    executed by the driver (used for training labels and metric evaluation).

- **Download command:**
  ```bash
  aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ ./data/
  ```

---

## Target Tech Stack

**Data Pipeline & Preprocessing**
- Storage & Compression: Apache Parquet & NumPy arrays (converting raw JSONs
  to Parquet speeds up feature-engineering read-times by up to 10x).
- Querying & Cleaning: Pandas — extract spatial clusters, handle missing
  coordinate values, normalize delivery time windows.
- S3 Client: AWS CLI, anonymous sync (see command above).

**Engineering and Modeling Engines**
- Machine Learning Framework: XGBoost, paired with Python's native handling
  of Prediction by Partial Matching (PPM) Markov structures.
- Mathematical Operations Solver: Google OR-Tools (Vehicle Routing
  Open-Source Library).
- Network & Graph Infrastructure: NetworkX — zone-to-zone spatial
  topologies, geographic centroids.

**Production & DevOps Packaging**
- Containerization: Docker (Ubuntu 20.04 base image) — pipeline runnable via
  standard CLI flags (`model-build`, `model-apply`, `model-score`) without
  environment dependency issues.

---

## Comparing the Existing Models

| Feature / Dimension | Framework A: "Pool & Select" | Framework B: "Two-Level Hierarchical" |
|---|---|---|
| **How it works** | Generates 50–100 candidate paths purely via math, then trains an ML model to score and choose the best one. | Uses ML to calculate transit preferences between geographical neighborhood clusters (zones) first, then applies math internally. |
| **Primary language** | C++ with custom parsers (e.g. rapidjson) for fast execution loops. | Python, clean tabular models (e.g. XGBoost) and structural arrays. |
| **ML/Optimization integration** | Loose — math generates blindly, ML acts as an outside judge. | Deep — ML explicitly alters the distance matrix used directly by the solver. |
| **Computational footprint** | Highly demanding; scales poorly. | Lightweight; zone grouping reduces complexity ~10x. |

**Chosen architecture: Two-Level Hierarchical.** More interpretable, faster
to train, mirrors real enterprise architecture, and better portfolio value
than raw C++ brute force.

---

## Mathematical Formulation & Optimization Function

Modify the cost function of an **Asymmetric Traveling Salesperson Problem
(ATSP)**. Instead of routing based solely on physical transit time (T_ij from
stop i to stop j), construct a **Hybrid Cost Function**:

```
C_ij = T_ij + α · P_ij
```

Where:
- `T_ij` — baseline geographic transit time from the data asset.
- `P_ij` — ML-derived penalty score (e.g. `1 - Probability of Transition`).
  If drivers rarely travel from zone i to zone j, `P_ij` spikes upward.
- `α` — tunable hyperparameter balancing pure travel efficiency against
  historical driver habit.

**Objective function** — minimize total weighted cost over all selected arcs
(`x_ij ∈ {0,1}`):

```
Minimize  Σ_i Σ_j  C_ij · x_ij
```

---

## Operational Constraints to Program

- **Asymmetric travel windows:** `T_ij ≠ T_ji` (one-way streets, turning loops).
- **Service time delays:** each stop has an immutable `service_time` penalty
  (parking, walking to the door).
- **Time-window hard targets:** premium packages have strict delivery
  windows (e.g. before 12:00 PM). Solver must flag violating paths as
  invalid.

---

## Core Machine Learning Techniques

- **Prediction by Partial Matching (PPM):** a compression-based Markov
  sequence model — feed historical zone transitions to calculate the
  sequential probability of a driver transitioning into an adjacent zone
  based on their previous two steps.
- **Gradient-Boosted Decision Trees (XGBoost):** extracted features (package
  count, aggregate zone volume, departure times) predict localized
  stop-level difficulty markers.

---

## How to Analyze and Evaluate the Project

Amazon uses a specialized variant of the **Kendall's Tau (τ) Distance**
metric to score submissions. It evaluates how many pairs of stops in the
predicted sequence are in the exact order chosen by the human driver.

**Modular script structure (`/src`):**
- `model_build.py` — ingests training Parquet assets, fits PPM/XGBoost
  structures, serializes weights.
- `model_apply.py` — loads serialized models, reads new target route
  requests, constructs the hybrid matrix, runs Google OR-Tools, outputs a
  JSON map sequence.
- `model_score.py` — mathematical comparison between output sequences and
  `actual_sequences.json`.

**Scoring scale:** a perfect replication of human behavior yields a score
of **0.0**. Standard geometric routing typically scores around **0.08 –
0.12**. The hybrid system's goal is to minimize this distance toward
**0.03 – 0.04**, reflecting elite competitive submissions.

**Portfolio visualization:** side-by-side plots — the erratic, jagged lines
of standard shortest-path routing vs. the hybrid system's smooth,
driver-friendly pathing curves.

---

## Project Status Update *(living section — update as work progresses)*

This section reflects decisions made and progress completed since the
original brief above was written. The section above is the untouched
original spec; treat this section as the current source of truth where the
two differ.

**Scope decisions made:**
- **PPM Markov dropped from the core deliverable.** Mentioned only as a
  possible future stretch goal in the README — not blocking the MVP.
  XGBoost is the sole ML component.
- **Simplified 3-milestone build order**, prioritizing a working
  end-to-end skeleton over perfecting any single stage in isolation:
  1. **Data pipeline** — parse raw JSON → clean Parquet.
  2. **OR-Tools baseline** — raw `T_ij` only, zero ML, to get any valid
     route out and prove the solver/scoring chain works.
  3. **XGBoost layer** — train difficulty scores, fold into the solver as
     the `α · P_ij` penalty term.
- Docker packaging deferred to the end, kept minimal (just enough to run
  the three CLI commands from the original brief).

**Data pipeline — complete:**
- Full dataset processed (9,164 routes total: 6,112 train / 3,052 eval).
- Output tables: `routes.parquet`, `packages.parquet`, `stops.parquet`,
  `travel_times.parquet` under `data/processed/`.
- Built using standard libraries wherever possible instead of hand-rolled
  code: stdlib `json` (`parse_constant`) for NaN handling where feasible,
  `ijson` for streaming the large `travel_times.json`, Dask
  (`scheduler="synchronous"`) for out-of-core batching on packages/stops.
  Verified memory-safe on an 8GB-RAM machine (~550 MiB peak).

**OR-Tools baseline — complete:**
- `model_apply.py` solves a single route as an ATSP using raw
  `travel_time_seconds` (no ML penalty), with a `Time` dimension enforcing
  `service_time` + transit + hard time-window constraints, using
  `PARALLEL_CHEAPEST_INSERTION` as the first-solution strategy (required
  for feasibility on tightly time-windowed routes — the default
  `PATH_CHEAPEST_ARC` failed on those).
- Verified on 4 test routes including a 100%-time-windowed stress case
  (33/33 stops constrained, zero scheduling slack) — solved cleanly.

**Scoring metric — RESOLVED:**
- No official Amazon scoring script was bundled with the dataset or found
  anywhere on the local machine, but the companion GitHub repo referenced
  by the challenge's own published methodology was located and fetched
  directly: `https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py`
  (MIT-CAVE = MIT Center for Transportation & Logistics, which co-organized
  the challenge with Amazon; MIT License, copyright MIT CTL — attribution
  and full license text are in `src/model_score.py`'s module docstring).
- The metric described above ("pairs of stops in the exact order") is the
  **ERP-based sequence-deviation metric** from Merchan et al. (2021):
  `score = seq_dev(actual, sub) × erp_per_edit(actual, sub, normalize_matrix(cost_mat), g)`.
  Confirmed **not** the standard `scipy.stats.kendalltau` rank-correlation
  coefficient (-1 to +1 scale) — that was mistakenly computed first
  (-0.144) before this distinction was caught.
- `src/model_score.py` is a **verified line-for-line port, not a
  reconstruction**. Verification method: a first pass via `WebFetch` was
  flagged as insufficient on its own (that tool summarizes fetched content
  through a model rather than returning raw bytes, so it doesn't guarantee
  byte-for-byte fidelity); re-fetched via direct HTTP instead, then
  cross-checked by *executing* both the genuinely-fetched official file and
  the local port on identical route data — bit-for-bit identical output on
  two test cases, not just a visual code diff:
  - Shuffled-route test case: `0.40715019833719274`
  - Real 33-stop, 100%-time-windowed OR-Tools-solved route: `0.17492527016752413`
- The 33-stop route's score is somewhat above the 0.08–0.12 naive-geometric
  baseline range, consistent with it being the stress-test case (zero
  scheduling slack, so the solver's ordering freedom is more constrained
  than a typical route).

**Known Limitations:**
- **Solver feasibility is ~90%, not 100%.** In a 30-route random sample from
  the training set (seed=42), 3/30 routes (10%) came back infeasible under
  `PARALLEL_CHEAPEST_INSERTION` within a 15s solve budget — each failed
  almost instantly (0.3–0.9s), not a near-miss timeout. Not yet root-caused.
- **One scoring-step timing outlier.** In that same 30-route run, one route
  (`RouteID_092c229e-9d6e-468b-accd-0372d87a181c`, 194 stops) took 1408.3s
  to score, vs. 15–18s for every other similarly-sized route (~150–200
  stops) in the sample. Not yet root-caused — plausibly in the ERP scoring
  step rather than the OR-Tools solve step, but unconfirmed.

**Hybrid cost function — headline result:**
- The XGBoost per-stop difficulty model (`model_build.py`) never reached a
  usable validation R² (see Known Limitations above and the model's own
  training output) — a feature swap moved it from -0.0721 to -0.0424, still
  not meaningfully positive, so per an explicit stop condition it was left
  out of the live cost matrix (kept in the repo as documented exploration)
  in favor of a lightweight zone-transition-frequency `P_ij`
  (`src/zone_penalty.py`): `P_ij = 1 - P(zone(j) | zone(i))`, the empirical
  transition probability observed across every training route's actual
  sequence — the direct frequency-count implementation of the brief's own
  `P_ij` definition, and the fallback for the dropped PPM idea.
- Wired into `model_apply.py` as `C_ij = T_ij + α · scale · P_ij`
  (α = 1.0, `scale` = each route's own mean `T_ij`), affecting only the
  arc-cost objective — the `Time` dimension's hard constraints stay on real
  T_ij + service_time, untouched.
- **Paired comparison, 15 random (not hand-picked) training routes
  (seed=99), each solved twice — with and without the penalty — at a
  consistent 60s budget per solve:**
  - Feasibility: 15/15 (100%) for both conditions — up from 27/30 (90%) in
    the earlier 15s-budget run.
  - Baseline average official score: 0.072827
  - Hybrid average official score: 0.049751
  - **Improvement: 31.69%** ((0.072827 − 0.049751) / 0.072827 × 100)
  - Both averages already land below the brief's 0.08–0.12 naive-geometric
    baseline range; the hybrid sits meaningfully closer to the brief's
    0.03–0.04 "elite competitive" target than the baseline does.
  - All 30 solves (15 routes × 2 conditions) used the full 60s budget —
    consistent with expected `GUIDED_LOCAL_SEARCH` behavior for this
    metaheuristic under a fixed wall-clock limit — so the 31.69% figure
    reflects best-within-budget performance for both conditions, not a
    fully converged optimum for either.
- Reproducible via `src/compare_baseline_vs_hybrid.py` (writes
  `results/comparison.csv`); visualized for the 33-stop stress-test route in
  `results/route_comparison.png` via `src/plot_route_comparison.py`.

**Not yet started:** full evaluation-set scoring, Dockerization.
