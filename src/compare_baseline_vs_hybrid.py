"""Fair headline comparison: pure-T_ij baseline vs. the zone-penalty hybrid
cost function (C_ij = T_ij + alpha * scale * P_ij, see model_apply.py and
zone_penalty.py), on a random (not hand-picked) sample of training routes.

Each sampled route is solved TWICE at the SAME time budget -- once with the
zone-transition penalty (alpha=1.0) and once without (pure T_ij) -- so the
two conditions are compared on equal footing. A shorter budget than 60s was
found to make OR-Tools' time-limited GUIDED_LOCAL_SEARCH inconsistent between
otherwise-identical runs (the same route scored 0.1749 at 60s and 0.2287 at
20s in manual testing), so an inconsistent-budget comparison isn't
trustworthy -- this script always uses one fixed --time-limit-seconds for
both conditions.

Reports, per route: baseline score, hybrid score, and each condition's solve
outcome (ok / infeasible / invalid). Then three averages:
- baseline average: over routes where the baseline solve succeeded
- hybrid average: over routes where the hybrid solve succeeded
- paired average: over routes where BOTH succeeded (the actual apples-to-
  apples headline number -- baseline-only and hybrid-only can differ in which
  routes they cover if one condition fails where the other doesn't)

Usage:
    python src/compare_baseline_vs_hybrid.py
    python src/compare_baseline_vs_hybrid.py --n-routes 30 --time-limit-seconds 30

No interactive input. Prints progress per route as it runs and writes the
full per-route results to --output-csv (default results/comparison.csv).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from model_apply import load_route, solve
from model_score import actual_sequence_list, cost_matrix_dict, isinvalid, score, submitted_sequence_list
from zone_penalty import build_pij_table

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT_CSV = Path("results/comparison.csv")
DEFAULT_N_ROUTES = 15
DEFAULT_TIME_LIMIT_SECONDS = 60
DEFAULT_SEED = 99
DEFAULT_ALPHA = 1.0


def run_one_route(processed_dir: Path, route_id: str, pij_table, alpha: float, time_limit_seconds: int) -> dict:
    route = load_route(processed_dir, route_id)
    stops_df = pd.read_parquet(processed_dir / "stops.parquet", filters=[("route_id", "==", route_id)])
    travel_df = pd.read_parquet(processed_dir / "travel_times.parquet", filters=[("route_id", "==", route_id)])
    actual = actual_sequence_list(stops_df)
    cost_mat = cost_matrix_dict(travel_df)

    row: dict = {"route_id": route_id, "stop_count": len(route.node_codes)}
    for label, pij, a in [("baseline", None, 0.0), ("hybrid", pij_table, alpha)]:
        t0 = time.time()
        order = solve(route, time_limit_seconds, pij_table=pij, alpha=a)
        solve_time = time.time() - t0
        if order is None:
            row[f"{label}_score"] = None
            row[f"{label}_status"] = "infeasible"
            row[f"{label}_solve_seconds"] = round(solve_time, 1)
            print(f"  {label}: INFEASIBLE ({solve_time:.1f}s)", flush=True)
            continue
        solved_codes = [route.node_codes[j] for j in order]
        sub = submitted_sequence_list(solved_codes)
        if isinvalid(actual, sub):
            row[f"{label}_score"] = None
            row[f"{label}_status"] = "invalid"
            row[f"{label}_solve_seconds"] = round(solve_time, 1)
            print(f"  {label}: INVALID solution ({solve_time:.1f}s)", flush=True)
            continue
        t_score = time.time()
        s = score(actual, sub, cost_mat)
        score_time = time.time() - t_score
        row[f"{label}_score"] = s
        row[f"{label}_status"] = "ok"
        row[f"{label}_solve_seconds"] = round(solve_time, 1)
        print(f"  {label}: score={s:.6f}  (solve {solve_time:.1f}s, score {score_time:.1f}s)", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    ap.add_argument("--n-routes", type=int, default=DEFAULT_N_ROUTES)
    ap.add_argument("--time-limit-seconds", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    routes = pd.read_parquet(args.processed_dir / "routes.parquet", columns=["route_id", "split", "stop_count"])
    train_routes = routes[routes.split == "train"]
    sample = train_routes.sample(n=args.n_routes, random_state=args.seed)
    print(f"sampled {args.n_routes} training routes (seed={args.seed}), stop_count range: "
          f"{sample.stop_count.min()}-{sample.stop_count.max()}")

    print("building zone-transition penalty table (once, reused for all routes)...", flush=True)
    t0 = time.time()
    pij_table = build_pij_table(args.processed_dir)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    results = []
    t_total = time.time()
    for i, route_id in enumerate(sample.route_id):
        stop_count = int(sample[sample.route_id == route_id].stop_count.iloc[0])
        print(f"\n[{i + 1}/{args.n_routes}] {route_id} ({stop_count} stops)", flush=True)
        row = run_one_route(args.processed_dir, route_id, pij_table, args.alpha, args.time_limit_seconds)
        results.append(row)
    total_time = time.time() - t_total
    print(f"\ntotal time: {total_time:.1f}s", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route_id", "stop_count",
        "baseline_score", "baseline_status", "baseline_solve_seconds",
        "hybrid_score", "hybrid_status", "hybrid_solve_seconds",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"saved per-route results to {args.output_csv}")

    print("\n" + "=" * 70)
    print(f"{'route_id':<48} {'baseline':>12} {'hybrid':>12}")
    for r in results:
        b = f"{r['baseline_score']:.6f}" if r["baseline_status"] == "ok" else r["baseline_status"].upper()
        h = f"{r['hybrid_score']:.6f}" if r["hybrid_status"] == "ok" else r["hybrid_status"].upper()
        print(f"{r['route_id']:<48} {b:>12} {h:>12}")

    baseline_scores = [r["baseline_score"] for r in results if r["baseline_status"] == "ok"]
    hybrid_scores = [r["hybrid_score"] for r in results if r["hybrid_status"] == "ok"]
    paired = [(r["baseline_score"], r["hybrid_score"]) for r in results if r["baseline_status"] == "ok" and r["hybrid_status"] == "ok"]

    n = args.n_routes
    print()
    print(f"baseline: {len(baseline_scores)}/{n} feasible ({n - len(baseline_scores)} infeasible/invalid)")
    print(f"hybrid:   {len(hybrid_scores)}/{n} feasible ({n - len(hybrid_scores)} infeasible/invalid)")
    if baseline_scores:
        print(f"baseline average (all feasible baseline runs): {sum(baseline_scores) / len(baseline_scores):.6f}")
    if hybrid_scores:
        print(f"hybrid average (all feasible hybrid runs):      {sum(hybrid_scores) / len(hybrid_scores):.6f}")
    if paired:
        b_paired = sum(b for b, h in paired) / len(paired)
        h_paired = sum(h for b, h in paired) / len(paired)
        print(f"\nPAIRED comparison (both conditions feasible on the same route, n={len(paired)}):")
        print(f"  baseline average: {b_paired:.6f}")
        print(f"  hybrid average:   {h_paired:.6f}")
        print(f"  delta:            {h_paired - b_paired:+.6f}  ({'hybrid better' if h_paired < b_paired else 'baseline better'})")
    else:
        print("\nno route had both conditions feasible -- no paired comparison possible")


if __name__ == "__main__":
    main()
