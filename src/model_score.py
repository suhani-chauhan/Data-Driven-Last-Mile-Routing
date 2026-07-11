# Scoring logic ported from https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py (MIT License, Copyright (c) 2021 MIT Center for Transportation & Logistics)
"""Score a solved route against Amazon's own Last-Mile Routing Research Challenge metric.

The scoring functions below (normalize_matrix, dist_erp, gap_sum,
erp_per_edit_helper, erp_per_edit, seq_dev, isinvalid, score) are a line-for-line
port of the official scoring code at
https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py.

Attribution: this is the official companion CLI/scoring repo for the 2021 Amazon
Last-Mile Routing Research Challenge, published by MIT-CAVE (MIT Center for
Transportation & Logistics, which co-organized the challenge with Amazon).

    MIT License
    Copyright (c) 2021 MIT Center for Transportation & Logistics
    https://github.com/MIT-CAVE/rc-cli/blob/main/license

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Verification note: the first fetch of that file went through WebFetch, which does
not return raw bytes -- its own docs say it "processes the content with a prompt
using a small, fast model" and returns that model's response, so treating that
result as verbatim source overstated the actual certainty at the time. Re-fetched
via direct HTTP (curl) to get the real file bytes, then cross-checked by *running*
both files on identical route data rather than eyeballing a diff: the genuinely-
fetched official score() and this module's score() produce bit-for-bit identical
output (0.40715019833719274 on a shuffled-route test case, 0.17492527016752413 on
the actual OR-Tools-solved 33-stop route), and route2list()'s output matches this
module's submitted_sequence_list() exactly. This is a verified port, not a
reconstruction -- only I/O (reading this project's Parquet tables instead of the
challenge's raw JSON submission format) is new.

score(actual, sub, cost_mat) = seq_dev(actual, sub) * erp_per_edit(actual, sub, normalize_matrix(cost_mat))

- seq_dev: strips the depot from both sequences, then for each adjacent pair of
  stops in the submitted order, takes the gap between their positions in the
  *actual* order minus 1, sums that, and normalizes by n(n-1)/2. 0 when every
  adjacent pair in the submission is also adjacent in the actual route; grows as
  the submission increasingly reorders/backtracks relative to the actual route.
- erp_per_edit: Edit Distance with Real Penalty between the two depot-closed
  sequences (each ends by repeating its start stop), using a z-score-normalized,
  then non-negative-shifted travel-time matrix as the substitution cost and a
  fixed gap penalty g (default 1000) for insertions/deletions, divided by the
  number of edits in the optimal alignment (0 if there are 0 edits).
- Final score: 0.0 = identical sequences, higher = worse. No upper bound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_GAP_PENALTY = 1000


# ---------------------------------------------------------------------------
# Verbatim port of https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py
# ---------------------------------------------------------------------------


def normalize_matrix(mat: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    new_mat = {origin: dict(dests) for origin, dests in mat.items()}
    time_list = [mat[origin][dest] for origin in mat for dest in mat[origin]]
    avg_time = np.mean(time_list)
    std_time = np.std(time_list)
    min_new_time = np.inf
    for origin in mat:
        for dest in mat[origin]:
            new_time = (mat[origin][dest] - avg_time) / std_time
            if new_time < min_new_time:
                min_new_time = new_time
            new_mat[origin][dest] = new_time
    for origin in new_mat:
        for dest in new_mat[origin]:
            new_mat[origin][dest] -= min_new_time
    return new_mat


def gap_sum(path: list[str], g: float) -> float:
    return sum(g for _ in path)


def dist_erp(p_1: str, p_2: str, mat: dict[str, dict[str, float]], g: float = 1000) -> float:
    if p_1 == "gap" or p_2 == "gap":
        return g
    return mat[p_1][p_2]


def erp_per_edit_helper(
    actual: list[str], sub: list[str], matrix: dict[str, dict[str, float]], g: float = 1000, memo: dict | None = None
) -> tuple[float, int]:
    if memo is None:
        memo = {}
    key = (tuple(actual), tuple(sub))
    if key in memo:
        return memo[key]
    if len(sub) == 0:
        d = gap_sum(actual, g)
        count = len(actual)
    elif len(actual) == 0:
        d = gap_sum(sub, g)
        count = len(sub)
    else:
        head_actual, head_sub = actual[0], sub[0]
        rest_actual, rest_sub = actual[1:], sub[1:]
        score1, count1 = erp_per_edit_helper(rest_actual, rest_sub, matrix, g, memo)
        score2, count2 = erp_per_edit_helper(rest_actual, sub, matrix, g, memo)
        score3, count3 = erp_per_edit_helper(actual, rest_sub, matrix, g, memo)
        option_1 = score1 + dist_erp(head_actual, head_sub, matrix, g)
        option_2 = score2 + dist_erp(head_actual, "gap", matrix, g)
        option_3 = score3 + dist_erp(head_sub, "gap", matrix, g)
        d = min(option_1, option_2, option_3)
        if d == option_1:
            count = count1 if head_actual == head_sub else count1 + 1
        elif d == option_2:
            count = count2 + 1
        else:
            count = count3 + 1
    memo[key] = (d, count)
    return d, count


def erp_per_edit(actual: list[str], sub: list[str], matrix: dict[str, dict[str, float]], g: float = 1000) -> float:
    total, count = erp_per_edit_helper(actual, sub, matrix, g)
    return total / count if count else 0


def seq_dev(actual: list[str], sub: list[str]) -> float:
    actual = actual[1:-1]
    sub = sub[1:-1]
    comp_list = [actual.index(i) for i in sub]
    comp_sum = 0
    for ind in range(1, len(comp_list)):
        comp_sum += abs(comp_list[ind] - comp_list[ind - 1]) - 1
    n = len(actual)
    return (2 / (n * (n - 1))) * comp_sum


def isinvalid(actual: list[str], sub: list[str]) -> bool:
    if len(actual) != len(sub) or set(actual) != set(sub):
        return True
    if actual[0] != sub[0]:
        return True
    return False


def score(actual: list[str], sub: list[str], cost_mat: dict[str, dict[str, float]], g: float = 1000) -> float:
    norm_mat = normalize_matrix(cost_mat)
    return seq_dev(actual, sub) * erp_per_edit(actual, sub, norm_mat, g)


# ---------------------------------------------------------------------------
# Adapters from this project's Parquet schema to the list/dict shapes above.
# route2list in the original operates on a {stop_code: order} dict read from raw
# JSON; here that's stops.parquet's stop_sequence_order column (already 0-indexed
# with the depot at 0, same convention).
# ---------------------------------------------------------------------------


def actual_sequence_list(stops_df: pd.DataFrame) -> list[str]:
    order_to_code = {int(row.stop_sequence_order): row.stop_code for row in stops_df.itertuples(index=False)}
    n = len(order_to_code)
    route_list = [order_to_code[i] for i in range(n)]
    route_list.append(route_list[0])  # closed loop, matching the official route2list
    return route_list


def submitted_sequence_list(solved_stop_codes: list[str]) -> list[str]:
    return solved_stop_codes + [solved_stop_codes[0]]


def cost_matrix_dict(travel_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    mat: dict[str, dict[str, float]] = {}
    for row in travel_df.itertuples(index=False):
        mat.setdefault(row.from_stop, {})[row.to_stop] = row.travel_time_seconds
    return mat


# ---------------------------------------------------------------------------


def main() -> None:
    from model_apply import load_route, pick_default_route, solve

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route-id", default=None)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--time-limit-seconds", type=int, default=30)
    ap.add_argument("--gap-penalty", type=float, default=DEFAULT_GAP_PENALTY)
    args = ap.parse_args()

    route_id = args.route_id or pick_default_route(args.processed_dir)
    route = load_route(args.processed_dir, route_id)
    order = solve(route, args.time_limit_seconds)
    if order is None:
        raise SystemExit(f"no feasible solution found for {route_id!r} within {args.time_limit_seconds}s")
    solved_codes = [route.node_codes[i] for i in order]

    stops_df = pd.read_parquet(args.processed_dir / "stops.parquet", filters=[("route_id", "==", route_id)])
    travel_df = pd.read_parquet(args.processed_dir / "travel_times.parquet", filters=[("route_id", "==", route_id)])

    actual = actual_sequence_list(stops_df)
    sub = submitted_sequence_list(solved_codes)
    cost_mat = cost_matrix_dict(travel_df)

    print(f"route: {route_id}")
    print(f"stops (excl. closing repeat): {len(sub) - 1}")
    if isinvalid(actual, sub):
        print("INVALID per Amazon's own check (stop set/length/start mismatch) -- no score computed")
        return

    s = score(actual, sub, cost_mat, g=args.gap_penalty)
    print(f"official Amazon score: {s:.4f}   (0.0 = identical to actual driver route, higher = worse, no upper bound)")


if __name__ == "__main__":
    main()
