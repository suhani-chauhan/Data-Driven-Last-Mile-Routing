"""Convert the raw almrrc2021 JSON files into cleaned Parquet tables.

Source files (data/raw/.../model_build_inputs/): route_data.json, package_data.json,
travel_times.json, actual_sequences.json. All four are top-level JSON objects keyed by
RouteID_<uuid>; within a route, RouteID + 2-letter stop code is the join key used to
line up route_data's `stops`, package_data's per-stop packages, and
actual_sequences's per-stop visit order.

Missing-value handling (explicit, not silently passed through):
- package_data.json: `time_window.start_time_utc` / `end_time_utc` are NaN for ~92%
  of packages (measured on the full training set). This is not corrupt data -- it
  means "no delivery time-window constraint applies". Dropping those rows would
  discard 92% of packages; inventing a time would fabricate a constraint that does
  not exist. So the row is kept, the window is stored as an explicit SQL NULL
  (nullable timestamp), and a `has_time_window` boolean is added so consumers don't
  have to infer the meaning of a null themselves.
- route_data.json: `stops[*].zone_id` is NaN for Station-type stops (and a few
  Dropoffs) that have no delivery zone. Kept as an explicit nullable string for the
  same reason -- it's a real "not applicable", not bad data.

Parsing strategy per file (measured, not assumed -- see PIPELINE_NOTES.md-equivalent
reasoning inline below): the raw files contain bare NaN/Infinity tokens (from
`json.dump(..., allow_nan=True)`), which stdlib `json` accepts natively (its default
`parse_constant` maps them to float('nan') / inf / -inf) but ijson's strict streaming
parser rejects.
- route_data.json (79 MB) and actual_sequences.json (9.6 MB) are small enough to load
  fully with plain `json.load` (measured peak RSS 430 MiB / 53 MiB) -- gets NaN
  handling for free, no custom code.
- package_data.json (375 MB) is both too large to fully load (measured peak 1.3 GiB,
  unsafe against this machine's <1.5 GiB free RAM) and contains NaN, so it's streamed
  via `json.JSONDecoder.raw_decode` over a growing buffer (stdlib, NaN-safe by
  construction -- see `_iter_json_records`), rather than routing it through ijson
  behind a hand-rolled NaN-to-null byte filter (an earlier version of this file did
  that; it took two rounds of real bugs -- a premature-EOF signal and a token split
  across a chunk boundary -- to get right, which is exactly the "custom low-level
  code" this approach avoids).
- travel_times.json (1.8 GB, no NaN) is streamed with plain `ijson.kvitems` -- ijson
  is the standard tool for this and there's no NaN incompatibility to work around.

Out-of-core batching for packages.parquet and stops.parquet (whose row counts scale
with total route count, ~1.46M and ~0.9M rows respectively at full scale) is handled
by `dask.dataframe.read_json(..., lines=True, blocksize=...)` rather than a hand-
rolled "flush every N routes" counter: the reshape pass just appends one JSON line
per row with no batching logic at all, and Dask decides partition boundaries by byte
size. Verified out-of-core (see PIPELINE_NOTES): peak RSS stayed ~300-330 MiB whether
reading a 500-route or a simulated 6112-route-scale JSONL file.
travel_times.parquet (~137M rows at full scale) stays on the existing
`pyarrow.parquet.ParquetWriter` incremental-write pattern -- that's the documented
approach for Parquet files larger than memory, and round-tripping 137M rows through
JSON text would produce a multi-GB intermediate for no benefit.
"""
from __future__ import annotations

import datetime as dt
import json
import mmap
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
import dask
import dask.dataframe as dd

DEFAULT_OUTPUT_DIR = Path("data/processed")

_RAW_ROOT = Path("data/raw")


@dataclass(frozen=True)
class DataSource:
    split: str  # "train" or "eval" -- lands in the split column so a null route_score
    # or scan_status on an eval row is never confused with the training-set NaN cases
    # (missing time windows, missing zone_id) that already get explicit-null handling.
    route_data_path: Path
    package_data_path: Path
    travel_times_path: Path
    sequences_path: Path


# The training and evaluation releases use different directory layouts: training
# keeps all four files together under model_build_inputs/ with plain names; the
# evaluation release splits them across model_apply_inputs/ (the eval_-prefixed
# route/package/travel-time files a model is meant to predict from) and
# model_score_inputs/ (eval_actual_sequences.json, ground truth used only to score a
# submitted prediction afterward -- never given to a model as input). Evaluation's
# route_data/package_data also genuinely lack route_score/scan_status, since those
# are outcome fields that would leak the answer if included in "apply" inputs.
TRAIN_SOURCE = DataSource(
    split="train",
    route_data_path=_RAW_ROOT / "almrrc2021-data-training/model_build_inputs/route_data.json",
    package_data_path=_RAW_ROOT / "almrrc2021-data-training/model_build_inputs/package_data.json",
    travel_times_path=_RAW_ROOT / "almrrc2021-data-training/model_build_inputs/travel_times.json",
    sequences_path=_RAW_ROOT / "almrrc2021-data-training/model_build_inputs/actual_sequences.json",
)
EVAL_SOURCE = DataSource(
    split="eval",
    route_data_path=_RAW_ROOT / "almrrc2021-data-evaluation/model_apply_inputs/eval_route_data.json",
    package_data_path=_RAW_ROOT / "almrrc2021-data-evaluation/model_apply_inputs/eval_package_data.json",
    travel_times_path=_RAW_ROOT / "almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json",
    sequences_path=_RAW_ROOT / "almrrc2021-data-evaluation/model_score_inputs/eval_actual_sequences.json",
)
ALL_SOURCES = [TRAIN_SOURCE, EVAL_SOURCE]

TRAVEL_TIMES_FLUSH_EVERY = 10  # routes per Parquet row group; keeps peak RSS well under 300 MiB

# Dask's out-of-core partition size for packages/stops JSONL, and the scheduler used
# to process them. The default threaded scheduler processes several partitions
# concurrently, which multiplies peak memory by however many run at once (measured
# ~546 MiB at blocksize=4MB); the synchronous (single-threaded) scheduler processes
# one partition at a time, which is what actually keeps memory bounded regardless of
# total dataset size on a machine this memory-constrained (measured ~270-290 MiB at
# blocksize 2-4MB on a simulated full-scale file) -- worth the lost parallelism here.
DASK_BLOCKSIZE = "4MB"
DASK_SCHEDULER = "synchronous"

# ---------------------------------------------------------------------------
# Streaming reader for package_data.json: stdlib json.JSONDecoder.raw_decode over a
# growing buffer. raw_decode parses one JSON value starting at an offset and reports
# where it ended -- it doesn't require the rest of the buffer to be valid JSON, so
# growing the window until it succeeds is enough to pull one top-level record at a
# time out of a huge {key: value, ...} file without loading the whole thing.
# ---------------------------------------------------------------------------

_decoder = json.JSONDecoder()


def _iter_json_records(path: Path) -> Iterator[tuple[str, dict]]:
    """Yield (key, value) for each top-level entry of a huge {key: value, ...} JSON
    file without loading it all into memory. mmap exposes the file for lazy,
    OS-paged access; raw_decode parses one JSON value at a given offset and reports
    where it ended without requiring the rest of the buffer to be valid JSON, so
    growing the decoded window until it succeeds is enough to pull records out one
    at a time. Both mmap and json.JSONDecoder.raw_decode are stdlib.
    """
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n = len(mm)
        pos = mm.find(b"{") + 1
        try:
            while True:
                while mm[pos] in b" \t\r\n,":
                    pos += 1
                if mm[pos : pos + 1] == b"}":
                    return
                key, key_end = _raw_decode_growing(mm, pos, n)
                pos += key_end
                while mm[pos] in b" \t\r\n":
                    pos += 1
                assert mm[pos : pos + 1] == b":"
                pos += 1
                while mm[pos] in b" \t\r\n":
                    pos += 1
                value, val_end = _raw_decode_growing(mm, pos, n)
                pos += val_end
                yield key, value
        finally:
            mm.close()


def _raw_decode_growing(mm: mmap.mmap, pos: int, n: int, chunk: int = 1 << 20, grow: int = 8) -> tuple[object, int]:
    size = chunk
    while True:
        text = mm[pos : pos + size].decode("utf-8")
        try:
            return _decoder.raw_decode(text, 0)
        except json.JSONDecodeError:
            size *= grow
            if pos + size > n:
                size = n - pos


def _iter_ijson_records(path: Path) -> Iterator[tuple[str, dict]]:
    with open(path, "rb") as f:
        yield from ijson.kvitems(f, "", use_float=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ROUTES_SCHEMA = pa.schema(
    [
        ("route_id", pa.string()),
        ("split", pa.dictionary(pa.int8(), pa.string())),
        ("station_code", pa.string()),
        ("route_date", pa.date32()),
        ("departure_time_utc", pa.time32("s")),
        ("executor_capacity_cm3", pa.float64()),
        ("route_score", pa.dictionary(pa.int8(), pa.string())),  # null for split="eval": outcome field, not released
        ("stop_count", pa.int32()),
    ]
)

STOPS_SCHEMA = pa.schema(
    [
        ("route_id", pa.string()),
        ("split", pa.dictionary(pa.int8(), pa.string())),
        ("stop_code", pa.string()),
        ("lat", pa.float64()),
        ("lng", pa.float64()),
        ("stop_type", pa.dictionary(pa.int8(), pa.string())),
        ("zone_id", pa.string()),
        ("package_count", pa.int32()),
        # delivered/rejected/delivery_attempted_count are always 0 for split="eval" --
        # scan_status isn't released for eval packages, not because nothing was rejected.
        ("delivered_count", pa.int32()),
        ("rejected_count", pa.int32()),
        ("delivery_attempted_count", pa.int32()),
        ("total_planned_service_time_seconds", pa.float64()),
        ("total_volume_cm3", pa.float64()),
        ("has_time_window", pa.bool_()),
        ("window_start_utc", pa.timestamp("s")),
        ("window_end_utc", pa.timestamp("s")),
        ("stop_sequence_order", pa.int32()),
    ]
)

PACKAGES_SCHEMA = pa.schema(
    [
        ("route_id", pa.string()),
        ("split", pa.dictionary(pa.int8(), pa.string())),
        ("stop_code", pa.string()),
        ("package_id", pa.string()),
        ("scan_status", pa.dictionary(pa.int8(), pa.string())),  # null for split="eval": outcome field, not released
        ("planned_service_time_seconds", pa.float64()),
        ("depth_cm", pa.float64()),
        ("height_cm", pa.float64()),
        ("width_cm", pa.float64()),
        ("volume_cm3", pa.float64()),
        ("has_time_window", pa.bool_()),
        ("window_start_utc", pa.timestamp("s")),
        ("window_end_utc", pa.timestamp("s")),
    ]
)

TRAVEL_TIMES_SCHEMA = pa.schema(
    [
        ("route_id", pa.string()),
        ("from_stop", pa.string()),
        ("to_stop", pa.string()),
        ("travel_time_seconds", pa.float64()),
    ]
)

STOPS_DTYPES = {
    "route_id": "string",
    "split": "category",
    "stop_code": "string",
    "lat": "float64",
    "lng": "float64",
    "stop_type": "category",
    "zone_id": "string",
    "package_count": "int32",
    "delivered_count": "int32",
    "rejected_count": "int32",
    "delivery_attempted_count": "int32",
    "total_planned_service_time_seconds": "float64",
    "total_volume_cm3": "float64",
    "has_time_window": "bool",
    "stop_sequence_order": "Int32",
}
STOPS_DATETIME_COLS = ["window_start_utc", "window_end_utc"]

PACKAGES_DTYPES = {
    "route_id": "string",
    "split": "category",
    "stop_code": "string",
    "package_id": "string",
    "scan_status": "category",
    "planned_service_time_seconds": "float64",
    "depth_cm": "float64",
    "height_cm": "float64",
    "width_cm": "float64",
    "volume_cm3": "float64",
    "has_time_window": "bool",
}
PACKAGES_DATETIME_COLS = ["window_start_utc", "window_end_utc"]


# ---------------------------------------------------------------------------
# Row builders (pure transforms, independent of how the source was parsed)
# ---------------------------------------------------------------------------


def _nan_to_none(v):
    # json's default parse_constant maps the dataset's bare NaN tokens to float('nan'),
    # not None -- for non-numeric fields (zone_id, time windows) that NaN stands in for
    # "missing", so convert it to a real null explicitly rather than let a stray float
    # NaN end up in a string/datetime column.
    return None if isinstance(v, float) and v != v else v


def _parse_window(raw) -> dt.datetime | None:
    raw = _nan_to_none(raw)
    if raw is None:
        return None
    return dt.datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def _empty_package_agg() -> dict:
    return {
        "package_count": 0,
        "delivered_count": 0,
        "rejected_count": 0,
        "delivery_attempted_count": 0,
        "total_planned_service_time_seconds": 0.0,
        "total_volume_cm3": 0.0,
        "has_time_window": False,
        "window_start_utc": None,
        "window_end_utc": None,
    }


def _route_row(route_id: str, rec: dict, split: str) -> dict:
    return {
        "route_id": route_id,
        "split": split,
        "station_code": rec["station_code"],
        "route_date": dt.datetime.strptime(rec["date_YYYY_MM_DD"], "%Y-%m-%d").date(),
        "departure_time_utc": dt.datetime.strptime(rec["departure_time_utc"], "%H:%M:%S").time(),
        "executor_capacity_cm3": rec["executor_capacity_cm3"],
        "route_score": rec.get("route_score"),  # absent (not NaN) for split="eval"
        "stop_count": len(rec["stops"]),
    }


def _package_rows_and_agg(route_id: str, rec: dict, split: str) -> tuple[list[dict], dict[str, dict]]:
    """Package rows and per-stop aggregates for a single route (route-local only --
    this is what lets stops.parquet be joined without a dataset-wide aggregate dict)."""
    package_rows = []
    agg: dict[str, dict] = {}
    for stop_code, packages in rec.items():
        a = agg.setdefault(stop_code, _empty_package_agg())
        for package_id, pkg in packages.items():
            tw = pkg["time_window"]
            start = _parse_window(tw["start_time_utc"])
            end = _parse_window(tw["end_time_utc"])
            dims = pkg["dimensions"]
            volume_cm3 = dims["depth_cm"] * dims["height_cm"] * dims["width_cm"]
            scan_status = pkg.get("scan_status")  # absent (not NaN) for split="eval"
            package_rows.append(
                {
                    "route_id": route_id,
                    "split": split,
                    "stop_code": stop_code,
                    "package_id": package_id,
                    "scan_status": scan_status,
                    "planned_service_time_seconds": pkg["planned_service_time_seconds"],
                    "depth_cm": dims["depth_cm"],
                    "height_cm": dims["height_cm"],
                    "width_cm": dims["width_cm"],
                    "volume_cm3": volume_cm3,
                    "has_time_window": start is not None,
                    "window_start_utc": start,
                    "window_end_utc": end,
                }
            )
            a["package_count"] += 1
            a["delivered_count"] += scan_status == "DELIVERED"
            a["rejected_count"] += scan_status == "REJECTED"
            a["delivery_attempted_count"] += scan_status == "DELIVERY_ATTEMPTED"
            a["total_planned_service_time_seconds"] += pkg["planned_service_time_seconds"]
            a["total_volume_cm3"] += volume_cm3
            if start is not None:
                a["has_time_window"] = True
                a["window_start_utc"] = start if a["window_start_utc"] is None else min(a["window_start_utc"], start)
                a["window_end_utc"] = end if a["window_end_utc"] is None else max(a["window_end_utc"], end)
    return package_rows, agg


def _stop_rows(route_id: str, route_rec: dict, package_agg: dict[str, dict], sequence: dict[str, int], split: str) -> list[dict]:
    rows = []
    for stop_code, stop in route_rec["stops"].items():
        a = package_agg.get(stop_code, _empty_package_agg())
        rows.append(
            {
                "route_id": route_id,
                "split": split,
                "stop_code": stop_code,
                "lat": stop["lat"],
                "lng": stop["lng"],
                "stop_type": stop["type"],
                "zone_id": _nan_to_none(stop.get("zone_id")),
                **a,
                "stop_sequence_order": sequence.get(stop_code),
            }
        )
    return rows


def _travel_time_rows(route_id: str, matrix: dict) -> list[dict]:
    return [
        {"route_id": route_id, "from_stop": from_stop, "to_stop": to_stop, "travel_time_seconds": seconds}
        for from_stop, row in matrix.items()
        for to_stop, seconds in row.items()
    ]


def _json_default(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return str(v)
    raise TypeError(v)


# ---------------------------------------------------------------------------
# travel_times.parquet -- streamed with ijson + pyarrow's incremental ParquetWriter
# (the documented pattern for Parquet output larger than memory).
# ---------------------------------------------------------------------------


def build_travel_times_parquet(sources: list[DataSource], out_path: Path) -> int:
    # route_id is a globally unique UUID across splits (verified no overlap between
    # train and eval route IDs), and travel times have no missing-value/outcome
    # ambiguity, so unlike routes/packages/stops this table doesn't need a split
    # column -- one can always join back to routes.parquet on route_id for that.
    writer = pq.ParquetWriter(out_path, TRAVEL_TIMES_SCHEMA)
    buffer: list[dict] = []
    routes_seen = 0
    total_rows = 0
    try:
        for source in sources:
            for route_id, matrix in _iter_ijson_records(source.travel_times_path):
                buffer.extend(_travel_time_rows(route_id, matrix))
                routes_seen += 1
                if routes_seen % TRAVEL_TIMES_FLUSH_EVERY == 0:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=TRAVEL_TIMES_SCHEMA))
                    total_rows += len(buffer)
                    buffer = []
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=TRAVEL_TIMES_SCHEMA))
            total_rows += len(buffer)
    finally:
        writer.close()
    return total_rows


# ---------------------------------------------------------------------------
# packages.parquet + stops.parquet -- reshape pass writes plain JSON-lines (no
# batching logic at all, just one line per row); Dask decides partitioning and
# handles the out-of-core Parquet write.
# ---------------------------------------------------------------------------


def _write_packages_and_stops_jsonl(source: DataSource, route_data: dict, sequences: dict, pkg_f, stop_f) -> None:
    for route_id, pkg_rec in _iter_json_records(source.package_data_path):
        route_rec = route_data[route_id]
        seq = sequences[route_id]["actual"]
        package_rows, package_agg = _package_rows_and_agg(route_id, pkg_rec, source.split)
        for row in package_rows:
            pkg_f.write(json.dumps(row, default=_json_default))
            pkg_f.write("\n")
        for row in _stop_rows(route_id, route_rec, package_agg, seq, source.split):
            stop_f.write(json.dumps(row, default=_json_default))
            stop_f.write("\n")


def _jsonl_to_parquet(jsonl_path: Path, out_path: Path, dtypes: dict, datetime_cols: list[str], schema: pa.Schema) -> int:
    # dtype="object" forces every column to a consistent raw dtype at read time.
    # Without it, pandas infers a dtype per partition independently, and a column
    # that's entirely null within one partition (e.g. scan_status across an
    # eval-only partition, since eval packages never have scan_status) gets
    # inferred as float64 there vs object in partitions that have real string
    # values -- Dask's cross-partition meta check then fails before the explicit
    # .astype() below even runs. Reproduced and confirmed fixed at full scale.
    with dask.config.set(scheduler=DASK_SCHEDULER):
        ddf = dd.read_json(jsonl_path, lines=True, blocksize=DASK_BLOCKSIZE, dtype="object")
        ddf = ddf.astype(dtypes)
        for col in datetime_cols:
            ddf[col] = dd.to_datetime(ddf[col])
        ddf.to_parquet(out_path, write_index=False, schema=schema)
        return len(ddf)


# ---------------------------------------------------------------------------


def run(sources: list[DataSource], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    route_rows = []
    packages_jsonl = output_dir / "_packages.jsonl.tmp"
    stops_jsonl = output_dir / "_stops.jsonl.tmp"
    with open(packages_jsonl, "w", encoding="utf-8") as pkg_f, open(stops_jsonl, "w", encoding="utf-8") as stop_f:
        for source in sources:
            with open(source.route_data_path, encoding="utf-8") as f:
                route_data = json.load(f)
            with open(source.sequences_path, encoding="utf-8") as f:
                sequences = json.load(f)

            route_rows.extend(_route_row(route_id, rec, source.split) for route_id, rec in route_data.items())
            _write_packages_and_stops_jsonl(source, route_data, sequences, pkg_f, stop_f)
            # route_data/sequences for this source aren't needed past this point -- drop
            # them before moving to the next source rather than let every split's copy
            # stay alive simultaneously.
            del route_data, sequences

    pq.write_table(pa.Table.from_pylist(route_rows, schema=ROUTES_SCHEMA), output_dir / "routes.parquet")
    print(f"routes.parquet: {len(route_rows)} rows")
    del route_rows

    n_packages = _jsonl_to_parquet(packages_jsonl, output_dir / "packages.parquet", PACKAGES_DTYPES, PACKAGES_DATETIME_COLS, PACKAGES_SCHEMA)
    print(f"packages.parquet: {n_packages} rows")
    n_stops = _jsonl_to_parquet(stops_jsonl, output_dir / "stops.parquet", STOPS_DTYPES, STOPS_DATETIME_COLS, STOPS_SCHEMA)
    print(f"stops.parquet: {n_stops} rows")
    packages_jsonl.unlink()
    stops_jsonl.unlink()

    n_tt = build_travel_times_parquet(sources, output_dir / "travel_times.parquet")
    print(f"travel_times.parquet: {n_tt} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    args = ap.parse_args()
    sources = {"train": [TRAIN_SOURCE], "eval": [EVAL_SOURCE], "all": ALL_SOURCES}[args.split]
    run(sources, args.output_dir)


if __name__ == "__main__":
    main()
