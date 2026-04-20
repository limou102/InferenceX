#!/usr/bin/env python3
"""Summarize sweep log files into a single table.

Scans a directory (default ./sweep-logs) for client/driver logs named like::

    isl{ISL}_osl{OSL}_tp{TP}_ep{EP_ENABLED}_conc{CONC}.log

For each file it extracts the following metrics from the file content:

  * Total Token throughput (tok/s)
  * Token Throughput per GPU (tok/s/gpu)
  * Output Token Throughput per GPU (tok/s/gpu)
  * Interactivity (tok/s/user)
  * TTFT (ms)
  * TPOT (ms)

Logs that are missing any metric are reported and skipped.

Produces two outputs:
  * a nicely aligned table printed to stdout
  * a CSV file (default ``summary.csv`` inside the log dir)

Usage::

    python3 summarize_sweep_logs.py
    python3 summarize_sweep_logs.py --log-dir ./sweep-logs --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Filename + metric patterns
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^isl(?P<isl>\d+)_osl(?P<osl>\d+)_tp(?P<tp>\d+)_ep(?P<ep>\d+)_conc(?P<conc>\d+)\.log$"
)

# Each metric: (csv_column_name, compiled_regex, value_type)
#
# Regexes are anchored so that "Mean TTFT (ms)" / "Median TTFT (ms)" etc. from
# the benchmark_serving summary do NOT match our plain "TTFT (ms)" row which
# comes from the final Key Metrics block.
NUM = r"([-+]?\d+(?:\.\d+)?)"

METRIC_SPECS: List[Tuple[str, re.Pattern, type]] = [
    ("tokens/s/gpu",         re.compile(rf"^\s*Token Throughput per GPU \(tok/s/gpu\)\s*:\s*{NUM}"), float),
    ("output_tokens/s/gpu",  re.compile(rf"^\s*Output Token Throughput per GPU\s*:\s*{NUM}"),       float),
    ("Interactivity",        re.compile(rf"^\s*Interactivity \(tok/s/user\)\s*:\s*{NUM}"),          float),
    ("TTFT (ms)",            re.compile(rf"^\s*TTFT \(ms\)\s*:\s*{NUM}"),                            float),
    ("TPOT (ms)",            re.compile(rf"^\s*TPOT \(ms\)\s*:\s*{NUM}"),                            float),
    ("tokens/s",             re.compile(rf"^\s*Total Token throughput \(tok/s\)\s*:\s*{NUM}"),       float),
]

# Final column ordering (filename-derived + metrics).
COLUMNS: List[str] = [
    "input_len",
    "output_len",
    "concurrency",
    "tp",
    "ep_enabled",
    "tokens/s/gpu",
    "output_tokens/s/gpu",
    "Interactivity",
    "TTFT (ms)",
    "TPOT (ms)",
    "tokens/s",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> Optional[Dict[str, int]]:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return {
        "input_len":   int(m.group("isl")),
        "output_len":  int(m.group("osl")),
        "tp":          int(m.group("tp")),
        "ep_enabled":  int(m.group("ep")),
        "concurrency": int(m.group("conc")),
    }


def parse_log(path: Path) -> Tuple[Dict[str, float], List[str]]:
    """Scan a log file for metrics.

    Returns (metrics, missing_column_names).
    For each metric we keep the LAST occurrence -- the Key Metrics block is
    printed at the end of the run, and the benchmark_serving summary appears
    only once, so last-wins is both safe and correct here.
    """
    metrics: Dict[str, float] = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            for col, pat, cast in METRIC_SPECS:
                m = pat.match(line)
                if m:
                    try:
                        metrics[col] = cast(m.group(1))
                    except ValueError:
                        pass
    missing = [col for col, _, _ in METRIC_SPECS if col not in metrics]
    return metrics, missing


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def format_value(col: str, value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        # Pick precision per column so the table stays compact but readable.
        if col in ("TTFT (ms)", "TPOT (ms)"):
            return f"{value:.2f}"
        if col == "Interactivity":
            return f"{value:.3f}"
        return f"{value:.2f}"
    return str(value)


def print_table(rows: List[Dict[str, object]], columns: List[str]) -> None:
    str_rows = [[format_value(c, r.get(c)) for c in columns] for r in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    sep = "  "
    header = sep.join(c.ljust(widths[i]) for i, c in enumerate(columns))
    divider = sep.join("-" * widths[i] for i in range(len(columns)))
    print(header)
    print(divider)
    for row in str_rows:
        print(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def write_csv(csv_path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="./sweep-logs",
                    help="Directory containing sweep client logs (default: ./sweep-logs)")
    ap.add_argument("--csv", default=None,
                    help="CSV output path (default: <log-dir>/summary.csv). Pass empty string to disable.")
    ap.add_argument("--pattern", default="isl*_osl*_tp*_ep*_conc*.log",
                    help="Glob pattern for client log filenames (default: %(default)s)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.is_dir():
        print(f"[summarize] ERROR: log dir not found: {log_dir}", file=sys.stderr)
        return 1

    files = sorted(log_dir.glob(args.pattern))
    if not files:
        print(f"[summarize] no log files matching '{args.pattern}' in {log_dir}", file=sys.stderr)
        return 1

    rows: List[Dict[str, object]] = []
    skipped = 0

    for path in files:
        # Skip server-side archives produced by the sweep (e.g. server_isl...log).
        if path.name.startswith("server_"):
            continue

        meta = parse_filename(path.name)
        if meta is None:
            print(f"[summarize] SKIP (unrecognized filename): {path.name}")
            skipped += 1
            continue

        metrics, missing = parse_log(path)
        if missing:
            print(f"[summarize] SKIP (missing {missing}): {path.name}")
            skipped += 1
            continue

        row: Dict[str, object] = {}
        row.update(meta)
        row.update(metrics)
        rows.append(row)

    if not rows:
        print(f"[summarize] no usable logs parsed ({skipped} skipped).", file=sys.stderr)
        return 1

    # Sort for a stable, human-friendly ordering.
    rows.sort(key=lambda r: (
        r["input_len"], r["output_len"], r["tp"], r["ep_enabled"], r["concurrency"],
    ))

    print_table(rows, COLUMNS)
    print()
    print(f"[summarize] parsed {len(rows)} log(s), skipped {skipped}.")

    if args.csv != "":
        csv_path = Path(args.csv) if args.csv else (log_dir / "summary.csv")
        write_csv(csv_path, rows, COLUMNS)
        print(f"[summarize] wrote CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
