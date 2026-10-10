"""Compare regenerated outputs with their committed versions (reproduction check of code v0.1).

    python tools/compare_to_git.py [--rev HEAD] [--rtol 1e-9] [paths ...]      (default: every modified file in data/)

CSV files are compared cell by cell: numbers with a relative tolerance (and an absolute one for values that are zero
up to rounding), text exactly; YAML and JSON files leaf by leaf,
ignoring bookkeeping keys (date, created, git_commit). Binary files (b8, png, pdf) are reported as identical or not.
Prints one line per file: IDENTICAL, EQUAL (within rtol), or DIFFERENT with the largest relative deviation and where.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
IGNORE = {"date", "created", "git_commit"}
ATOL = 1e-12  # numbers closer than this are equal (values that are zero up to rounding); set by --atol


def git_show(rev, rel):
    return subprocess.run(["git", "show", f"{rev}:./{rel}"], cwd=ROOT, capture_output=True, check=True).stdout


def modified(prefix="data"):
    out = subprocess.run(["git", "status", "--porcelain", "--", prefix], cwd=ROOT, capture_output=True, text=True).stdout
    return [line[3:] for line in out.splitlines() if line[:2].strip() == "M"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _rel(a, b):
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if a == b or abs(a - b) <= ATOL:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare_csv(old: bytes, new: bytes):
    def rows(b):
        text = b.decode("utf-8")
        return list(csv.reader(l for l in io.StringIO(text) if not l.startswith("#")))
    ro, rn = rows(old), rows(new)
    if len(ro) != len(rn):
        return math.inf, f"{len(ro)} vs {len(rn)} rows"
    worst, where = 0.0, ""
    head = ro[0] if ro else []
    for i, (a, b) in enumerate(zip(ro, rn)):
        if len(a) != len(b):
            return math.inf, f"row {i}: {len(a)} vs {len(b)} columns"
        for j, (x, y) in enumerate(zip(a, b)):
            nx, ny = _num(x), _num(y)
            if nx is None or ny is None:
                if x != y:
                    return math.inf, f"row {i}, column {head[j] if j < len(head) else j}: {x!r} vs {y!r}"
                continue
            r = _rel(nx, ny)
            if r > worst:
                worst, where = r, f"row {i}, column {head[j] if j < len(head) else j}: {x} vs {y}"
    return worst, where


def _leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k) not in IGNORE:
                yield from _leaves(v, path + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, path + (str(i),))
    else:
        yield path, x


def compare_tree(old, new):
    """Largest relative deviation over the common leaves; keys present on one side only are reported (and make the
    files DIFFERENT only through the returned note, flagged with a deviation of inf if a common leaf differs in text)."""
    lo, ln = dict(_leaves(old)), dict(_leaves(new))
    added, removed = sorted(set(ln) - set(lo)), sorted(set(lo) - set(ln))
    note = ""
    if added or removed:
        note = (f"keys added {['.'.join(k) for k in added][:4]}" if added else "") +                (f" keys removed {['.'.join(k) for k in removed][:4]}" if removed else "")
    worst, where = 0.0, note
    for k in set(lo) & set(ln):
        a, b = lo[k], ln[k]
        if isinstance(a, bool) or isinstance(b, bool) or _num(a) is None or _num(b) is None:
            if a != b:
                return math.inf, f"{'.'.join(k)}: {a!r} vs {b!r}"
            continue
        r = _rel(float(a), float(b))
        if r > worst:
            worst, where = r, (note + "; " if note else "") + f"{'.'.join(k)}: {a} vs {b}"
    return worst, where


def compare(rel, rev):
    new = (ROOT / rel).read_bytes()
    old = git_show(rev, rel)
    if old == new:
        return 0.0, "", True
    suffix = pathlib.PurePath(rel).suffix
    if suffix == ".csv":
        return (*compare_csv(old, new), False)
    if suffix in (".yaml", ".yml"):
        return (*compare_tree(yaml.safe_load(old), yaml.safe_load(new)), False)
    if suffix == ".json":
        return (*compare_tree(json.loads(old), json.loads(new)), False)
    return math.inf, "binary content differs", False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12, help="absolute tolerance (numbers that are zero up to rounding)")
    a = ap.parse_args()
    global ATOL
    ATOL = a.atol
    paths = a.paths or modified()
    bad = 0
    for rel in paths:
        worst, where, identical = compare(rel, a.rev)
        if identical:
            print(f"IDENTICAL  {rel}")
        elif worst <= a.rtol:
            extra = f"; {where}" if where.startswith("keys") else ""
            print(f"EQUAL      {rel} (max rel. deviation {worst:.1e}; bookkeeping fields ignored{extra})")
        else:
            bad += 1
            print(f"DIFFERENT  {rel}: max rel. deviation {worst:.3g} at {where}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
