#!/usr/bin/env python3
"""Reaction-coverage benchmark runner for the MENDEL rule pipeline.

Reads ``data/reaction_coverage_benchmark.json`` and, for each reaction, runs the
full rule pipeline (parser -> identifier -> descriptor -> predictor -> negotiator)
and decides pass/fail purely from machine-readable expected fields. A single
reaction failing never aborts the run.

Pass/fail logic per reaction:
  * expected_failure_mode set  -> bucket "expected_failure" (recorded, never a CI failure)
  * parse error (no expected_failure_mode) -> "parse_failure" (CI failure)
  * otherwise all of the following must hold to "pass":
      - expected_mechanism_hint (if not null): mechanism_hint == expected
      - allowed_mechanism_hints (if not null): mechanism_hint in the set
      - required_group_types (if not null): subset of detected group types
      - required_reaction_center_group_types (if not null): subset of reaction-center
        group types
      - max_reaction_center_groups (if not null): n center groups <= bound

Usage:
    python scripts/reaction_coverage_benchmark.py
    python scripts/reaction_coverage_benchmark.py --output reports/reaction_coverage_benchmark.json
    python scripts/reaction_coverage_benchmark.py --csv reports/reaction_coverage_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mendel.identifier import identify_functional_groups  # noqa: E402
from mendel.negotiation import run_full_rule_pipeline  # noqa: E402
from mendel.parser import parse_reaction_smiles  # noqa: E402

DEFAULT_BENCHMARK_PATH = _REPO_ROOT / "data" / "reaction_coverage_benchmark.json"

# Status buckets
PASS = "pass"
FAIL = "fail"
EXPECTED_FAILURE = "expected_failure"
PARSE_FAILURE = "parse_failure"


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> list[dict[str, Any]]:
    """Load the benchmark reaction list."""
    raw = json.loads(Path(path).read_text())
    reactions = raw.get("reactions", raw) if isinstance(raw, dict) else raw
    return list(reactions)


def _detected_group_types(groups: list[Any]) -> list[str]:
    return sorted({g.group_type.value for g in groups})


def _reaction_center_group_types(result: Any) -> list[str]:
    types: set[str] = set()
    for a in getattr(result, "assignments", []):
        if getattr(a, "is_reaction_center", False):
            gt = getattr(a, "group_type", None)
            if gt is not None:
                types.add(gt.value)
    return sorted(types)


def _n_reaction_center_groups(result: Any) -> int:
    return sum(
        1 for a in getattr(result, "assignments", [])
        if getattr(a, "is_reaction_center", False)
    )


def _assignment_summary(result: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in getattr(result, "assignments", []):
        final_role = getattr(a, "final_role", None)
        out.append({
            "group_id": getattr(a, "group_id", None),
            "group_type": getattr(getattr(a, "group_type", None), "value", None),
            "final_role": getattr(final_role, "value", None),
            "subrole": getattr(a, "subrole", None),
            "is_reaction_center": getattr(a, "is_reaction_center", False),
            "confidence": getattr(a, "final_confidence", None),
            "reason": getattr(a, "reason", None),
        })
    return out


def _check(case: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Return a list of failure reasons; empty means all checks pass."""
    reasons: list[str] = []

    expected_hint = case.get("expected_mechanism_hint")
    allowed = case.get("allowed_mechanism_hints")
    hint = observed["mechanism_hint"]
    if expected_hint is not None and hint != expected_hint:
        reasons.append(f"mechanism_hint {hint!r} != expected {expected_hint!r}")
    if allowed is not None and hint not in allowed:
        reasons.append(f"mechanism_hint {hint!r} not in allowed {allowed!r}")

    required = case.get("required_group_types")
    if required is not None:
        missing = sorted(set(required) - set(observed["detected_group_types"]))
        if missing:
            reasons.append(f"missing required group types: {missing}")

    required_centers = case.get("required_reaction_center_group_types")
    if required_centers is not None:
        missing_c = sorted(
            set(required_centers) - set(observed["reaction_center_group_types"])
        )
        if missing_c:
            reasons.append(f"missing required reaction-center group types: {missing_c}")

    max_centers = case.get("max_reaction_center_groups")
    if max_centers is not None and observed["n_reaction_center_groups"] > max_centers:
        reasons.append(
            f"n_reaction_center_groups {observed['n_reaction_center_groups']} "
            f"> max {max_centers}"
        )

    return reasons


def evaluate_reaction(case: dict[str, Any]) -> dict[str, Any]:
    """Run the pipeline for one benchmark case and classify the result."""
    name = case.get("name", "<unnamed>")
    smiles = case["reaction_smiles"]
    context = case.get("context", "unknown")
    expected_failure_mode = case.get("expected_failure_mode")

    result_row: dict[str, Any] = {
        "name": name,
        "reaction_smiles": smiles,
        "context": context,
        "parse_ok": False,
        "n_groups": 0,
        "detected_group_types": [],
        "mechanism_hint": None,
        "n_reaction_center_groups": 0,
        "reaction_center_group_types": [],
        "assignments": [],
        "error": None,
        "expected_failure_mode": expected_failure_mode,
        "status": None,
        "reasons": [],
    }

    try:
        parsed = parse_reaction_smiles(smiles, context=context)
        groups = identify_functional_groups(parsed)
        result = run_full_rule_pipeline(smiles, context=context)
        result_row.update({
            "parse_ok": True,
            "n_groups": len(groups),
            "detected_group_types": _detected_group_types(groups),
            "mechanism_hint": getattr(result, "mechanism_hint", None),
            "n_reaction_center_groups": _n_reaction_center_groups(result),
            "reaction_center_group_types": _reaction_center_group_types(result),
            "assignments": _assignment_summary(result),
        })
    except Exception as exc:  # never abort the whole run on one reaction
        result_row["error"] = f"{type(exc).__name__}: {exc}"

    # Classify
    if expected_failure_mode is not None:
        result_row["status"] = EXPECTED_FAILURE
        if not result_row["parse_ok"]:
            result_row["reasons"] = [f"parse/coverage failure: {result_row['error']}"]
        else:
            result_row["reasons"] = _check(case, result_row)
        return result_row

    if not result_row["parse_ok"]:
        result_row["status"] = PARSE_FAILURE
        result_row["reasons"] = [result_row["error"] or "parse failed"]
        return result_row

    reasons = _check(case, result_row)
    result_row["status"] = PASS if not reasons else FAIL
    result_row["reasons"] = reasons
    return result_row


def evaluate_benchmark(
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all cases. Returns (per-reaction results, summary)."""
    results = [evaluate_reaction(c) for c in cases]

    by_status: dict[str, int] = {}
    by_mechanism: dict[str, dict[str, int]] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        hint = r["mechanism_hint"] or "<none>"
        bucket = by_mechanism.setdefault(hint, {"total": 0, PASS: 0, FAIL: 0})
        bucket["total"] += 1
        if r["status"] in (PASS, FAIL):
            bucket[r["status"]] += 1

    summary = {
        "total": len(results),
        "passed": by_status.get(PASS, 0),
        "failed": by_status.get(FAIL, 0),
        "expected_failures": by_status.get(EXPECTED_FAILURE, 0),
        "parse_failures": by_status.get(PARSE_FAILURE, 0),
        "by_mechanism": by_mechanism,
        # CI is red only on hard failures.
        "ci_ok": by_status.get(FAIL, 0) == 0 and by_status.get(PARSE_FAILURE, 0) == 0,
    }
    return results, summary


def _print_summary(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    icon = {PASS: "PASS", FAIL: "FAIL", EXPECTED_FAILURE: "xfail", PARSE_FAILURE: "PERR"}
    print("\nReaction coverage benchmark")
    print("=" * 92)
    print(f"{'status':6s} {'name':46s} {'hint':22s} {'#ctr':>4s}")
    print("-" * 92)
    for r in results:
        print(
            f"{icon.get(r['status'], r['status']):6s} "
            f"{r['name'][:46]:46s} "
            f"{str(r['mechanism_hint'])[:22]:22s} "
            f"{r['n_reaction_center_groups']:>4d}"
        )
        if r["status"] in (FAIL, PARSE_FAILURE) and r["reasons"]:
            for reason in r["reasons"]:
                print(f"       └─ {reason}")
    print("-" * 92)
    print(
        f"total={summary['total']}  passed={summary['passed']}  "
        f"failed={summary['failed']}  expected_failures={summary['expected_failures']}  "
        f"parse_failures={summary['parse_failures']}"
    )
    print("\nBy mechanism category:")
    for hint, b in sorted(summary["by_mechanism"].items()):
        print(f"  {hint:24s} total={b['total']:2d} pass={b[PASS]:2d} fail={b[FAIL]:2d}")
    print("=" * 92)
    print("CI:", "OK" if summary["ci_ok"] else "FAILED (hard failures present)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to the benchmark JSON.",
    )
    parser.add_argument("--output", default=None, help="Write full JSON results here.")
    parser.add_argument("--csv", default=None, help="Write a CSV summary here.")
    args = parser.parse_args(argv)

    cases = load_benchmark(args.benchmark)
    results, summary = evaluate_benchmark(cases)
    _print_summary(results, summary)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2) + "\n"
        )
        print(f"\nwrote JSON: {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "name", "status", "context", "parse_ok", "mechanism_hint",
            "n_groups", "n_reaction_center_groups", "detected_group_types",
            "reaction_center_group_types", "expected_failure_mode", "error", "reasons",
        ]
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = dict(r)
                row["detected_group_types"] = "|".join(r["detected_group_types"])
                row["reaction_center_group_types"] = "|".join(
                    r["reaction_center_group_types"]
                )
                row["reasons"] = "; ".join(r["reasons"])
                writer.writerow(row)
        print(f"wrote CSV: {csv_path}")

    return 0 if summary["ci_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
