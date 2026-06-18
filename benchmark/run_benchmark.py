"""Benchmark harness — compares system scores across human vs LLM text samples.

Usage:
    python benchmark/run_benchmark.py [--url http://localhost:8000]

The script discovers all .txt files under benchmark/human/ and benchmark/llm/,
sends each to the /api/v1/analyze endpoint, and prints a comparison report with:
  - Per-group statistics (mean, median, std, min, max)
  - Per-component discrimination analysis (gap between human and LLM means)
  - Individual file results sorted by overall score

Each benchmark file must begin with a metadata header:
    source: human|llm
    lang: en|tr
    ---
    <text body>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
HUMAN_DIR = BASE_DIR / "human"
LLM_DIR = BASE_DIR / "llm"

_TIMEOUT = 30
_COMPONENT_KEYS = [
    "repetition",
    "transition_overuse",
    "low_burstiness",
    "lexical_poverty",
    "cliche_density",
    "readability",
]


def _parse_file(path: Path) -> tuple[dict, str]:
    """Parse metadata header and text body from a benchmark file.

    Args:
        path: Path to a .txt benchmark file.

    Returns:
        Tuple of (metadata_dict, text_body).

    Raises:
        ValueError: If the file has no '---' separator.
    """
    raw = path.read_text(encoding="utf-8")
    if "---" not in raw:
        raise ValueError(f"{path}: missing '---' separator")
    header_part, body = raw.split("---", 1)
    meta: dict = {}
    for line in header_part.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta, body.strip()


def _analyze(text: str, lang: str | None, base_url: str) -> dict:
    """Call the /api/v1/analyze endpoint and return the JSON response.

    Args:
        text: Text to analyze.
        lang: Language code ("en" or "tr") or None for auto-detect.
        base_url: Base URL of the API server.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    payload: dict = {"text": text}
    if lang in ("en", "tr"):
        payload["language"] = lang
    resp = requests.post(f"{base_url}/api/v1/analyze", json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _load_group(directory: Path, base_url: str) -> list[dict]:
    """Load and analyze all .txt files in a directory.

    Args:
        directory: Directory containing benchmark .txt files.
        base_url: API base URL.

    Returns:
        List of result dicts with keys: file, source, lang, overall,
        components, contributions.
    """
    results = []
    for path in sorted(directory.glob("*.txt")):
        try:
            meta, text = _parse_file(path)
            response = _analyze(text, meta.get("lang"), base_url)
            ar = response["academic_risk"]
            results.append(
                {
                    "file": path.name,
                    "source": meta.get("source", "?"),
                    "lang": meta.get("lang", "?"),
                    "model": meta.get("model", "—"),
                    "overall": ar["overall_score"],
                    "components": ar["component_scores"],
                    "contributions": ar.get("contribution_scores", {}),
                    "risk_level": ar["risk_level"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {path.name}: {exc}", file=sys.stderr)
    return results


def _stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of floats.

    Args:
        values: Non-empty list of floats.

    Returns:
        Dict with mean, median, stdev, min, max.
    """
    if not values:
        return {"mean": 0, "median": 0, "stdev": 0, "min": 0, "max": 0}
    return {
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "stdev": round(statistics.pstdev(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def _print_group_stats(label: str, results: list[dict]) -> None:
    """Print summary statistics for one group (human or llm).

    Args:
        label: Group label string.
        results: List of result dicts from _load_group.
    """
    scores = [r["overall"] for r in results]
    s = _stats(scores)
    print(f"\n{'─'*50}")
    print(f"  {label}  (n={len(results)})")
    print(f"{'─'*50}")
    print(f"  Mean   : {s['mean']:>6.1f}")
    print(f"  Median : {s['median']:>6.1f}")
    print(f"  Std    : {s['stdev']:>6.1f}")
    print(f"  Min    : {s['min']:>6.1f}")
    print(f"  Max    : {s['max']:>6.1f}")
    print()
    print(f"  {'File':<35} {'Score':>6}  {'Risk':<12}")
    print(f"  {'─'*35} {'─'*6}  {'─'*12}")
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(f"  {r['file']:<35} {r['overall']:>6.1f}  {r['risk_level']}")


def _print_component_discrimination(human: list[dict], llm: list[dict]) -> None:
    """Print per-component discrimination analysis.

    Computes mean score per component for each group and reports the gap.
    Components with gap < 5 are flagged as weak discriminators.

    Args:
        human: Human group results.
        llm: LLM group results.
    """
    print(f"\n{'═'*60}")
    print("  COMPONENT DISCRIMINATION ANALYSIS")
    print(f"{'═'*60}")
    print(f"  {'Component':<22} {'Human':>8} {'LLM':>8} {'Gap':>8}  Verdict")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")

    for key in _COMPONENT_KEYS:
        h_vals = [r["components"].get(key, 0) for r in human]
        l_vals = [r["components"].get(key, 0) for r in llm]
        h_mean = statistics.mean(h_vals) if h_vals else 0
        l_mean = statistics.mean(l_vals) if l_vals else 0
        gap = l_mean - h_mean
        if abs(gap) < 5:
            verdict = "weak discriminator"
        else:
            verdict = "LLM > human" if gap > 0 else "human > LLM"
        print(f"  {key:<22} {h_mean:>8.1f} {l_mean:>8.1f} {gap:>+8.1f}  {verdict}")


def _print_contribution_breakdown(label: str, results: list[dict]) -> None:
    """Print mean weighted contributions for a group.

    Args:
        label: Group label string.
        results: List of result dicts.
    """
    if not results or not results[0]["contributions"]:
        return
    print(f"\n  Contribution breakdown — {label}:")
    print(f"  {'Component':<22} {'Mean contribution':>18}")
    print(f"  {'─'*22} {'─'*18}")
    for key in _COMPONENT_KEYS:
        vals = [r["contributions"].get(key, 0) for r in results]
        mean_val = statistics.mean(vals) if vals else 0
        print(f"  {key:<22} {mean_val:>18.2f}")
    totals = [r["overall"] for r in results]
    print(f"  {'─'*22} {'─'*18}")
    print(f"  {'TOTAL':<22} {statistics.mean(totals):>18.2f}")


def main() -> None:
    """Entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Academic Writing Auditor — Benchmark Harness"
    )
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  ACADEMIC WRITING AUDITOR — BENCHMARK RUN")
    print(f"  API: {args.url}")
    print(f"{'═'*60}")

    print("\nLoading human samples...")
    human = _load_group(HUMAN_DIR, args.url)

    print("Loading LLM samples...")
    llm = _load_group(LLM_DIR, args.url)

    if not human and not llm:
        print("No samples found. Add .txt files to benchmark/human/ and benchmark/llm/")
        sys.exit(1)

    _print_group_stats("HUMAN", human)
    _print_group_stats("LLM", llm)

    if human and llm:
        _print_component_discrimination(human, llm)

    if human:
        _print_contribution_breakdown("HUMAN", human)
    if llm:
        _print_contribution_breakdown("LLM", llm)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
