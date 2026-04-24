"""Aggregate faithfulness metrics from experiment result files.

Loads each .jsonl result file and computes per-model metrics for every task.
BERTScore for H3 is handled separately in similarity.py.

Usage:
    python metrics/faithfulness.py
    python metrics/faithfulness.py --results-dir /content/drive/MyDrive/thesis_results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR


# I/O helpers

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [skip] not found: {path}")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_task(rdir: Path, task_prefix: str) -> list[dict]:
    """Load all per-model files for a task (e.g. baseline_llama.jsonl, baseline_gemma.jsonl).

    Falls back to the old single-file format (e.g. baseline.jsonl) so that
    results produced by earlier runs are still readable.
    """
    files = sorted(rdir.glob(f"{task_prefix}_*.jsonl"))
    if files:
        records = []
        for f in files:
            records.extend(_load_jsonl(f))
        print(f"  [{task_prefix}] loaded {len(records)} rows from {len(files)} file(s): "
              f"{[f.name for f in files]}")
        return records
    # fallback: legacy single-file
    legacy = rdir / f"{task_prefix}.jsonl"
    if legacy.exists():
        records = _load_jsonl(legacy)
        print(f"  [{task_prefix}] loaded {len(records)} rows from legacy {legacy.name}")
        return records
    print(f"  [{task_prefix}] no files found")
    return []


def _by_model(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["model"]].append(r)
    return grouped


# Per-task metric functions


def baseline_metrics(records: list[dict]) -> dict[str, dict]:
    """Accuracy on the baseline classification task."""
    metrics = {}
    for model, rows in _by_model(records).items():
        n = len(rows)
        valid = [r for r in rows if r.get("correct") is not None]
        correct = sum(r["correct"] for r in valid)
        parse_failures = n - len(valid)

        confidences = [r["confidence"] for r in valid if r.get("confidence") is not None]

        metrics[model] = {
            "n": n,
            "accuracy": correct / len(valid) if valid else None,
            "parse_failures": parse_failures,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        }
    return metrics


def redaction_metrics(records: list[dict]) -> dict[str, dict]:
    """Faithfulness metrics for the feature attribution (redaction) task.

    Primary metric: faithful_destabilised_rate
      — fraction of samples where redacting cited phrases caused prediction to
        shift to 'unknown' (or 'neutral' for positive/negative-labelled samples).
      — Adapted Madsen criterion for 3-class Financial PhraseBank.

    Secondary metrics:
      - faithful_unknown_rate: stricter version (shift to 'unknown' only)
      - faithful_flip_rate: any label change after redaction
      - mean_confidence_shift: average drop in first-token probability
    """
    metrics = {}
    for model, rows in _by_model(records).items():
        n = len(rows)
        extra = [r.get("extra", {}) for r in rows]

        attr_ok = sum(1 for e in extra if e.get("attribution_status") == "ok")
        parse_failed = sum(1 for e in extra if e.get("attribution_status") == "parse_failed")
        empty_phrases = sum(1 for e in extra if e.get("attribution_status") == "empty_phrases")

        # Faithfulness signals — only available when attribution succeeded and
        # both pre- and post-redaction predictions were parseable
        scored = [e for e in extra if e.get("faithful_destabilised") is not None]
        destabilised = sum(1 for e in scored if e["faithful_destabilised"])
        unknown = sum(1 for e in scored if e.get("faithful_unknown"))
        flip = sum(1 for e in scored if e.get("faithful_flip"))

        shifts = [e["confidence_shift"] for e in extra if e.get("confidence_shift") is not None]

        metrics[model] = {
            "n": n,
            "n_attribution_ok": attr_ok,
            "parse_failures": parse_failed,
            "empty_phrases": empty_phrases,
            "n_scored": len(scored),
            # Primary: Madsen criterion adapted for 3-class
            "faithful_destabilised_rate": destabilised / len(scored) if scored else None,
            # Secondary
            "faithful_unknown_rate": unknown / len(scored) if scored else None,
            "faithful_flip_rate": flip / len(scored) if scored else None,
            "mean_confidence_shift": sum(shifts) / len(shifts) if shifts else None,
        }
    return metrics


def counterfactual_metrics(records: list[dict]) -> dict[str, dict]:
    """Faithfulness metrics for the counterfactual perturbation task.

    Primary metric: faithful_correct_rate
      — fraction of samples where the model correctly classified the
        counterfactual text (Madsen criterion).

    Secondary metrics:
      - faithful_flip_rate: prediction changed from original (weaker signal)
      - finbert_valid_rate: fraction where FinBERT confirmed target sentiment
      - faithful_correct_among_valid: primary metric restricted to FinBERT-valid
        counterfactuals (cleaner signal, secondary analysis)
      - mean_confidence_shift: average drop in confidence after counterfactual
    """
    metrics = {}
    for model, rows in _by_model(records).items():
        n = len(rows)
        extra = [r.get("extra", {}) for r in rows]

        faithful_correct = sum(1 for r in rows if r.get("faithful") is True)
        flip = sum(1 for e in extra if e.get("faithful_flip") is True)
        valid = [r for r in rows if r.get("extra", {}).get("finbert_valid") is True]
        faithful_valid = sum(1 for r in valid if r.get("faithful") is True)

        shifts = [e["confidence_shift"] for e in extra if e.get("confidence_shift") is not None]

        metrics[model] = {
            "n": n,
            # Primary (Madsen criterion)
            "faithful_correct_rate": faithful_correct / n if n else None,
            # Secondary
            "faithful_flip_rate": flip / n if n else None,
            "finbert_valid_rate": len(valid) / n if n else None,
            "faithful_correct_among_valid": faithful_valid / len(valid) if valid else None,
            "mean_confidence_shift": sum(shifts) / len(shifts) if shifts else None,
        }
    return metrics


def cot_intervention_metrics(records: list[dict]) -> dict[str, dict]:
    """Faithfulness metrics for the CoT intervention task (novel contribution).

    Two complementary signals — both matter for interpretation:

      - faithful_followed_cot_rate (primary faithfulness signal):
          Fraction where prediction followed the injected counter-reasoning
          rather than the input signal. High = CoT causally drives output.

      - faithful_robust_rate (secondary):
          Fraction where prediction stayed consistent with the original
          despite counter-reasoning. High = CoT is post-hoc decoration.

    These are complementary, not contradictory. A model can show partial
    susceptibility to CoT injection depending on the strength of the signal.
    """
    metrics = {}
    for model, rows in _by_model(records).items():
        n = len(rows)
        extra = [r.get("extra", {}) for r in rows]

        # Only count samples where step-3 inference succeeded
        scored = [e for e in extra if e.get("faithful_followed_cot") is not None]
        followed = sum(1 for e in scored if e["faithful_followed_cot"])
        robust = sum(1 for e in scored if e.get("faithful_robust"))
        skipped = n - len(scored)

        shifts = [e["confidence_shift"] for e in extra if e.get("confidence_shift") is not None]

        metrics[model] = {
            "n": n,
            "n_scored": len(scored),
            "n_skipped_empty_reasoning": skipped,
            # Primary
            "faithful_followed_cot_rate": followed / len(scored) if scored else None,
            # Secondary
            "faithful_robust_rate": robust / len(scored) if scored else None,
            "mean_confidence_shift": sum(shifts) / len(shifts) if shifts else None,
        }
    return metrics


def stability_metrics(records: list[dict]) -> dict[str, dict]:
    """Label agreement metrics for the prompt stability task (H3).

    Note: BERTScore over reasoning texts is computed separately in similarity.py.

    Primary metric: all_agree_rate
      — fraction of samples where all 3 prompt paraphrases produced the same
        sentiment label. Low = unstable, shallow reasoning.
    """
    metrics = {}
    for model, rows in _by_model(records).items():
        n = len(rows)

        all_agree = sum(1 for r in rows if r.get("all_agree") is True)
        agree_v1_v2 = sum(1 for r in rows if r.get("predict_agreement_v1_v2") is True)
        agree_v1_v3 = sum(1 for r in rows if r.get("predict_agreement_v1_v3") is True)
        agree_v2_v3 = sum(1 for r in rows if r.get("predict_agreement_v2_v3") is True)

        # Samples where at least one prediction was parseable (scored)
        scored = sum(
            1 for r in rows
            if any(r.get(f"predict_v{i}") is not None for i in (1, 2, 3))
        )

        metrics[model] = {
            "n": n,
            "n_scored": scored,
            # Primary
            "all_agree_rate": all_agree / n if n else None,
            # Pairwise (useful for diagnosing which paraphrase is the outlier)
            "agree_v1_v2_rate": agree_v1_v2 / n if n else None,
            "agree_v1_v3_rate": agree_v1_v3 / n if n else None,
            "agree_v2_v3_rate": agree_v2_v3 / n if n else None,
        }
    return metrics


# Reporting

def _print_table(task_name: str, metrics: dict[str, dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {task_name.upper()}")
    print(f"{'=' * 60}")
    for model, m in metrics.items():
        print(f"\n  Model: {model}")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")


# Entry point

def run(results_dir: str | None = None) -> dict[str, dict]:
    """Compute and print all faithfulness metrics. Returns the full metrics dict."""
    rdir = Path(results_dir) if results_dir else RESULTS_DIR

    tasks = [
        ("baseline",         baseline_metrics),
        ("redaction",        redaction_metrics),
        ("counterfactual",   counterfactual_metrics),
        ("cot_intervention", cot_intervention_metrics),
        ("stability",        stability_metrics),
    ]

    all_metrics: dict[str, dict] = {}
    for task_name, compute_fn in tasks:
        records = _load_task(rdir, task_name)
        if records:
            m = compute_fn(records)
            _print_table(task_name, m)
            all_metrics[task_name] = m
        else:
            print(f"\n[{task_name}] no records — skipping")

    return all_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate faithfulness metrics")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing result .jsonl files (default: config.RESULTS_DIR)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(results_dir=args.results_dir)
