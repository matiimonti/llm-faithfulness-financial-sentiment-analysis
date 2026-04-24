"""BERTScore computation for prompt stability results (H3).

This is run separately from the stability experiment to avoid loading the
bert-score model during inference, keeping GPU memory available for the LLMs.

For each observation in stability.jsonl, it computes BERTScore F1 between
each pair of reasoning texts produced by the three prompt paraphrases:
  - v1 vs v2, v1 vs v3, v2 vs v3

Aggregates to mean F1 per model and per paraphrase pair.

Usage:
    python metrics/similarity.py
    python metrics/similarity.py --results-dir /content/drive/MyDrive/thesis_results
    python metrics/similarity.py --lang en  # default; change for multilingual
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR


def _load_stability_records(rdir: Path) -> list[dict]:
    """Load stability results from per-model files or legacy single file."""
    files = sorted(rdir.glob("stability_*.jsonl"))
    if files:
        records = []
        for f in files:
            with open(f) as fh:
                records.extend(json.loads(l) for l in fh if l.strip())
        print(f"Loaded {len(records)} rows from {[f.name for f in files]}")
        return records
    legacy = rdir / "stability.jsonl"
    if legacy.exists():
        with open(legacy) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        print(f"Loaded {len(records)} rows from legacy stability.jsonl")
        return records
    print("No stability files found.")
    return []


def compute_bertscore(
    results_dir: str | None = None,
    lang: str = "en",
) -> dict[str, dict]:
    """Compute BERTScore F1 between reasoning texts for each model.

    Returns a dict: model -> {bertscore_f1_v1_v2, v1_v3, v2_v3, mean}.
    """
    try:
        from bert_score import score as _bert_score
    except ImportError:
        raise ImportError(
            "bert-score is not installed. Run:  pip install bert-score"
        )

    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    records = _load_stability_records(rdir)

    if not records:
        return {}

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    all_metrics: dict[str, dict] = {}

    for model, rows in by_model.items():
        print(f"\nComputing BERTScore for {model} ({len(rows)} observations)...")

        # Build candidate/reference lists for each pair.
        # Skip rows where either reasoning text is empty.
        pair_cands: dict[str, list[str]] = {"v1_v2": [], "v1_v3": [], "v2_v3": []}
        pair_refs: dict[str, list[str]] = {"v1_v2": [], "v1_v3": [], "v2_v3": []}

        for r in rows:
            r1 = (r.get("reasoning_v1") or "").strip()
            r2 = (r.get("reasoning_v2") or "").strip()
            r3 = (r.get("reasoning_v3") or "").strip()

            if r1 and r2:
                pair_cands["v1_v2"].append(r1)
                pair_refs["v1_v2"].append(r2)
            if r1 and r3:
                pair_cands["v1_v3"].append(r1)
                pair_refs["v1_v3"].append(r3)
            if r2 and r3:
                pair_cands["v2_v3"].append(r2)
                pair_refs["v2_v3"].append(r3)

        model_metrics: dict[str, float | None] = {}
        valid_f1s: list[float] = []

        for pair_name in ("v1_v2", "v1_v3", "v2_v3"):
            cands = pair_cands[pair_name]
            refs = pair_refs[pair_name]

            if not cands:
                print(f"{pair_name}: no valid pairs — skipping")
                model_metrics[f"bertscore_f1_{pair_name}"] = None
                model_metrics[f"n_{pair_name}"] = 0
                continue

            # BERTScore returns (P, R, F1) tensors of shape (n,)
            _, _, F1 = _bert_score(cands, refs, lang=lang, verbose=False)
            mean_f1 = F1.mean().item()
            model_metrics[f"bertscore_f1_{pair_name}"] = mean_f1
            model_metrics[f"n_{pair_name}"] = len(cands)
            valid_f1s.append(mean_f1)
            print(f"  {pair_name}: BERTScore F1 = {mean_f1:.4f}  (n={len(cands)})")

        # Mean across all three pairs (unweighted — pairs have similar n)
        model_metrics["bertscore_f1_mean"] = (
            sum(valid_f1s) / len(valid_f1s) if valid_f1s else None
        )

        if model_metrics["bertscore_f1_mean"] is not None:
            print(f"mean BERTScore F1: {model_metrics['bertscore_f1_mean']:.4f}")

        all_metrics[model] = model_metrics

    return all_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERTScore for prompt stability results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing stability.jsonl (default: config.RESULTS_DIR)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for BERTScore (default: en)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compute_bertscore(results_dir=args.results_dir, lang=args.lang)
