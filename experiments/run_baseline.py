"""Baseline sentiment classification experiment.

Runs ClassifyTask over all faithfulness models on Financial PhraseBank
and saves per-observation results to results/baseline.jsonl.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --models llama gemma
    python experiments/run_baseline.py --sample 100
"""

import argparse
import gc
import json
import logging
import sys
import traceback
from pathlib import Path

import torch

# logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALL_FAITHFULNESS_MODELS, RESULT_FILES, SAVE_EVERY
from data.financial_phrasebank import load_dataset
from models import ALL_MODELS
from tasks.classify import ClassifyTask


def _output_file(model_key: str, output_dir: str | None) -> Path:
    base = Path(output_dir) if output_dir else RESULT_FILES["baseline"].parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"baseline_{model_key}.jsonl"


def run(model_keys: list[str], sample_size: int | None, output_dir: str | None = None) -> None:  # None gives the whole ds
    observations = load_dataset(sample_size)

    for model_key in model_keys:
        if model_key not in ALL_MODELS:
            print(f"Unknown model '{model_key}', skipping.")
            continue

        output_file = _output_file(model_key, output_dir)
        print(f"\n── Running baseline: {model_key} ──")
        print(f"   Output: {output_file}")

        model = ALL_MODELS[model_key]()
        task = ClassifyTask(model)

        correct = 0
        errors = 0

        with open(output_file, "w") as f:
            for i, obs in enumerate(observations):
                try:
                    result = task.run(obs)
                    f.write(json.dumps(result.to_dict()) + "\n")

                    if result.correct:
                        correct += 1

                    if (i + 1) % SAVE_EVERY == 0:
                        f.flush()
                        processed = i + 1 - errors
                        acc = correct / processed if processed > 0 else 0.0
                        print(f"[{i + 1}/{len(observations)}] accuracy so far: {acc:.3f}")

                except Exception as e:
                    errors += 1
                    print(f"Error on obs {obs.id}: {e}")
                    traceback.print_exc()

        total = len(observations) - errors
        acc = correct / total if total > 0 else 0
        print(f"Final accuracy ({model_key}): {acc:.3f}  ({errors} errors)")
        print(f"Saved → {output_file}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_FAITHFULNESS_MODELS.keys()),
        help="Which models to run (default: all)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of observations to sample (default: full dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: config.RESULTS_DIR)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(model_keys=args.models, sample_size=args.sample, output_dir=args.output_dir)
