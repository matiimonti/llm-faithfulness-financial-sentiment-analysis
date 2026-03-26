"""Baseline sentiment classification experiment.

Runs ClassifyTask over all faithfulness models on Financial PhraseBank
and saves per-observation results to results/baseline.jsonl.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --models llama gemma
    python experiments/run_baseline.py --sample 100
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALL_FAITHFULNESS_MODELS, RESULT_FILES, SAVE_EVERY
from data.financial_phrasebank import load_dataset
from models import ALL_MODELS
from tasks.classify import ClassifyTask


def run(model_keys: list[str], sample_size: int | None) -> None:
    observations = load_dataset(sample_size)
    output_file = RESULT_FILES["baseline"]

    with open(output_file, "w") as f:
        for model_key in model_keys:
            if model_key not in ALL_MODELS:
                print(f"Unknown model '{model_key}', skipping.")
                continue

            print(f"\n── Running baseline: {model_key} ──")
            model = ALL_MODELS[model_key]()
            task = ClassifyTask(model)

            correct = 0
            errors = 0

            for i, obs in enumerate(observations):
                try:
                    result = task.run(obs)
                    f.write(json.dumps(result.to_dict()) + "\n")

                    if result.correct:
                        correct += 1

                    if (i + 1) % SAVE_EVERY == 0:
                        f.flush()
                        acc = correct / (i + 1 - errors)
                        print(f"  [{i + 1}/{len(observations)}] accuracy so far: {acc:.3f}")

                except Exception as e:
                    errors += 1
                    print(f"  Error on obs {obs.id}: {e}")
                    traceback.print_exc()

            total = len(observations) - errors
            acc = correct / total if total > 0 else 0
            print(f"  Final accuracy ({model_key}): {acc:.3f}  ({errors} errors)")

            # Free GPU memory before loading next model
            del model
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nResults saved to {output_file}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(model_keys=args.models, sample_size=args.sample)
