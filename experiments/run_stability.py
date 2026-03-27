"""Prompt stability experiment (H3).

Runs PromptStabilityTask over all faithfulness models on Financial PhraseBank
and saves per-observation results to results/stability.jsonl.

Each row contains predictions and reasoning texts for 3 prompt paraphrases.
BERTScore over reasoning texts is computed separately in metrics/similarity.py.

Usage:
    python experiments/run_stability.py
    python experiments/run_stability.py --models llama gemma
    python experiments/run_stability.py --sample 100
"""

import argparse
import gc
import json
import logging
import sys
import traceback
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALL_FAITHFULNESS_MODELS, RESULT_FILES, SAVE_EVERY
from data.financial_phrasebank import load_dataset
from models import ALL_MODELS
from tasks.prompt_stability import PromptStabilityTask


def run(model_keys: list[str], sample_size: int | None) -> None:
    observations = load_dataset(sample_size)
    output_file = RESULT_FILES["stability"]

    with open(output_file, "w") as f:
        for model_key in model_keys:
            if model_key not in ALL_MODELS:
                print(f"Unknown model '{model_key}', skipping.")
                continue

            print(f"\n── Running prompt stability: {model_key} ──")
            model = ALL_MODELS[model_key]()
            task = PromptStabilityTask(model)

            all_agree_count = 0
            scored = 0
            errors = 0

            for i, obs in enumerate(observations):
                try:
                    result = task.run(obs)
                    f.write(json.dumps(result.to_dict()) + "\n")

                    if result.all_agree is not None:
                        scored += 1
                        if result.all_agree:
                            all_agree_count += 1

                    if (i + 1) % SAVE_EVERY == 0:
                        f.flush()
                        agree_rate = all_agree_count / scored if scored > 0 else 0.0
                        print(f"  [{i + 1}/{len(observations)}] all_agree rate so far: {agree_rate:.3f}")

                except Exception as e:
                    errors += 1
                    print(f"  Error on obs {obs.id}: {e}")
                    traceback.print_exc()

            agree_rate = all_agree_count / scored if scored > 0 else 0.0
            print(f"  Final ({model_key}): all_agree rate: {agree_rate:.3f}  ({errors} errors)")

            del model
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nResults saved to {output_file}")
    print("Note: run metrics/similarity.py to compute BERTScore over reasoning texts.")


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
