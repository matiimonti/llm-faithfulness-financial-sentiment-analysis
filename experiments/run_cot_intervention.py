"""CoT Intervention faithfulness experiment (novel contribution).

Runs CoTInterventionTask over all faithfulness models on Financial PhraseBank
and saves per-observation results to results/cot_intervention.jsonl.

Two complementary faithfulness signals are tracked:
  - faithful_followed_cot: prediction followed the injected false reasoning
  - faithful_robust: prediction stayed consistent with the input signal

Usage:
    python experiments/run_cot_intervention.py
    python experiments/run_cot_intervention.py --models llama gemma
    python experiments/run_cot_intervention.py --sample 100
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
from tasks.cot_intervention import CoTInterventionTask


def _output_file(model_key: str, output_dir: str | None) -> Path:
    base = Path(output_dir) if output_dir else RESULT_FILES["cot_intervention"].parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"cot_intervention_{model_key}.jsonl"


def run(model_keys: list[str], sample_size: int | None, output_dir: str | None = None) -> None:
    observations = load_dataset(sample_size)

    for model_key in model_keys:
        if model_key not in ALL_MODELS:
            print(f"Unknown model '{model_key}', skipping.")
            continue

        output_file = _output_file(model_key, output_dir)
        print(f"\n── Running CoT intervention: {model_key} ──")
        print(f"   Output: {output_file}")

        model = ALL_MODELS[model_key]()
        task = CoTInterventionTask(model)

        followed_count = 0
        robust_count = 0
        scored = 0
        errors = 0

        with open(output_file, "w") as f:
            for i, obs in enumerate(observations):
                try:
                    result = task.run(obs)
                    f.write(json.dumps(result.to_dict()) + "\n")

                    if result.extra.get("faithful_followed_cot") is not None:
                        scored += 1
                        if result.extra["faithful_followed_cot"]:
                            followed_count += 1
                        if result.extra["faithful_robust"]:
                            robust_count += 1

                    if (i + 1) % SAVE_EVERY == 0:
                        f.flush()
                        followed_rate = followed_count / scored if scored > 0 else 0.0
                        robust_rate = robust_count / scored if scored > 0 else 0.0
                        print(
                            f"[{i + 1}/{len(observations)}] "
                            f"followed_cot: {followed_rate:.3f}  robust: {robust_rate:.3f}"
                        )

                except Exception as e:
                    errors += 1
                    print(f"Error on obs {obs.id}: {e}")
                    traceback.print_exc()

        followed_rate = followed_count / scored if scored > 0 else 0.0
        robust_rate = robust_count / scored if scored > 0 else 0.0
        print(
            f"Final ({model_key}): "
            f"followed_cot: {followed_rate:.3f}  robust: {robust_rate:.3f}  "
            f"({errors} errors)"
        )
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
