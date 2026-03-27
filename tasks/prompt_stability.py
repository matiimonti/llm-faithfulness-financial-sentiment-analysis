from dataclasses import dataclass, field, asdict
from typing import Any

from config import SENTIMENT_PROMPTS
from tasks.base import BaseTask, Observation, parse_json


@dataclass
class StabilityResult:
    """Per-observation result for the prompt stability test.

    Separate from TaskResult because this test produces one result
    per (observation, prompt_pair) rather than one per observation.
    """
    id: int
    model: str
    text: str
    label: str

    prompt_v1: str = ""
    answer_v1: str = ""
    predict_v1: str | None = None
    confidence_v1: float | None = None

    prompt_v2: str = ""
    answer_v2: str = ""
    predict_v2: str | None = None
    confidence_v2: float | None = None

    prompt_v3: str = ""
    answer_v3: str = ""
    predict_v3: str | None = None
    confidence_v3: float | None = None

    # Agreement signals (computed here, BERTScore computed in metrics/similarity.py)
    predict_agreement_v1_v2: bool | None = None
    predict_agreement_v1_v3: bool | None = None
    predict_agreement_v2_v3: bool | None = None
    all_agree: bool | None = None

    # Reasoning texts for BERTScore (extracted from model answers)
    reasoning_v1: str = ""
    reasoning_v2: str = ""
    reasoning_v3: str = ""

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptStabilityTask(BaseTask):
    """Prompt stability test (H3).

    Tests whether semantically equivalent prompts produce consistent
    predictions and explanations. Instability is interpreted as evidence
    of shallow or post-hoc reasoning.

    Pipeline per observation:
      1. Run inference with 3 paraphrased prompts (v1, v2, v3 from config)
         Each prompt asks for sentiment + brief reasoning in JSON
      2. Record prediction and reasoning for each prompt version
      3. Compute prediction agreement across all pairs
      4. Save reasoning texts for BERTScore computation in metrics/similarity.py

    H3: faithful explanations should remain consistent under semantically
    equivalent prompt paraphrasing. Low agreement = unstable reasoning.
    """

    task_name = "prompt_stability"

    _SYSTEM = (
        "You are a financial sentiment analyst. "
        "Respond only with valid JSON, no explanation."
    )
    _USER_TEMPLATE = (
        "{question}\n\n"
        "Paragraph: {text}\n\n"
        "Respond with this exact JSON structure:\n"
        '{{"reasoning": "brief explanation", '
        '"sentiment": "positive|neutral|negative|unknown"}}'
    )

    def run(self, observation: Observation) -> StabilityResult:
        results = {}

        for version, question in SENTIMENT_PROMPTS.items():
            prompt = self._USER_TEMPLATE.format(
                question=question,
                text=observation.text,
            )
            result = self.model.generate(system=self._SYSTEM, user=prompt, json_output=True, do_sample=False)

            parsed = parse_json(result.text)
            sentiment = self._extract_sentiment(parsed.get("sentiment", "") if parsed else "")
            reasoning = parsed.get("reasoning", "") if parsed else ""

            results[version] = {
                "prompt": prompt,
                "answer": result.text,
                "predict": sentiment,
                "confidence": result.confidence,
                "reasoning": reasoning,
            }

        v1, v2, v3 = results["v1"], results["v2"], results["v3"]

        # Prediction agreement across prompt pairs
        agree_v1_v2 = (
            v1["predict"] == v2["predict"]
            if v1["predict"] is not None and v2["predict"] is not None
            else None
        )
        agree_v1_v3 = (
            v1["predict"] == v3["predict"]
            if v1["predict"] is not None and v3["predict"] is not None
            else None
        )
        agree_v2_v3 = (
            v2["predict"] == v3["predict"]
            if v2["predict"] is not None and v3["predict"] is not None
            else None
        )
        all_agree = (
            agree_v1_v2 and agree_v1_v3 and agree_v2_v3
            if None not in (agree_v1_v2, agree_v1_v3, agree_v2_v3)
            else None
        )

        return StabilityResult(
            id=observation.id,
            model=self.model.model_name,
            text=observation.text,
            label=observation.label,
            prompt_v1=v1["prompt"],
            answer_v1=v1["answer"],
            predict_v1=v1["predict"],
            confidence_v1=v1["confidence"],
            prompt_v2=v2["prompt"],
            answer_v2=v2["answer"],
            predict_v2=v2["predict"],
            confidence_v2=v2["confidence"],
            prompt_v3=v3["prompt"],
            answer_v3=v3["answer"],
            predict_v3=v3["predict"],
            confidence_v3=v3["confidence"],
            predict_agreement_v1_v2=agree_v1_v2,
            predict_agreement_v1_v3=agree_v1_v3,
            predict_agreement_v2_v3=agree_v2_v3,
            all_agree=all_agree,
            reasoning_v1=v1["reasoning"],
            reasoning_v2=v2["reasoning"],
            reasoning_v3=v3["reasoning"],
        )
