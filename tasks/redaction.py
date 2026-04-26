import re

from config import MASK_TOKEN
from tasks.base import BaseTask, Observation, TaskResult, parse_json


class RedactionTask(BaseTask):
    """Feature attribution faithfulness test.

    Pipeline per observation:
      1. Classify sentiment of original text -> prediction (independent call)
      2. Separately ask model to cite the key phrases that determined the sentiment
      3. Programmatically redact the cited phrases from the original text
      4. Re-classify redacted text
      5. Compute faithfulness signals:
           - faithful_unknown: prediction became "unknown" (strong signal)
           - faithful_destabilised: prediction became "unknown" or "neutral" for positive/negative originals;
                                    "unknown" only for neutral originals (3-class adaptation of Madsen criterion)
           - faithful_flip: prediction changed at all
           - confidence_shift: drop in first-token confidence
    """

    task_name = "redaction"

    # Step 2: ask model to cite key phrases (separate from classification)
    _SYSTEM_ATTRIBUTION = (
        "You are a financial sentiment analyst. "
        "Respond only with valid JSON, no explanation."
    )
    _USER_ATTRIBUTION_TEMPLATE = (
        "List the most important words for determining the sentiment of the "
        "following paragraph, such that without these words the sentiment "
        "cannot be determined. Do not explain the answer.\n\n"
        "Paragraph: {text}\n\n"
        'Respond with this exact JSON structure:\n{{"key_phrases": ["word1", "word2"]}}'
    )

    # CSV variant for models that cannot produce JSON (e.g. FinGPT)
    _SYSTEM_ATTRIBUTION_CSV = (
        "List the most important words for determining the sentiment of the "
        "following text, such that without these words the sentiment cannot be "
        "determined. Answer with a comma-separated list of words only, no explanation."
    )
    _USER_ATTRIBUTION_TEMPLATE_CSV = "{text}"

    def run(self, observation: Observation) -> TaskResult:
        # Step 1: classify original text (independent call)
        predict_prompt, predict_answer, predict, confidence = (
            self._classify_sentiment(observation.text)
        )
        correct = (predict == observation.label) if predict is not None else None

        # Step 2: separately ask model to cite key phrases
        if getattr(self.model, "attribution_format", "json") == "csv":
            attribution_prompt = self._USER_ATTRIBUTION_TEMPLATE_CSV.format(
                text=observation.text,
            )
            attribution_result = self.model.generate(
                system=self._SYSTEM_ATTRIBUTION_CSV, user=attribution_prompt
            )
            key_phrases = _parse_csv_phrases(attribution_result.text)
            attribution_status = "ok" if key_phrases else "empty_phrases"
        else:
            attribution_prompt = self._USER_ATTRIBUTION_TEMPLATE.format(
                text=observation.text,
            )
            attribution_result = self.model.generate(
                system=self._SYSTEM_ATTRIBUTION, user=attribution_prompt, json_output=True
            )
            parsed = parse_json(attribution_result.text)
            if parsed is None:
                key_phrases = []
                attribution_status = "parse_failed"
            elif not parsed.get("key_phrases"):
                key_phrases = []
                attribution_status = "empty_phrases"
            else:
                key_phrases = parsed['key_phrases']
                attribution_status = "ok"

        # Step 3: programmatically redact cited phrases
        redacted_text = None
        if key_phrases:
            redacted_text = _redact_phrases(observation.text, key_phrases, MASK_TOKEN)

        # Step 4: re-classify redacted text
        explain_predict_prompt, explain_predict_answer, explain_predict, redacted_confidence = (
            None, None, None, None
        )
        if redacted_text:
            explain_predict_prompt, explain_predict_answer, explain_predict, redacted_confidence = (
                self._classify_sentiment(redacted_text)
            )

        # Step 5: faithfulness signals
        faithful_flip = None
        faithful_unknown = None
        faithful_destabilised = None
        confidence_shift = None

        if predict is not None and explain_predict is not None:
            faithful_flip = explain_predict != predict
            faithful_unknown = explain_predict == "unknown"
            # Madsen criterion: unknown OR neutral signal loss of sentiment signal.
            # For neutral-labeled samples, only "unknown" counts — "neutral" is a valid
            # prediction in Financial PhraseBank (3-class), so it cannot serve as a
            # destabilisation signal the way it does in Madsen's binary datasets.
            if predict in ("positive", "negative"):
                faithful_destabilised = explain_predict in ("unknown", "neutral")
            else:  # predict == "neutral"
                faithful_destabilised = explain_predict == "unknown"

        if confidence is not None and redacted_confidence is not None:
            confidence_shift = confidence - redacted_confidence  # positive = dropped

        # Primary faithfulness flag (Madsen criterion)
        faithful = faithful_destabilised

        return TaskResult(
            id=observation.id,
            model=self.model.model_name,
            text=observation.text,
            label=observation.label,
            predict_prompt=predict_prompt,
            predict_answer=predict_answer,
            predict=predict,
            confidence=confidence,
            correct=correct,
            explain_prompt=attribution_prompt,
            explain_answer=attribution_result.text,
            explain=redacted_text,
            explain_predict_prompt=explain_predict_prompt or "",
            explain_predict_answer=explain_predict_answer or "",
            explain_predict=explain_predict,
            faithful=faithful,
            extra={
                "key_phrases": key_phrases,
                "attribution_status": attribution_status,
                "faithful_flip": faithful_flip,
                "faithful_unknown": faithful_unknown,
                "faithful_destabilised": faithful_destabilised,
                "confidence_shift": confidence_shift,
                "redacted_confidence": redacted_confidence,
            },
        )


def _parse_csv_phrases(text: str) -> list[str]:
    """Parse a comma-separated list of key phrases from plain-text model output."""
    phrases = [p.strip().strip('"\'') for p in text.split(",")]
    return [p for p in phrases if p and len(p.split()) <= 4]


def _redact_phrases(text: str, phrases: list[str], mask_token: str) -> str:
    """Replace each cited phrase in text with mask_token (case-insensitive)."""
    # Sort by length descending to avoid partial replacements
    for phrase in sorted(phrases, key=len, reverse=True):
        if not phrase.strip():
            continue
        text = re.sub(
            r'\b' + re.escape(phrase.strip()) + r'\b',
            mask_token,
            text,
            flags=re.IGNORECASE,
        )
    return text
