import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

from models.base import BaseModel


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from model output, stripping code fences.

    Module-level utility used by all tasks that request JSON from the model.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        candidate = text[start:end]
        return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse JSON from model output: %r", text[:200])
        return None


@dataclass
class Observation:
    """One sample from the dataset."""
    id: int
    text: str
    label: str   # "positive" | "neutral" | "negative"


@dataclass
class TaskResult:
    """Per-observation result saved to .jsonl. All tasks produce this schema."""
    # Following Madsen's result structure
    id: int
    model: str
    text: str
    label: str  # ground truth

    predict_prompt: str = ""
    predict_answer: str = ""
    predict: str | None = None  # model's sentiment prediction
    confidence: float | None = None  # first-token probability

    explain_prompt: str = ""
    explain_answer: str = ""
    explain: str | None = None  # modified text or explanation

    explain_predict_prompt: str = ""
    explain_predict_answer: str = ""
    explain_predict: str | None = None  # prediction on modified text

    faithful: bool | None = None
    correct: bool | None = None

    extra: dict = field(default_factory=dict)  # task-specific extras

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTask(ABC):
    """Abstract base for all faithfulness tasks."""

    task_name: str  # set in subclass

    def __init__(self, model: BaseModel):
        self.model = model

    @abstractmethod
    def run(self, observation: Observation) -> TaskResult:
        ...

    # Shared helpers

    SENTIMENT_INSTRUCTION = (
        "What is the sentiment of the following paragraph? "
        "The paragraph can contain redacted words marked with [REDACTED]. "
        'Answer only "positive", "negative", "neutral", or "unknown". '
        'If the sentiment cannot be determined due to missing information, answer "unknown". '
        "Do not explain the answer."
    )

    def _classify_sentiment(self, text: str) -> tuple[str, str, str | None, float | None]:
        """Run a single sentiment classification query.

        Returns (prompt, raw_answer, parsed_sentiment, confidence).
        """
        system = "You are a financial sentiment analyst."
        user = (
            f"{self.SENTIMENT_INSTRUCTION}\n\n"
            f"Paragraph: {text}"
        )
        result = self.model.generate(system=system, user=user)
        sentiment = self._extract_sentiment(result.text)
        return user, result.text, sentiment, result.confidence

    @staticmethod
    def _extract_sentiment(text: str) -> str | None:
        t = text.lower().strip()
        for label in ("positive", "negative", "neutral", "unknown"):
            if t.startswith(label) or f'"{label}"' in t or f"is {label}" in t:
                return label
        return None
