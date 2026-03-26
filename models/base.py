from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclass
class GenerationResult:
    text: str  # decoded output text
    confidence: float | None  # probability of the first predicted token (if available)


class BaseModel(ABC):
    """Abstract wrapper around a HuggingFace causal LM.

    Subclasses only need to implement `_build_messages` to handle
    model-specific chat templates (e.g. Gemma merges system+user).
    """

    model_name: str  # set in subclass

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    @abstractmethod
    def _build_messages(self, system: str, user: str) -> list[dict]:
        """Build the message list for this model's chat template."""
        ...

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> GenerationResult:
        messages = self._build_messages(system, user)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode only the newly generated tokens
        new_tokens = output.sequences[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # First-token confidence (probability of the most likely first token)
        confidence = None
        if output.scores:
            probs = torch.softmax(output.scores[0][0], dim=-1)
            confidence = probs.max().item()

        return GenerationResult(text=text.strip(), confidence=confidence)
