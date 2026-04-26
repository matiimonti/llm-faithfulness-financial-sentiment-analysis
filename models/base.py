from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import GENERATION
# Autotokenizer imports the right tokenizer for each model automatically
# AutoModelForCausalLM loads the right causal language model 

# Sentiment labels used for confidence extraction in JSON-format calls
SENTIMENT_LABELS = ("positive", "negative", "neutral", "unknown")


@dataclass  # data container for whatever comes back from a model call
class GenerationResult:
    text: str  # decoded output text
    confidence: float | None  # probability of the sentiment-relevant token


class BaseModel(ABC):
    """Abstract wrapper around a HuggingFace causal LM.

    Subclasses only need to implement `_build_messages` to handle
    model-specific chat templates (e.g. Gemma merges system+user).
    """

    model_name: str  # set in subclass
    attribution_format: str = "json"  # override to "csv" for models that can't produce JSON

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # if tokenizer has no pad token (LLaMA)
        self.model = AutoModelForCausalLM.from_pretrained(  # load model weights
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()  # puts model in inference mode - disables dropout

    @abstractmethod
    def _build_messages(self, system: str, user: str) -> list[dict]:
        """Build the message list for this model's chat template."""
        ...

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        json_output: bool = False,
    ) -> GenerationResult:
        """Generate a response.

        Args:
            max_new_tokens: Overrides GENERATION["max_new_tokens"] if provided.
            temperature:    Overrides GENERATION["temperature"] if provided.
                            Ignored when do_sample=False.
            do_sample:      Overrides GENERATION["do_sample"] if provided.
                            Set False for deterministic (greedy) decoding.
            json_output:    Set True when the model is asked to output JSON.
                            Confidence will be extracted from the sentiment label
                            token inside the output rather than the first token.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else GENERATION["max_new_tokens"]
        do_sample = do_sample if do_sample is not None else GENERATION["do_sample"]
        temperature = temperature if temperature is not None else GENERATION["temperature"]

        messages = self._build_messages(system, user)
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # apply_chat_template returns a tensor or BatchEncoding depending on transformers version
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(self.device)
        else:
            input_ids = tokenized.to(self.device)

        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)

        # Decode only the newly generated tokens
        new_token_ids = output.sequences[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        confidence = None
        if output.scores:
            if not json_output:
                # Direct label call: confidence = probability of the actually
                # generated first token (e.g. "positive", "negative", ...)
                first_token_id = new_token_ids[0].item()
                probs = torch.softmax(output.scores[0][0], dim=-1)
                confidence = probs[first_token_id].item()
            else:
                # JSON call: scan the output for the first sentiment label token
                # and return its probability at that position in the sequence.
                confidence = _extract_label_confidence(
                    new_token_ids, output.scores, SENTIMENT_LABELS, self.tokenizer
                )

        return GenerationResult(text=text.strip(), confidence=confidence)


def _extract_label_confidence(
    token_ids: torch.Tensor,
    scores: tuple,
    labels: tuple[str, ...],
    tokenizer,
) -> float | None:
    """Find the first sentiment label token in the output and return its probability.

    For JSON-format outputs, the model generates e.g. {"sentiment": "positive"}.
    We scan the token sequence for the first token that corresponds to one of
    the sentiment labels and return its probability at that position.
    """
    label_token_ids = {}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if ids:
            label_token_ids[ids[0]] = label  # first token of the label word

    for pos, token_id in enumerate(token_ids):
        tid = token_id.item()
        if tid in label_token_ids:
            probs = torch.softmax(scores[pos][0], dim=-1)
            return probs[tid].item()

    return None
