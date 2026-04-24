import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import GENERATION
from models.base import BaseModel, GenerationResult, SENTIMENT_LABELS, _extract_label_confidence


class FinGPTModel(BaseModel):
    model_name = "FinGPT/fingpt-sentiment_llama2-7b-lora"
    _base_model_name = "meta-llama/Llama-2-7b-hf"

    def __init__(self, device: str = "cuda"):
        # FinGPT is a LoRA adapter — load base model first, then apply adapter.
        # We override __init__ entirely to skip BaseModel's from_pretrained call.
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, self.model_name)
        self.model.eval()

    def _build_messages(self, system: str, user: str) -> list[dict]:
        # Not used — FinGPT does not use a chat template.
        raise NotImplementedError

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        json_output: bool = False,
    ) -> GenerationResult:
        # FinGPT was trained with a plain instruction format, not a chat template.
        # We bypass apply_chat_template and tokenize the prompt directly.
        max_new_tokens = max_new_tokens if max_new_tokens is not None else GENERATION["max_new_tokens"]
        do_sample = do_sample if do_sample is not None else GENERATION["do_sample"]
        temperature = temperature if temperature is not None else GENERATION["temperature"]

        prompt = f"Instruction: {system}\nInput: {user}\nAnswer: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

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

        new_token_ids = output.sequences[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        confidence = None
        if output.scores:
            if not json_output:
                first_token_id = new_token_ids[0].item()
                probs = torch.softmax(output.scores[0][0], dim=-1)
                confidence = probs[first_token_id].item()
            else:
                confidence = _extract_label_confidence(
                    new_token_ids, output.scores, SENTIMENT_LABELS, self.tokenizer
                )

        return GenerationResult(text=text.strip(), confidence=confidence)
