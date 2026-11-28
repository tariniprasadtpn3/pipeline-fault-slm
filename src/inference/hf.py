"""Run a Hugging Face chat model on dataset prompts (used by the LoRA notebook).

The model and tokenizer are passed in and nothing here imports torch or
transformers, so tests use stubs on CPU.
"""

from __future__ import annotations

from typing import Any, Callable

from src.dataset.builder import SYSTEM_PROMPT


def hf_predictor(model: Any, tokenizer: Any, max_new_tokens: int = 96) -> Callable[[str], str]:
    """Adapt a chat model to the eval harness `prompt -> text` interface.

    Decoding is greedy and only the newly generated tokens are returned.
    """

    def predict(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # return_dict is explicit because its default differs between
        # transformers releases (True from v5).
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return predict
