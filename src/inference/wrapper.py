"""Inference wrapper: turn a failure log into a validated action.

The wrapper is model-agnostic. A *generator* is any callable
`(messages, prefill) -> text` (a transformers/peft pipeline in real use, a
`MockModel` in tests). The wrapper adds the parts a raw model lacks:

- the same prompt the model was trained on (`build_prompt` + system prompt)
- constrained JSON output: the assistant turn is prefilled with `{"action": "`
  so the model can only continue a JSON action object, and the first balanced
  JSON object in the reply is extracted (stray prose or fences are ignored)
- schema validation through `parse_action`; invalid output is retried with the
  validation errors fed back, up to `max_retries`, and never returned as ok
- `action_json_schema()` exports the closed vocabulary as a JSON Schema for
  grammar-guided decoders (outlines, vLLM guided decoding, llama.cpp)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.agent import ACTION_PARAMS, parse_action
from src.dataset.builder import SYSTEM_PROMPT, build_prompt

PREFILL = '{"action": "'
Messages = List[Dict[str, str]]
Generator = Callable[[Messages, str], str]


def extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} in text (string-aware), or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def action_json_schema() -> Dict[str, Any]:
    """JSON Schema (draft 2020-12) of the closed action vocabulary."""
    branches = []
    for name, spec in sorted(ACTION_PARAMS.items()):
        branches.append(
            {
                "type": "object",
                "properties": {
                    "action": {"const": name},
                    "params": {
                        "type": "object",
                        "properties": {p: {} for p in sorted(spec)},
                        "required": sorted(spec),
                        "additionalProperties": False,
                    },
                },
                "required": ["action", "params"],
                "additionalProperties": False,
            }
        )
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "oneOf": branches}


@dataclass
class InferenceResult:
    ok: bool
    action: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    raw_outputs: List[str] = field(default_factory=list)


class ActionModel:
    """Validated log -> action inference over any generator."""

    def __init__(self, generator: Generator, max_retries: int = 2):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.generator = generator
        self.max_retries = max_retries

    def predict(self, log: str) -> InferenceResult:
        messages: Messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(log)},
        ]
        result = InferenceResult(ok=False)
        for _ in range(self.max_retries + 1):
            reply = self.generator(messages, PREFILL)
            text = reply if reply.lstrip().startswith("{") else PREFILL + reply
            raw = extract_json_object(text)
            result.attempts += 1
            result.raw_outputs.append(text)
            parsed = parse_action(raw if raw is not None else text)
            if parsed.ok:
                result.ok, result.action, result.errors = True, parsed.action, []
                return result
            result.errors = parsed.errors
            messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "That action was invalid: "
                    + "; ".join(parsed.errors)
                    + ". Return only a corrected JSON action.",
                },
            ]
        return result

    def as_predictor(self) -> Callable[[str], str]:
        """Adapt to the eval harness `prompt -> text` interface."""

        def predict(prompt: str) -> str:
            log = prompt.split("Failure log:\n", 1)[-1].rsplit("\n\nReturn the remediation", 1)[0]
            res = self.predict(log)
            return json.dumps(res.action) if res.ok else (res.raw_outputs[-1] if res.raw_outputs else "")

        return predict


class MockModel:
    """Scripted generator for tests: returns canned replies in order.

    Records every call in `.calls` as (messages, prefill). The last reply
    repeats once the script is exhausted.
    """

    def __init__(self, replies: Sequence[str]):
        if not replies:
            raise ValueError("MockModel needs at least one reply")
        self.replies = list(replies)
        self.calls: List[tuple] = []

    def __call__(self, messages: Messages, prefill: str) -> str:
        self.calls.append((messages, prefill))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
