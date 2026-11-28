"""Stand-ins for a Hugging Face model and tokenizer: no torch, no download."""
from contextlib import contextmanager


class Ids:
    """Just enough of a (1, n) tensor: the code under test only reads its shape."""

    def __init__(self, n):
        self.shape = (1, n)


class Encoding(dict):
    """Like transformers' BatchEncoding: a mapping with `.to()` and no `.shape`."""

    device = None

    def to(self, device):
        self.device = device
        return self


class StubTokenizer:
    """`reply(adapter_on) -> text` decides what the paired StubModel "generates"."""

    def __init__(self, reply=lambda adapter_on: "ok"):
        self.reply = reply
        self.conversations = []

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            return_tensors=None, return_dict=True):
        assert add_generation_prompt and return_tensors == "pt"
        self.conversations.append(messages)
        ids = Ids(3)
        return Encoding(input_ids=ids, attention_mask=Ids(3)) if return_dict else ids

    def decode(self, tokens, skip_special_tokens=False):
        assert skip_special_tokens
        return "".join(self.reply(on) for on in tokens)


class StubModel:
    device = "stub-device"

    def __init__(self):
        self.adapter_on = True
        self.calls = []

    def eval(self):
        return self

    @contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        # three prompt tokens, then one "new token" carrying the adapter state
        return [["prompt"] * 3 + [self.adapter_on]]
