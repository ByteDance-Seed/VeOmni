"""Frozen-prefix checkpoint regression: same trainable gradients, less replay."""

import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


_fix = runpy.run_path(str(Path(__file__).parents[2] / "veomni/models/transformers/qwen4_exp/frozen_prefix.py"))[
    "remove_frozen_prefix_input_grads"
]


def fix(model, args):
    return _fix(
        model,
        frozen_prefix_layers=args.train.qwen38_freeze_prefix_layers,
        text_sft=args.train.qwen38_text_sft,
        gradient_checkpointing=args.train.gradient_checkpointing.enable,
        use_reentrant=args.train.gradient_checkpointing.enable_reentrant,
        lora_enabled=bool(args.model.lora_config),
    )


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x).sin()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen4_exp")
        self.model = nn.Module()
        lm = nn.Module()
        self.model.language_model = lm
        lm.embed_tokens = nn.Embedding(19, 8)
        lm.layers = nn.ModuleList(Block() for _ in range(4))
        lm.embed_tokens.requires_grad_(False)
        for layer in lm.layers[:3]:
            layer.requires_grad_(False)
        self.head = nn.Linear(8, 3)
        self._require_grads_hook = lm.embed_tokens.register_forward_hook(lambda _m, _a, y: y.requires_grad_(True))

    def disable_input_require_grads(self):
        self._require_grads_hook.remove()

    def forward(self, ids):
        x = self.model.language_model.embed_tokens(ids)
        for layer in self.model.language_model.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return self.head(x)


def arguments(**changes):
    train = dict(
        qwen38_freeze_prefix_layers=3,
        qwen38_text_sft=True,
        gradient_checkpointing=SimpleNamespace(enable=True, enable_reentrant=False),
    )
    train.update(changes)
    return SimpleNamespace(train=SimpleNamespace(**train), model=SimpleNamespace(lora_config=None))


def test_gradients_equal_and_frozen_prefix_is_not_recomputed():
    torch.manual_seed(7)
    old = Model()
    new = copy.deepcopy(old)
    assert fix(new, arguments())
    ids = torch.tensor([[1, 4, 6, 2]])
    y0, y1 = old(ids), new(ids)
    torch.testing.assert_close(y0, y1, rtol=0, atol=0)
    y0.square().sum().backward()
    y1.square().sum().backward()
    for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None, name
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        else:
            assert a.grad is None and b.grad is None
    assert [x.calls for x in old.model.language_model.layers] == [2, 2, 2, 2]
    assert [x.calls for x in new.model.language_model.layers] == [1, 1, 1, 2]


@pytest.mark.parametrize("mode", ["reentrant", "lora", "multimodal", "no_prefix", "no_checkpoint"])
def test_unrelated_training_keeps_input_hook(mode):
    model = Model()
    args = arguments()
    if mode == "reentrant":
        args.train.gradient_checkpointing.enable_reentrant = True
    if mode == "lora":
        args.model.lora_config = {"rank": 8}
    if mode == "multimodal":
        args.train.qwen38_text_sft = False
    if mode == "no_prefix":
        args.train.qwen38_freeze_prefix_layers = 0
    if mode == "no_checkpoint":
        args.train.gradient_checkpointing.enable = False
    assert not fix(model, args)
    assert model.model.language_model.embed_tokens(torch.tensor([1])).requires_grad


def test_trainable_prefix_is_rejected():
    model = Model()
    model.model.language_model.layers[0].requires_grad_(True)
    with pytest.raises(ValueError, match="trainable embedding/prefix"):
        fix(model, arguments())


def test_real_transformers_checkpoint_hook():
    transformers = pytest.importorskip("transformers")

    class HFModel(transformers.PreTrainedModel):
        supports_gradient_checkpointing = True

        def __init__(self):
            super().__init__(transformers.PretrainedConfig())
            self.config.model_type = "qwen4_exp"
            toy = Model()
            toy.disable_input_require_grads()
            self.model, self.head = toy.model, toy.head
            self.gradient_checkpointing = False

        def get_input_embeddings(self):
            return self.model.language_model.embed_tokens

        forward = Model.forward

    torch.manual_seed(9)
    old, new = HFModel(), HFModel()
    new.load_state_dict(old.state_dict())
    for model in (old, new):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not hasattr(old, "_require_grads_hook"):
        pytest.skip("Installed Transformers does not force embedding grads without PEFT")
    assert fix(new, arguments())
    ids = torch.tensor([[1, 4, 6, 2]])
    y0, y1 = old(ids), new(ids)
    torch.testing.assert_close(y0, y1, rtol=0, atol=0)
    y0.square().sum().backward()
    y1.square().sum().backward()
    for a, b in zip(old.parameters(), new.parameters()):
        if a.requires_grad:
            assert a.grad is not None
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        else:
            assert a.grad is None and b.grad is None
    assert [x.calls for x in old.model.language_model.layers] == [2, 2, 2, 2]
    assert [x.calls for x in new.model.language_model.layers] == [1, 1, 1, 2]
