# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""Forward isolation for instance-local attention and rope handles.

Handle snapshots in ``test_auto_registry.py`` stay, but they do not execute
``forward``. These cases construct two models under different ops configs,
poison the global config, then run text / vision / condition forwards.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import torch

from tests.models.compare import eager_ops_config, ops_config_scope, qwen_image_inputs
from tests.models.tiny_configs import tiny_qwen2_config, tiny_qwen2_vl_config, tiny_wan_config
from veomni.ops import VeomniOp


def _output_tensor(result):
    if torch.is_tensor(result):
        return result
    if hasattr(result, "logits") and result.logits is not None:
        return result.logits
    if hasattr(result, "last_hidden_state"):
        return result.last_hidden_state
    if isinstance(result, (tuple, list)):
        return _output_tensor(result[0])
    raise TypeError(f"unsupported forward result {type(result)!r}")


def _handle(model, path: str) -> VeomniOp:
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    assert isinstance(module, VeomniOp), path
    return module


def _attn_cfg(impl: str) -> SimpleNamespace:
    cfg = eager_ops_config()
    cfg.attn_implementation = impl
    return cfg


def _poison_cfg() -> SimpleNamespace:
    cfg = eager_ops_config()
    cfg.attn_implementation = "flex_attention"
    cfg.rotary_pos_emb_implementation = "liger_kernel"
    cfg.rotary_pos_emb_vision_implementation = "npu"
    cfg.rms_norm_implementation = "liger_kernel"
    return cfg


def _assert_isolated(
    *,
    build: Callable[[], torch.nn.Module],
    run: Callable[[torch.nn.Module], object],
    attn_paths: Sequence[str],
    sticky_paths: Sequence[str] = (),
    eager_cfg: SimpleNamespace,
    alt_cfg: SimpleNamespace,
    poison_cfg: SimpleNamespace,
) -> None:
    with ops_config_scope(eager_cfg):
        seed = build()
        state = {key: value.detach().clone() for key, value in seed.state_dict().items()}

    def construct(cfg: SimpleNamespace) -> torch.nn.Module:
        with ops_config_scope(cfg):
            model = build()
        model.load_state_dict(state)
        model.eval()
        return model

    def check(model: torch.nn.Module, attn_impl: str) -> None:
        for path in attn_paths:
            assert _handle(model, path).impl == attn_impl, path
        for path in sticky_paths:
            assert _handle(model, path).impl == "eager", path

    eager = construct(eager_cfg)
    alternate = construct(alt_cfg)
    check(eager, "eager")
    check(alternate, alt_cfg.attn_implementation)

    with torch.no_grad():
        with ops_config_scope(eager_cfg):
            eager_ref = _output_tensor(run(eager))
        with ops_config_scope(alt_cfg):
            alt_ref = _output_tensor(run(alternate))
        with ops_config_scope(poison_cfg):
            eager_out = _output_tensor(run(eager))
            alt_out = _output_tensor(run(alternate))
            check(eager, "eager")
            check(alternate, alt_cfg.attn_implementation)

    torch.testing.assert_close(eager_out, eager_ref)
    torch.testing.assert_close(alt_out, alt_ref)


def test_qwen2_text_forward_keeps_construction_impls():
    from veomni.models.transformers.qwen2.generated.patched_modeling_qwen2_gpu import Qwen2ForCausalLM

    config = tiny_qwen2_config()
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    _assert_isolated(
        build=lambda: Qwen2ForCausalLM(config),
        run=lambda model: model(input_ids=input_ids, use_cache=False),
        attn_paths=("model.layers.0.self_attn.veomni_attn",),
        sticky_paths=("model.layers.0.self_attn.veomni_rope",),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_qwen2_vl_vision_forward_keeps_construction_impls():
    from veomni.models.transformers.qwen2_vl.generated.patched_modeling_qwen2_vl_gpu import (
        Qwen2VLForConditionalGeneration,
    )

    config = tiny_qwen2_vl_config()
    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    image.pop("labels")
    _assert_isolated(
        build=lambda: Qwen2VLForConditionalGeneration(config),
        run=lambda model: model(input_ids=ids, use_cache=False, **image),
        attn_paths=(
            "model.language_model.layers.0.self_attn.veomni_attn",
            "model.visual.blocks.0.attn.veomni_attn",
        ),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
    )


def test_wan_condition_forward_keeps_construction_impls():
    from veomni.models.transformers.wan.modeling_wan import WanModel

    config = tiny_wan_config()
    inputs = {
        "x": torch.randn(2, config.in_dim, 2, 8, 8),
        "timestep": torch.rand(2),
        "context": torch.randn(2, config.text_len, config.text_dim),
    }
    _assert_isolated(
        build=lambda: WanModel(config),
        run=lambda model: model(**inputs),
        attn_paths=("blocks.0.self_attn.attn.veomni_attn",),
        sticky_paths=("blocks.0.self_attn.veomni_rope",),
        eager_cfg=_attn_cfg("eager"),
        alt_cfg=_attn_cfg("sdpa"),
        poison_cfg=_poison_cfg(),
    )
