"""Qwen3-MoE SeedOmni module: registration, EP wiring, and the carrier contract.

Three independent contracts are covered:

* **Wiring** — the family registers its three factories, the modules build under
  every dense attention backend (``eager`` / ``sdpa``), the accelerated class
  declares the ``ep`` plan its fused experts need, and both module YAMLs resolve
  a real ``ep`` dimension. This is what
  ``configs/seed_omni/Qwen/moe/`` relies on.
* **Config guards** — the backbone forces ``tie_word_embeddings`` off, because it
  owns no ``lm_head`` and the shared post-load tie would otherwise run against
  its ``nn.Identity()`` embedding.
* **Carrier contract** — the composed
  ``qwen3_text_encoder.encode -> qwen3_moe_llm -> qwen3_text_encoder.decode``
  graph produces a finite loss and scatters one hidden-state slice per carrier
  segment. Numerical parity against the monolithic reference under FSDP2 and SP
  lives in ``test_qwen3_moe_accel_align.py``.
* **Inference and config** — ``generate()`` refuses SP, keeps the prompt mask
  across decode steps, and the shipped YAMLs resolve to the expected module set
  and graph endpoints.

The split checkpoint the converter writes stores experts **already fused**
(``experts.gate_up_proj`` / ``experts.down_proj``), so the attached per-expert
converter is a no-op there; it only acts on a plain per-expert HF release. That
the two layouts coincide is what makes ``ep_sharded_stream_load``'s dim-0 slice
valid; the converter's fuse/no-op behaviour is covered by
``tests/models/base/test_checkpoint_tensor_converter.py``, the streaming loader by
``tests/utils/test_ep_sharded_expert_stream_load.py``, and this file only pins the
wiring that attaches it (``num_experts`` lives on ``config.text_config``).
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.tools.training_utils import make_eager_ops_config
from veomni.arguments.arguments_types import AcceleratorConfig, ModelArguments
from veomni.arguments.omni_arguments_types import build_omni_module_runtime_args
from veomni.models.checkpoint.convert import get_checkpoint_tensor_converter
from veomni.models.checkpoint.weights import post_process_after_weight_loading
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import Qwen3ChatTemplate
from veomni.models.seed_omni.modules.qwen3.text_encoder.configuration import Qwen3TextEncoderConfig
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem


_QWEN3_MOE_TRAIN_CONFIG_DIR = "configs/seed_omni/Qwen/moe/train"
TEXT_ENCODER_MODULE = "qwen3_text_encoder"
MOE_LLM_MODULE = "qwen3_moe_llm"

_GRAPH_EDGES = [
    {"from": "qwen3_text_encoder.encode", "to": "qwen3_moe_llm"},
    {"from": "qwen3_moe_llm", "to": "qwen3_text_encoder.decode"},
    {"from": "qwen3_text_encoder.decode", "to": "end"},
]

# 2 + 1 and 1 + 3 tokens: two samples, uneven, so the composed carrier has to
# concatenate per-segment embeds rather than assume one width.
_MOE_CONVERSATIONS = [
    [
        (torch.tensor([5, 6], dtype=torch.long), torch.tensor([-100, -100], dtype=torch.long), "user"),
        (torch.tensor([7], dtype=torch.long), torch.tensor([7], dtype=torch.long), "assistant"),
    ],
    [
        (torch.tensor([11], dtype=torch.long), torch.tensor([-100], dtype=torch.long), "user"),
        (torch.tensor([12, 13, 14], dtype=torch.long), torch.tensor([12, 13, 14], dtype=torch.long), "assistant"),
    ],
]


class _FakeTokenizer:
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        del token
        return 1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


def _tiny_qwen3_moe_cfg() -> dict:
    """Smallest config that still has >1 layer, >1 head, and >1 expert per layer."""
    return {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
    }


def _build_conversation_list() -> list[list[ConversationItem]]:
    out: list[list[ConversationItem]] = []
    for sample in _MOE_CONVERSATIONS:
        items: list[ConversationItem] = []
        for ids, labels, role in sample:
            items.append(
                ConversationItem(
                    type="text",
                    value=ids.clone(),
                    role=role,
                    meta={
                        "labels": labels.clone(),
                        "attention_mask": torch.ones_like(ids),
                        _OMNI_TOKEN_LEN: int(ids.numel()),
                    },
                )
            )
        out.append(items)
    return out


def _build_qwen3_moe_modules(dtype: torch.dtype = torch.float32, *, moe_overrides: dict | None = None):
    """A fresh ``qwen3_text_encoder`` + ``qwen3_moe_llm`` pair on tiny configs."""
    TextEncoderAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_text_encoder"]()
    MoeAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()

    te_cfg = Qwen3TextEncoderConfig(
        vocab_size=128,
        hidden_size=32,
        tie_word_embeddings=True,
        ops_implementation=make_eager_ops_config(),
    )
    moe_cfg = MoeConfig(
        text_config={**_tiny_qwen3_moe_cfg(), **(moe_overrides or {})},
        ops_implementation=make_eager_ops_config(),
    )

    torch.manual_seed(2026)
    text_encoder = TextEncoderAcc(te_cfg).to(dtype=dtype)
    text_encoder._chat_template = Qwen3ChatTemplate(_FakeTokenizer())
    moe = MoeAcc(moe_cfg).to(dtype=dtype)
    return text_encoder, moe


def _run_graph(text_encoder: torch.nn.Module, moe: torch.nn.Module) -> torch.Tensor:
    omni_cfg = OmniConfig(
        _module_entries={
            "qwen3_text_encoder": {"model_path": "qwen3_text_encoder"},
            "qwen3_moe_llm": {"model_path": "qwen3_moe_llm"},
        },
        training_graphs={"default": _GRAPH_EDGES},
        generation_graphs={},
    )
    omni = OmniModel(
        omni_cfg,
        {"qwen3_text_encoder": text_encoder, "qwen3_moe_llm": moe},
    )
    out = omni.forward({"conversation_list": _build_conversation_list()}, node_runner=TrainNodeRunner())
    return out["loss"]


# --------------------------------------------------------------------------- wiring


def test_qwen3_moe_registers_config_model_and_accelerated_factories():
    """A split-checkpoint subfolder resolves through all three registries."""
    assert "qwen3_moe_llm" in set(OMNI_CONFIG_REGISTRY.valid_keys())
    assert "qwen3_moe_llm" in set(OMNI_MODEL_REGISTRY.valid_keys())
    assert "qwen3_moe_llm" in set(OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys())
    # The vocabulary half is MoE-agnostic and must stay the dense module.
    assert "qwen3_text_encoder" in set(OMNI_MODEL_REGISTRY.valid_keys())


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_qwen3_moe_builds_under_each_dense_attention_backend(attn_implementation):
    """The patched attention takes the backend explicitly, so neither spelling may be gated out."""
    Moe = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()

    cfg = MoeConfig(
        text_config=_tiny_qwen3_moe_cfg(),
        ops_implementation=make_eager_ops_config(attn_implementation=attn_implementation),
    )
    model = Moe(cfg).eval()
    assert model.config.model_type == "qwen3_moe_llm"
    # No wte / lm_head on the backbone: the embedding is an ``nn.Identity``
    # placeholder and the head is absent, because both live in
    # ``qwen3_text_encoder``. Asserting ``is not None`` would pass either way.
    assert isinstance(model.language_model.embed_tokens, nn.Identity)
    assert model.get_output_embeddings() is None
    assert not hasattr(model, "lm_head")


def test_qwen3_moe_declares_the_expert_parallel_plan():
    """Both fused expert tensors are ``Shard(0)`` over ``ep``, keyed module-locally."""
    MoeAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    model = MoeAcc(MoeConfig(text_config=_tiny_qwen3_moe_cfg(), ops_implementation=make_eager_ops_config()))

    plan = model.get_parallel_plan().extra_parallel_plan
    assert set(plan) == {"ep"}
    # ``self.language_model`` is the bare ``Qwen3MoeModel``, so no ``model.`` prefix.
    assert set(plan["ep"]) == {
        "language_model.layers.*.mlp.experts.gate_up_proj",
        "language_model.layers.*.mlp.experts.down_proj",
    }
    assert all(spec.dim == 0 for spec in plan["ep"].values())


def test_qwen3_moe_ep_sharded_stream_load_applies_to_the_backbone_only():
    """``ep_sharded_stream_load`` is opt-in per module, so the text encoder keeps the broadcast load.

    This is the per-module accelerator override the split config depends on: the
    flag reads each rank's own expert rows, and the encoder has none.

    The effective ``ep`` dimension is asserted rather than the ``ep_size`` field,
    because on a per-module entry ``ep_size`` is inert: the config is merged onto
    ``asdict(ModelArguments)``, which already lists ``"ep"``, so
    ``AcceleratorConfig.__post_init__`` never appends it. Asserting the field
    would pass while Expert Parallel is silently disabled.
    """
    global_args = ModelArguments(model_path="/tmp/qwen3-moe-unused-root")
    modules = build_omni_module_runtime_args(
        global_args=global_args,
        model_path="/tmp/qwen3-moe-unused-root",
        modules=f"{_QWEN3_MOE_TRAIN_CONFIG_DIR}/modules_train.yaml",
    )

    assert set(modules) == {"qwen3_text_encoder", "qwen3_moe_llm"}
    backbone = modules["qwen3_moe_llm"]
    encoder = modules["qwen3_text_encoder"]

    def effective_ep_dim(args) -> int:
        return dict(zip(args.accelerator.extra_parallel_names, args.accelerator.extra_parallel_sizes)).get("ep", 1)

    assert effective_ep_dim(backbone) == 4
    assert effective_ep_dim(encoder) == 1

    assert backbone.ep_sharded_stream_load is True
    # Streaming reads every rank's own slice, so rank0-broadcast must be off.
    assert backbone.broadcast_model_weights_from_rank0 is False

    assert encoder.ep_sharded_stream_load is False
    assert encoder.broadcast_model_weights_from_rank0 is True


def test_qwen3_moe_ep_lists_stay_aligned_when_the_launcher_adds_a_dim():
    """The three ``extra_parallel_*`` lists are paired by position.

    A module entry inherits the launcher accelerator's lists, so writing only
    ``extra_parallel_sizes`` leaves a stale name/placement behind as soon as the
    launcher contributes another dim (a text encoder's ``emb`` is the realistic
    case). The lengths then disagree, and ``ParallelState`` rejects the mesh —
    or, worse, ``emb`` silently takes the ``ep`` size. Both shipped YAMLs must
    pin all three lists.
    """
    launcher = AcceleratorConfig(
        extra_parallel_names=["emb"],
        extra_parallel_sizes=[8],
        extra_parallel_placement_innermost=[False],
    )
    global_args = ModelArguments(model_path="/tmp/qwen3-moe-unused-root")
    global_args.accelerator = launcher

    for filename in ("train/modules_train.yaml", "infer/modules_infer_fsdp.yaml"):
        modules = build_omni_module_runtime_args(
            global_args=global_args,
            model_path="/tmp/qwen3-moe-unused-root",
            modules=f"configs/seed_omni/Qwen/moe/{filename}",
        )
        acc = modules["qwen3_moe_llm"].accelerator
        lengths = {
            len(acc.extra_parallel_names),
            len(acc.extra_parallel_sizes),
            len(acc.extra_parallel_placement_innermost),
        }
        assert lengths == {1}, (filename, acc.extra_parallel_names, acc.extra_parallel_sizes)
        assert acc.extra_parallel_names == ["ep"]
        assert acc.extra_parallel_sizes == [4]


def test_qwen3_moe_inference_configs_resolve_the_intended_ep_dim():
    """The fsdp inference YAML enables ``ep``; the eager one must stay single-process.

    ``modules_infer_eager.yaml`` is the un-distributed default, so an ``ep`` dim
    there would ask for an EP mesh with no wrap to carry it.
    """
    for filename in ("modules_infer_fsdp.yaml", "modules_infer_eager.yaml"):
        modules = build_omni_module_runtime_args(
            global_args=ModelArguments(model_path="/tmp/qwen3-moe-unused-root"),
            model_path="/tmp/qwen3-moe-unused-root",
            modules=f"{_QWEN3_MOE_TRAIN_CONFIG_DIR.rsplit('/', 1)[0]}/infer/{filename}",
        )
        backbone = modules["qwen3_moe_llm"]
        effective = dict(
            zip(backbone.accelerator.extra_parallel_names, backbone.accelerator.extra_parallel_sizes)
        ).get("ep", 1)
        expected = 1 if "eager" in filename else 4
        assert effective == expected, (filename, effective)


def test_qwen3_moe_backbone_never_ties_embeddings():
    """An upstream ``tie_word_embeddings: true`` must not reach the post-load tie.

    ``Qwen3MoeLlm`` owns no ``lm_head`` (the vocabulary module does), so the shared
    tie would run against this class's ``nn.Identity()`` input embedding and raise
    ``RuntimeError: Failed to tie input/output embeddings``. Only the *outer* flag
    is forced off — enough to short-circuit the AND, and it leaves ``text_config``
    as the caller wrote it, since the real tie lives in ``qwen3_text_encoder``.
    """
    MoeAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()

    cfg = MoeConfig(
        text_config=dict(_tiny_qwen3_moe_cfg(), tie_word_embeddings=True),
        # Pin the backend: unset means ``veomni_flash_attention_2``, which a CPU
        # host cannot construct (the autouse guard rejects the preload) — the
        # tie check would never be reached.
        ops_implementation=make_eager_ops_config(),
    )
    assert cfg.tie_word_embeddings is False
    # The nested config is the caller's; the vocabulary module reads it for the one
    # tie that is real, so it must keep whatever the upstream checkpoint declares.
    assert cfg.text_config.tie_word_embeddings is True

    model = MoeAcc(cfg)
    # The real post-load hook, not a reimplementation of its condition.
    post_process_after_weight_loading(model, {})


# ------------------------------------------------------------------ checkpoint wiring


def test_qwen3_moe_backbone_wires_a_converter_on_its_nested_config():
    """The converter factory must read ``num_experts`` off ``config.text_config``.

    This wrapper keeps the upstream config nested, so ``model.config.num_experts``
    does not exist — a factory reading it would raise at load time. The converter's
    own fuse/no-op behaviour is already covered by
    ``tests/models/base/test_checkpoint_tensor_converter.py``; what is specific to
    this module is that the factory is attached and resolves the right expert count.
    """
    _text_encoder, moe = _build_qwen3_moe_modules()

    converter = get_checkpoint_tensor_converter(moe)
    assert converter is not None
    assert converter.num_experts == _tiny_qwen3_moe_cfg()["num_experts"]
    # Nothing on the wrapper itself: the count only exists on the nested config.
    assert not hasattr(moe.config, "num_experts")


def test_qwen3_moe_config_accepts_an_instantiated_text_config():
    """``text_config`` may arrive as a mapping, a ``Qwen3MoeConfig``, or ``None``.

    ``from_dict`` round-trips hand back a config object, and a caller building the
    inner config itself would otherwise hit
    ``TypeError: argument after ** must be a mapping``. The dense Qwen3 sibling
    already accepts all three.
    """
    from transformers import Qwen3MoeConfig

    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    hidden_size = 32

    inner = Qwen3MoeConfig(vocab_size=128, hidden_size=hidden_size, num_hidden_layers=1)
    from_instance = MoeConfig(text_config=inner)
    assert from_instance.text_config.tie_word_embeddings is False
    assert from_instance.text_config.hidden_size == hidden_size

    from_mapping = MoeConfig(text_config={"vocab_size": 128, "hidden_size": hidden_size})
    assert isinstance(from_mapping.text_config, Qwen3MoeConfig)
    assert from_mapping.text_config.hidden_size == hidden_size

    assert isinstance(MoeConfig().text_config, Qwen3MoeConfig)


def test_qwen3_moe_config_leaves_the_nested_tie_flag_alone():
    """``text_config`` is the caller's object and its tie flag is theirs to set.

    The wrapper forces ``tie_word_embeddings`` off on its *own* top-level config,
    because it has no output head to tie. The nested config belongs to the upstream
    architecture: ``qwen3_text_encoder`` reads it for the one real
    ``embed_tokens``↔``lm_head`` tie, so the wrapper must neither replace the object
    nor rewrite the field.
    """
    from transformers import Qwen3MoeConfig

    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    inner = Qwen3MoeConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1, tie_word_embeddings=True)

    wrapper = MoeConfig(text_config=inner)

    assert inner.tie_word_embeddings is True, "the caller's object must be untouched"
    assert wrapper.text_config is inner
    assert wrapper.text_config.tie_word_embeddings is True
    assert wrapper.tie_word_embeddings is False
    # The nested config still drives the backbone it was written for.
    assert wrapper.text_config.vocab_size == 128
    assert wrapper.text_config.num_hidden_layers == 1


# ----------------------------------------------------------------------------- parity


def test_qwen3_moe_graph_reaches_the_conversation_carrier():
    """A full forward through the composed graph produces a finite loss.

    This pins the carrier contract — every non-dummy segment receives hidden
    states, so the decode head sees the whole sequence. Numerical parity against
    the monolithic reference, under FSDP2 and SP, is
    ``test_qwen3_moe_accel_align.py``; that file also carries the tolerances.
    """
    text_encoder, moe = _build_qwen3_moe_modules()

    loss = _run_graph(text_encoder, moe)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_qwen3_moe_backbone_returns_one_hidden_state_per_non_dummy_segment():
    """``forward_post`` scatters the packed backbone output back onto every segment.

    The composed decode head needs one hidden-state slice per carrier item; getting
    the count or the lengths wrong silently misaligns the labels against the logits.
    The backbone reads ``item.value`` as an already-embedded ``[T, D]`` tensor (the
    text encoder produced it upstream), so the carrier here carries embeddings.
    """
    _text_encoder, moe = _build_qwen3_moe_modules()
    hidden_size = _tiny_qwen3_moe_cfg()["hidden_size"]

    torch.manual_seed(7)
    carrier: list[list[ConversationItem]] = []
    for ids, labels, role in [seg for sample in _MOE_CONVERSATIONS for seg in sample]:
        length = int(ids.numel())
        carrier.append(
            [
                ConversationItem(
                    type="text",
                    value=torch.randn(length, hidden_size),
                    role=role,
                    meta={
                        "labels": labels.clone(),
                        "attention_mask": torch.ones(length, dtype=torch.long),
                        _OMNI_TOKEN_LEN: length,
                    },
                )
            ]
        )

    expected_lengths = [int(item.value.shape[0]) for sample in carrier for item in sample]

    moe.train()
    pre = moe.pre_forward("forward", conversation_list=carrier)
    outputs = moe.forward(**pre)
    result = moe.post_forward("forward", **outputs)

    scattered = [item.value for sample in result["conversation_list"] for item in sample]
    assert [int(t.shape[0]) for t in scattered] == expected_lengths
    assert all(torch.isfinite(t).all() for t in scattered)
    # The backbone is hidden-size wide, not vocab-wide: no lm_head lives here.
    assert all(t.shape[-1] == hidden_size for t in scattered)


# ------------------------------------------------------------------- inference


def test_qwen3_moe_generate_rejects_ulysses_sequence_parallel(monkeypatch):
    """``generate()`` feeds the full prompt, which conflicts with Ulysses' shard."""
    import veomni.models.seed_omni.modules.base.llm_packing as llm_packing

    monkeypatch.setattr(llm_packing, "get_parallel_state", lambda: SimpleNamespace(sp_size=2))
    Moe = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    moe = Moe(MoeConfig(text_config=_tiny_qwen3_moe_cfg(), ops_implementation=make_eager_ops_config()))

    with pytest.raises(ValueError, match="sp_size=1"):
        moe.generate(conversation_list=[ConversationItem(type="text", value="hi", role="user")])


def test_qwen3_moe_decode_keeps_the_prompt_attention_mask(monkeypatch):
    """Masked prompt positions stay masked once their keys are in the KV cache."""
    import veomni.models.seed_omni.modules.base.llm_packing as llm_packing

    monkeypatch.setattr(llm_packing, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))
    Moe = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    moe = Moe(MoeConfig(text_config=_tiny_qwen3_moe_cfg(), ops_implementation=make_eager_ops_config()))

    seen: list[torch.Tensor | None] = []
    real_forward = moe.forward

    def record_forward(*, attention_mask=None, **kwargs):
        seen.append(None if attention_mask is None else attention_mask.detach().clone())
        return real_forward(attention_mask=attention_mask, **kwargs)

    moe.forward = record_forward

    hidden = _tiny_qwen3_moe_cfg()["hidden_size"]
    conversation = [
        ConversationItem(
            type="text",
            value=torch.randn(4, hidden),
            role="user",
            meta={"attention_mask": torch.tensor([1, 1, 0, 0]), _OMNI_TOKEN_LEN: 4},
        )
    ]
    moe.generate(conversation)  # prefill
    moe.generate(conversation)  # first decode

    assert seen[0].tolist() == [[1, 1, 0, 0]]
    assert seen[1].tolist() == [[1, 1, 0, 0, 1]]


def test_qwen3_moe_train_and_infer_yaml_configs_load():
    """The shipped YAMLs resolve to the expected modules and graph endpoints."""
    from veomni.arguments.omni_arguments_types import (
        OmniArguments,
        OmniDataArguments,
        OmniInferArguments,
        OmniModelRuntimeArguments,
        build_omni_model_runtime_args,
    )

    cfg_dir = Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Qwen" / "moe"
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=".",
            model_config={
                "modules": str(cfg_dir / "train/modules_train.yaml"),
                "train_graph": str(cfg_dir / "train/graph_train.yaml"),
                "infer_graph": str(cfg_dir / "train/graph_infer.yaml"),
            },
        ),
        data=OmniDataArguments(train_path="."),
        infer=OmniInferArguments(),
    )
    cfg = build_omni_model_runtime_args(args).to_hf_config()

    assert set(cfg._module_entries) == {TEXT_ENCODER_MODULE, MOE_LLM_MODULE}
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert f"{TEXT_ENCODER_MODULE}.encode" in endpoints and MOE_LLM_MODULE in endpoints
    assert cfg.generation_graph["initial"] == "text_ar"


# ----------------------------------------------------------------------- freeze


def test_qwen3_moe_freeze_model_freezes_the_backbone():
    """``freeze: true`` must actually freeze — the runtime calls this hook only if it exists.

    Without it the flag would be stored but never applied, leaving the backbone
    trainable and in the optimizer, unlike dense Qwen3.
    """
    Moe = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()

    frozen = Moe(MoeConfig(text_config=_tiny_qwen3_moe_cfg(), freeze=True, ops_implementation=make_eager_ops_config()))
    frozen.freeze_model()
    assert all(not p.requires_grad for p in frozen.language_model.parameters())

    trainable = Moe(
        MoeConfig(text_config=_tiny_qwen3_moe_cfg(), freeze=False, ops_implementation=make_eager_ops_config())
    )
    trainable.freeze_model()
    assert all(p.requires_grad for p in trainable.language_model.parameters())


# ------------------------------------------------------------------------ flops


def test_qwen3_moe_flops_account_for_dense_layers_mixed_into_the_stack():
    """``mlp_only_layers`` / ``decoder_sparse_step`` build dense MLPs; charge them as such.

    Qwen3-30B-A3B proportions: ``moe_intermediate_size`` * top-k (768 × 8) exceeds
    ``intermediate_size`` (2048), so a dense layer is *cheaper* than a sparse one.
    Counting every layer as sparse over-reports FLOPs for those configs.
    """
    MoeAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()

    base = {
        "vocab_size": 256,
        "hidden_size": 256,
        "intermediate_size": 1024,
        "moe_intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 128,
        "num_experts": 16,
        "num_experts_per_tok": 4,
    }

    def flops(**overrides) -> float:
        cfg = MoeConfig(text_config={**base, **overrides}, ops_implementation=make_eager_ops_config())
        return MoeAcc(cfg).estimate_flops([256])

    all_sparse = flops()
    every_other_dense = flops(decoder_sparse_step=2)
    all_dense = flops(mlp_only_layers=[0, 1, 2, 3])

    # 512 × 4 (top-k) = 2048 > 1024, so sparse layers cost more than dense ones.
    assert all_sparse > every_other_dense > all_dense


def test_qwen3_moe_shipped_base_yaml_targets_a_split_root():
    """``base.yaml`` must not point at an upstream root while streaming is on.

    The HF-view validator rejects ``ep_sharded_stream_load`` because streaming reads
    raw checkpoint slices and cannot apply the layout's key renames, so an upstream
    ``model_path`` plus the shipped streaming overlay fails at startup. Guard the
    pairing here rather than in a doc note a reader can miss.
    """
    import yaml

    config_dir = Path(__file__).resolve().parents[4] / "configs/seed_omni/Qwen/moe/train"
    base = yaml.safe_load((config_dir / "base.yaml").read_text(encoding="utf-8"))
    modules = yaml.safe_load((config_dir / "modules_train.yaml").read_text(encoding="utf-8"))

    assert modules["qwen3_moe_llm"]["ep_sharded_stream_load"] is True
    # A split root: the converted output, not the upstream checkpoint directory.
    assert str(base["model"]["model_path"]).endswith("-split"), base["model"]["model_path"]


# --------------------------------------------------------------------- aux loss


def test_qwen3_moe_router_aux_loss_joins_the_graph_loss_when_enabled():
    """A mid-graph module contributes by returning ``_loss``; ``OmniModel`` sums every node's.

    With ``output_router_logits`` on, the backbone adds
    ``router_aux_loss_coef * load_balancing_loss`` — the objective the upstream
    ``Qwen3MoeForCausalLM`` would compute. Off by default, so the common path must
    add nothing at all.
    """

    def losses(router_logits: bool) -> dict[str, float]:
        text_encoder, moe = _build_qwen3_moe_modules(
            moe_overrides={"output_router_logits": router_logits, "router_aux_loss_coef": 0.01}
        )
        omni_cfg = OmniConfig(
            _module_entries={
                TEXT_ENCODER_MODULE: {"model_path": TEXT_ENCODER_MODULE},
                MOE_LLM_MODULE: {"model_path": MOE_LLM_MODULE},
            },
            training_graphs={"default": _GRAPH_EDGES},
            generation_graphs={},
        )
        omni = OmniModel(omni_cfg, {TEXT_ENCODER_MODULE: text_encoder, MOE_LLM_MODULE: moe})
        out = omni.forward({"conversation_list": _build_conversation_list()}, node_runner=TrainNodeRunner())
        return {name: float(value.detach()) for name, value in out["losses"].items()}

    off = losses(router_logits=False)
    assert set(off) == {f"{TEXT_ENCODER_MODULE}.decode"}

    on = losses(router_logits=True)
    assert set(on) == {f"{TEXT_ENCODER_MODULE}.decode", f"{MOE_LLM_MODULE}.forward"}
    # Positive coefficient on a load-balancing term: real, finite, non-zero.
    assert on[f"{MOE_LLM_MODULE}.forward"] > 0
    assert math.isfinite(on[f"{MOE_LLM_MODULE}.forward"])


# ------------------------------------------------------------------------- lora


def test_qwen3_moe_maps_semantic_moe_lora_targets_onto_the_fused_experts():
    """``gate_proj`` / ``up_proj`` / ``down_proj`` must reach the fused expert parameters.

    Those names target per-expert ``nn.Linear`` modules in an unfused model, but
    this backbone stores v5 fused ``experts.gate_up_proj`` / ``experts.down_proj``.
    Without the hook the resolver keeps the names as module targets, they match
    nothing, and the experts silently get no adapters while attention does.
    """
    from veomni.lora.target_mapping import resolve_fused_moe_lora_targets

    _text_encoder, moe = _build_qwen3_moe_modules()
    resolved = resolve_fused_moe_lora_targets(moe, {"lora_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"]})

    # Semantic expert names collapse into the two fused parameters.
    assert resolved["lora_modules"] == ["q_proj"]
    assert resolved["target_parameters"] == [
        "language_model.layers.*.mlp.experts.gate_up_proj",
        "language_model.layers.*.mlp.experts.down_proj",
    ]


def test_qwen3_moe_lora_targets_fail_loudly_when_they_match_nothing():
    """A pattern that reaches no parameter raises, rather than injecting nothing."""
    from veomni.lora.target_mapping import resolve_fused_moe_lora_targets

    _text_encoder, moe = _build_qwen3_moe_modules()
    # Hide the backbone's layers so the produced expert patterns match nothing.
    moe.language_model.layers = nn.ModuleList()

    with pytest.raises(ValueError, match="did not match any parameter"):
        resolve_fused_moe_lora_targets(moe, {"lora_modules": ["gate_proj"]})


def test_qwen3_moe_runtime_freeze_step_leaves_nothing_trainable():
    """The freeze must land through the runtime hook that precedes optimizer setup.

    Calling ``freeze_model()`` directly only proves the method works; what decides
    whether the backbone enters the optimizer is ``_freeze_model_module``, which
    invokes the hook only when the attribute exists. This drives that path, so a
    wrapper that quietly loses the method fails here rather than in a 30B run.
    """
    from veomni.models.model_runtime import VeOmniModelRuntime

    Moe = OMNI_MODEL_REGISTRY["qwen3_moe_llm"]()
    MoeConfig = OMNI_CONFIG_REGISTRY["qwen3_moe_llm"]()
    moe = Moe(MoeConfig(text_config=_tiny_qwen3_moe_cfg(), freeze=True, ops_implementation=make_eager_ops_config()))
    assert any(p.requires_grad for p in moe.parameters()), "precondition: starts trainable"

    # The runtime's own step, invoked unbound so no FSDP/mesh setup is needed.
    VeOmniModelRuntime._freeze_model_module(SimpleNamespace(model=moe, _setup_lora=lambda: None))

    assert not any(p.requires_grad for p in moe.language_model.parameters())
