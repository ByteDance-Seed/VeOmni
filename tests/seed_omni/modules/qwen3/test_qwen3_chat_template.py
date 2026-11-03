"""Unit tests for the Qwen3 ChatML template, preprocessor, and modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.helpers import load_from_omni, save_as_omni
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
)
from veomni.models.seed_omni.modules.base.llm_packing import (
    pack_llm_conversations_for_forward,
    scatter_llm_hidden_states,
)
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import Qwen3ChatTemplate
from veomni.models.seed_omni.modules.qwen3.text_encoder.configuration import Qwen3TextEncoderConfig
from veomni.models.seed_omni.modules.qwen3.text_encoder.processing import Qwen3TextEncoderPreprocessor
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from veomni.utils.tensor_utils import naflatten


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class FakeTokenizer:
    """Minimal tokenizer for chat-template construction (markers are fixed literals)."""

    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


def _template() -> Qwen3ChatTemplate:
    return Qwen3ChatTemplate(FakeTokenizer())


def test_apply_qwen3_chat_template_wraps_roles():
    sample = [
        ConversationItem(type="text", value="hi", role="user"),
        ConversationItem(type="text", value="hello", role="assistant"),
    ]
    parts = _template().apply_chat_template(sample)
    # Standard Qwen ChatML: every turn closed with <|im_end|>\n.
    assert parts[0].value == "<|im_start|>user\n"
    assert parts[1].value == "hi"
    assert parts[2].value == "<|im_end|>\n"  # user turn closed (unsupervised)
    assert parts[2].meta["loss_mask"] == 0
    assert parts[3].value == "<|im_start|>assistant\n"
    assert parts[4].value == "hello"
    assert parts[4].meta["loss_mask"] == 1
    assert parts[5].value == "<|im_end|>\n"  # assistant turn closed (supervised)
    assert parts[5].meta["loss_mask"] == 1


def test_apply_qwen3_generation_prompt_appends_assistant_prefix():
    # Input is an already-templated (turn-closed) prompt; generation prompt only
    # appends the assistant prefix.
    sample = _template().apply_chat_template([ConversationItem(type="text", value="hi", role="user")])
    parts = _template().apply_generation_prompt(sample)
    assert parts[-1].value == "<|im_start|>assistant\n"
    assert parts[-2].value == "<|im_end|>\n"  # closing of the user turn from apply_chat_template


def test_qwen3_text_preprocessor_matches_inmodule_pipeline_and_sets_token_len():
    tmpl = _template()

    def _raw_sample():
        return [
            ConversationItem(type="text", value="hello", role="user"),
            ConversationItem(type="text", value="hi there", role="assistant"),
        ]

    batch = [_raw_sample(), _raw_sample()]
    Qwen3TextEncoderPreprocessor(tmpl)({"conversation_list": batch})

    worker_ids = []
    for sample in batch:
        for part in sample:
            if part.type == "text":
                assert part.value.dtype == torch.long and part.value.device.type == "cpu"
                assert part.meta[_OMNI_TOKEN_LEN] == int(part.value.numel())
        worker_ids.extend(tmpl.pack_input_ids(sample))
    worker_flat, worker_shape = naflatten(worker_ids)

    ref_ids = []
    for sample in [_raw_sample(), _raw_sample()]:
        parts = tmpl.tokenize_conversation(sample)
        ref_ids.extend(tmpl.pack_input_ids(parts))
    ref_flat, ref_shape = naflatten(ref_ids)

    assert torch.equal(worker_flat, ref_flat)
    assert torch.equal(worker_shape, ref_shape)


def _short_dialog() -> list[ConversationItem]:
    return [
        ConversationItem(type="text", value="hello", role="user"),
        ConversationItem(type="text", value="hi there", role="assistant"),
    ]


def test_qwen3_tokenize_conversation_suffix_truncates_and_preserves_assistant_answer():
    tmpl = _template()

    def _long_prompt_short_answer() -> list[ConversationItem]:
        return [
            ConversationItem(type="text", value="x" * 64, role="user"),
            ConversationItem(type="text", value="42", role="assistant"),
        ]

    full = tmpl.tokenize_conversation(_long_prompt_short_answer())
    full_ids = torch.cat(tmpl.pack_input_ids(full))
    full_labels = torch.cat([part.meta["labels"] for part in full])
    full_mask = torch.cat([part.meta["attention_mask"] for part in full])

    # Budget is smaller than the user prompt alone, so prefix truncation would
    # discard the entire assistant response and leave only -100 labels.
    budget = 40
    assert int(full[0].meta[_OMNI_TOKEN_LEN]) > budget

    truncated = tmpl.tokenize_conversation(_long_prompt_short_answer(), max_seq_len=budget)
    trunc_len = sum(int(part.meta[_OMNI_TOKEN_LEN]) for part in truncated)
    assert trunc_len == budget

    trunc_ids = torch.cat(tmpl.pack_input_ids(truncated))
    trunc_labels = torch.cat([part.meta["labels"] for part in truncated])
    trunc_mask = torch.cat([part.meta["attention_mask"] for part in truncated])
    assert torch.equal(trunc_ids, full_ids[-budget:])
    assert torch.equal(trunc_labels, full_labels[-budget:])
    assert torch.equal(trunc_mask, full_mask[-budget:])
    assert int((trunc_labels != -100).sum()) == int((full_labels != -100).sum()) > 0


def test_qwen3_text_preprocessor_forwards_max_seq_len_when_training():
    tmpl = _template()
    batch = [_short_dialog()]
    Qwen3TextEncoderPreprocessor(tmpl)({"conversation_list": batch}, max_seq_len=8)
    assert sum(int(part.meta[_OMNI_TOKEN_LEN]) for part in batch[0]) == 8

    # Inference leaves the sequence uncapped even if max_seq_len is passed.
    infer_batch = [_short_dialog()]
    Qwen3TextEncoderPreprocessor(tmpl)(
        {"conversation_list": infer_batch},
        inference=True,
        max_seq_len=8,
    )
    assert sum(int(part.meta[_OMNI_TOKEN_LEN]) for part in infer_batch[0]) > 8


def _tiny_qwen3_cfg() -> dict:
    return {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 128,
    }


def _save_fake_fast_tokenizer(save_dir: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "<|im_start|>": 4, "<|im_end|>": 5}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    ).save_pretrained(save_dir)


def test_qwen3_llm_and_text_encoder_save_reload_via_registry(tmp_path: Path):
    Qwen3Llm = OMNI_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    Qwen3TextEncoder = OMNI_MODEL_REGISTRY["qwen3_text_encoder"]()

    llm = Qwen3Llm(
        Qwen3LlmConfig(
            text_config=_tiny_qwen3_cfg(),
            ops_implementation={"attn_implementation": "eager", "rms_norm_implementation": "eager"},
        )
    )
    assert llm.config.text_config._attn_implementation == "eager"
    assert llm.config.text_config.ops_config.rms_norm_implementation == "eager"
    save_as_omni(tmp_path / "llm", "qwen3_llm", llm)
    cfg, llm2 = load_from_omni(tmp_path / "llm", "qwen3_llm")
    assert cfg.model_type == "qwen3_llm"
    assert cfg.text_config.ops_config.rms_norm_implementation == "eager"
    assert isinstance(llm2, Qwen3Llm)
    assert isinstance(llm2.language_model.get_input_embeddings(), torch.nn.Identity)

    te = Qwen3TextEncoder(Qwen3TextEncoderConfig(vocab_size=128, hidden_size=32, tie_word_embeddings=True))
    _save_fake_fast_tokenizer(save_as_omni(tmp_path / "te", "qwen3_text_encoder", te))
    rcfg, te2 = load_from_omni(tmp_path / "te", "qwen3_text_encoder")
    assert rcfg.model_type == "qwen3_text_encoder"
    assert isinstance(te2, Qwen3TextEncoder)
    assert te2.config.vocab_size == 128


def test_qwen3_llm_rejects_flex_attention():
    """The packed path hands the backbone a 2-D mask, which Flex cannot isolate."""
    Qwen3Llm = OMNI_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    with pytest.raises(ValueError, match="flex_attention"):
        Qwen3Llm(
            Qwen3LlmConfig(
                text_config=_tiny_qwen3_cfg(),
                ops_implementation={"attn_implementation": "flex_attention"},
            )
        )


def test_qwen3_llm_rejects_nested_text_config_flex_attention():
    """Outer eager must not let an explicit nested Flex selection reach the backbone."""
    Qwen3Llm = OMNI_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    nested = {**_tiny_qwen3_cfg(), "attn_implementation": "flex_attention"}
    cfg = Qwen3LlmConfig(
        text_config=nested,
        ops_implementation={"attn_implementation": "eager"},
    )
    assert cfg._attn_implementation == "eager"
    assert "flex_attention" in cfg.text_config._attn_implementation
    with pytest.raises(ValueError, match="flex_attention"):
        Qwen3Llm(cfg)


def test_qwen3_llm_generate_rejects_ulysses_sequence_parallel(monkeypatch):
    import veomni.models.seed_omni.modules.base.llm_packing as llm_packing

    monkeypatch.setattr(llm_packing, "get_parallel_state", lambda: SimpleNamespace(sp_size=2))
    Qwen3Llm = OMNI_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    llm = Qwen3Llm(
        Qwen3LlmConfig(
            text_config=_tiny_qwen3_cfg(),
            ops_implementation={"attn_implementation": "eager"},
        )
    )
    with pytest.raises(ValueError, match="sp_size=1"):
        llm.generate(conversation_list=[ConversationItem(type="text", value="hi", role="user")])


def test_qwen3_llm_decode_keeps_the_prompt_attention_mask(monkeypatch):
    """Masked prompt positions stay masked once their keys are in the cache."""
    import veomni.models.seed_omni.modules.base.llm_packing as llm_packing

    monkeypatch.setattr(llm_packing, "get_parallel_state", lambda: SimpleNamespace(sp_size=1))
    Qwen3Llm = OMNI_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    llm = Qwen3Llm(
        Qwen3LlmConfig(
            text_config=_tiny_qwen3_cfg(),
            ops_implementation=make_eager_ops_config(attn_implementation="eager"),
        )
    )

    seen: list[torch.Tensor | None] = []
    real_forward = llm.forward

    def record_forward(*, attention_mask=None, **kwargs):
        seen.append(None if attention_mask is None else attention_mask.detach().clone())
        return real_forward(attention_mask=attention_mask, **kwargs)

    llm.forward = record_forward

    conversation = [
        ConversationItem(
            type="text",
            value=torch.randn(4, 32),
            role="user",
            meta={"attention_mask": torch.tensor([1, 1, 0, 0])},
        )
    ]
    llm.generate(conversation)  # prefill
    llm.generate(conversation)  # first decode

    assert seen[0].tolist() == [[1, 1, 0, 0]]
    assert seen[1].tolist() == [[1, 1, 0, 0, 1]]


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa", "veomni_sdpa"])
def test_qwen3_accelerated_packed_matches_separate_forwards_and_grads(monkeypatch, attn_impl):
    import copy

    import veomni.models.seed_omni.modules.qwen3.llm.accelerated.accelerated as qwen3_llm_acc

    monkeypatch.setattr(
        qwen3_llm_acc,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=False, sp_size=1, sp_group=None),
    )
    Qwen3LlmAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()
    torch.manual_seed(42)
    llm_packed = Qwen3LlmAcc(
        Qwen3LlmConfig(
            text_config=_tiny_qwen3_cfg(),
            ops_implementation=make_eager_ops_config(attn_implementation=attn_impl),
        )
    ).eval()
    llm_sep = copy.deepcopy(llm_packed).eval()

    s0_u = torch.randn(3, 32, requires_grad=True)
    s0_a = torch.randn(2, 32, requires_grad=True)
    s1_u = torch.randn(4, 32, requires_grad=True)

    s0_u_sep = s0_u.detach().clone().requires_grad_(True)
    s0_a_sep = s0_a.detach().clone().requires_grad_(True)
    s1_u_sep = s1_u.detach().clone().requires_grad_(True)

    conversations = [
        [
            ConversationItem(type="text", value=s0_u, role="user"),
            ConversationItem(type="text", value=s0_a, role="assistant"),
        ],
        [
            ConversationItem(type="text", value=s1_u, role="user"),
        ],
    ]
    pre_out = llm_packed.pre_forward("forward", conversation_list=conversations)
    fwd_out = llm_packed(**pre_out)
    out_conv = llm_packed.post_forward("forward", **fwd_out)["conversation_list"]

    assert [tuple(item.value.shape) for item in out_conv[0]] == [(3, 32), (2, 32)]
    assert [tuple(item.value.shape) for item in out_conv[1]] == [(4, 32)]

    conv_0 = [
        [
            ConversationItem(type="text", value=s0_u_sep, role="user"),
            ConversationItem(type="text", value=s0_a_sep, role="assistant"),
        ]
    ]
    pre_0 = llm_sep.pre_forward("forward", conversation_list=conv_0)
    out_0 = llm_sep.post_forward("forward", **llm_sep(**pre_0))["conversation_list"]

    conv_1 = [[ConversationItem(type="text", value=s1_u_sep, role="user")]]
    pre_1 = llm_sep.pre_forward("forward", conversation_list=conv_1)
    out_1 = llm_sep.post_forward("forward", **llm_sep(**pre_1))["conversation_list"]

    torch.testing.assert_close(out_conv[0][0].value, out_0[0][0].value, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_conv[0][1].value, out_0[0][1].value, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_conv[1][0].value, out_1[0][0].value, atol=1e-5, rtol=1e-5)

    w0 = torch.randn_like(out_conv[0][0].value)
    w1 = torch.randn_like(out_conv[0][1].value)
    w2 = torch.randn_like(out_conv[1][0].value)

    loss_packed = (
        (out_conv[0][0].value * w0).sum() + (out_conv[0][1].value * w1).sum() + (out_conv[1][0].value * w2).sum()
    )
    loss_packed.backward()

    loss_sep = (out_0[0][0].value * w0).sum() + (out_0[0][1].value * w1).sum() + (out_1[0][0].value * w2).sum()
    loss_sep.backward()

    torch.testing.assert_close(s0_u.grad, s0_u_sep.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s0_a.grad, s0_a_sep.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s1_u.grad, s1_u_sep.grad, atol=1e-5, rtol=1e-5)
    for (n1, p1), (_, p2) in zip(llm_packed.named_parameters(), llm_sep.named_parameters(), strict=True):
        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-5, msg=n1)

    # Verify pack_llm_conversations_for_forward + scatter_llm_hidden_states preserves attention_mask across calls.
    packed_embeds, mask, pids, _ = pack_llm_conversations_for_forward(out_conv, torch.device("cpu"))
    assert tuple(packed_embeds.shape) == (1, 9, 32)
    assert tuple(mask.shape) == (1, 9)
    assert tuple(pids.shape) == (1, 9)
    scatter_llm_hidden_states(out_conv, [torch.zeros(3, 32), torch.zeros(2, 32), torch.zeros(4, 32)])


def test_qwen3_text_encoder_generate_works_with_omitted_generation_kwargs():
    Qwen3TextEncoder = OMNI_MODEL_REGISTRY["qwen3_text_encoder"]()
    te = Qwen3TextEncoder(Qwen3TextEncoderConfig(vocab_size=32, hidden_size=16, tie_word_embeddings=True))
    te._tokenizer = SimpleNamespace(
        eos_token_id=0,
        convert_tokens_to_ids=lambda tok: 1,
        decode=lambda ids, skip_special_tokens=True: "decoded",
    )
    te._chat_template = Qwen3ChatTemplate(te._tokenizer)
    te._prompt_encoded = True

    conversation_list = [ConversationItem(type="output", value=torch.randn(1, 16), role="assistant")]
    out = te.generate(conversation_list=conversation_list)
    assert "conversation_list" in out
    assert out["conversation_list"][-1].value.shape == (1, 16)


def test_qwen3_train_and_infer_yaml_configs_load():
    from veomni.arguments.omni_arguments_types import (
        OmniArguments,
        OmniDataArguments,
        OmniInferArguments,
        OmniModelRuntimeArguments,
        build_omni_model_runtime_args,
    )

    cfg_dir = Path(__file__).resolve().parents[4] / "configs" / "seed_omni" / "Qwen" / "qwen3_0.6b" / "train"
    args = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=".",
            model_config={
                "modules": str(cfg_dir / "modules_train.yaml"),
                "train_graph": str(cfg_dir / "graph_train.yaml"),
                "infer_graph": str(cfg_dir / "graph_infer.yaml"),
            },
        ),
        data=OmniDataArguments(train_path="."),
        infer=OmniInferArguments(),
    )
    cfg = build_omni_model_runtime_args(args).to_hf_config()
    assert set(cfg._module_entries) == {"qwen3_text_encoder", "qwen3_llm"}
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "qwen3_text_encoder.encode" in endpoints and "qwen3_llm" in endpoints
    assert cfg.generation_graph["initial"] == "text_ar"


_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")
_GRAPH_EDGES = [
    {"from": "qwen3_text_encoder.encode", "to": "qwen3_llm"},
    {"from": "qwen3_llm", "to": "qwen3_text_encoder.decode"},
    {"from": "qwen3_text_encoder.decode", "to": "end"},
]
# Packed lengths 3 + 4 = 7 tokens (odd -> exercises SP=2 padding on both
# qwen3_text_encoder.encode and qwen3_llm.forward).
_ALIGN_CONVERSATIONS = [
    [
        (torch.tensor([5, 6], dtype=torch.long), torch.tensor([-100, -100], dtype=torch.long), "user"),
        (torch.tensor([7], dtype=torch.long), torch.tensor([7], dtype=torch.long), "assistant"),
    ],
    [
        (torch.tensor([11], dtype=torch.long), torch.tensor([-100], dtype=torch.long), "user"),
        (torch.tensor([12, 13, 14], dtype=torch.long), torch.tensor([12, 13, 14], dtype=torch.long), "assistant"),
    ],
]


def _build_align_conversation_list() -> list[list[ConversationItem]]:
    out: list[list[ConversationItem]] = []
    for sample in _ALIGN_CONVERSATIONS:
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


def _build_align_packed_features() -> list[dict[str, torch.Tensor]]:
    features: list[dict[str, torch.Tensor]] = []
    for sample in _ALIGN_CONVERSATIONS:
        ids = torch.cat([seg[0] for seg in sample], dim=0)
        labels = torch.cat([seg[1] for seg in sample], dim=0)
        features.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": labels,
            }
        )
    return features


def _monolithic_and_split_qwen3(tied: bool) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as HFQwen3Config

    from veomni.models.configuration_base import get_veomni_config_class
    from veomni.models.seed_omni.modules.qwen3.hf_layout import QWEN3_HF_LAYOUT
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3ForCausalLM

    Qwen3Config = get_veomni_config_class(HFQwen3Config)
    ops = make_eager_ops_config(attn_implementation="veomni_sdpa")
    hf_cfg = Qwen3Config(
        **_tiny_qwen3_cfg(),
        tie_word_embeddings=tied,
        ops_implementation=ops,
    )
    hf_cfg._attn_implementation = "veomni_sdpa"

    torch.manual_seed(2026)
    monolithic = Qwen3ForCausalLM(hf_cfg)

    Qwen3TextEncoderAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_text_encoder"]()
    Qwen3LlmAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3_llm"]()
    Qwen3LlmConfig = OMNI_CONFIG_REGISTRY["qwen3_llm"]()

    te_cfg = Qwen3TextEncoderConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        tie_word_embeddings=tied,
        ops_implementation=make_eager_ops_config(),
    )
    llm_cfg = Qwen3LlmConfig(
        text_config=_tiny_qwen3_cfg(),
        ops_implementation=make_eager_ops_config(attn_implementation="veomni_sdpa"),
    )
    text_encoder = Qwen3TextEncoderAcc(te_cfg)
    text_encoder._chat_template = _template()
    llm = Qwen3LlmAcc(llm_cfg)

    te_sd = {}
    llm_sd = {}
    for key, val in monolithic.state_dict().items():
        routed = QWEN3_HF_LAYOUT.route(key)
        if routed is None:
            continue
        mod_name, mod_key = routed
        if mod_name == "qwen3_text_encoder":
            if tied and mod_key == "lm_head.weight":
                continue
            te_sd[mod_key] = val.detach().clone()
        elif mod_name == "qwen3_llm":
            llm_sd[mod_key] = val.detach().clone()
    text_encoder.load_state_dict(te_sd, strict=True)
    llm.load_state_dict(llm_sd, strict=True)
    return monolithic, text_encoder, llm


def _run_qwen3_omni_graph_step(
    text_encoder: torch.nn.Module,
    llm: torch.nn.Module,
    conversation_list: list[list[ConversationItem]],
) -> torch.Tensor:
    from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
    from veomni.models.seed_omni.configuration_omni import OmniConfig
    from veomni.models.seed_omni.modeling_omni import OmniModel

    omni_cfg = OmniConfig(
        _module_entries={
            "qwen3_text_encoder": {"model_path": "qwen3_text_encoder"},
            "qwen3_llm": {"model_path": "qwen3_llm"},
        },
        training_graphs={"default": _GRAPH_EDGES},
        generation_graphs={},
    )
    omni = OmniModel(
        omni_cfg,
        {"qwen3_text_encoder": text_encoder, "qwen3_llm": llm},
    )
    out = omni.forward({"conversation_list": conversation_list}, node_runner=TrainNodeRunner())
    return out["loss"]


def _collect_omni_as_source_dict(
    text_encoder: torch.nn.Module,
    llm: torch.nn.Module,
    *,
    grads: bool,
) -> dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    from veomni.models.seed_omni.modules.qwen3.hf_layout import QWEN3_HF_LAYOUT

    def _full(t: torch.Tensor) -> torch.Tensor:
        return t.full_tensor().detach().clone() if isinstance(t, DTensor) else t.detach().clone()

    out: dict[str, torch.Tensor] = {}
    for mod_name, mod in (("qwen3_text_encoder", text_encoder), ("qwen3_llm", llm)):
        for name, param in mod.named_parameters():
            src_key = QWEN3_HF_LAYOUT.source_key(mod_name, name)
            assert src_key is not None, (mod_name, name)
            target = param.grad if grads else param
            assert target is not None, (mod_name, name)
            out[src_key] = _full(target)
    return out


def _qwen3_fsdp2_sp_align_worker(rank: int, rendezvous: str, out_dir: str) -> None:
    import json

    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    from veomni.data.data_collator import MainCollator
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state

    world = 2
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        sp1_ps = _init_parallel_state(
            dp_size=world,
            dp_shard_size=world,
            ulysses_size=1,
            dp_mode="fsdp2",
            device_type="cpu",
            name="sp1",
        )
        sp2_ps = _init_parallel_state(
            dp_size=1,
            dp_shard_size=1,
            ulysses_size=world,
            dp_mode="fsdp2",
            device_type="cpu",
            name="sp2",
        )
        _init_parallel_state(dp_size=world, ulysses_size=1, dp_mode="ddp", device_type="cpu", name="ref_sp1")

        reports: dict[str, dict] = {}
        for sp_size, ps_name, ps in ((1, "sp1", sp1_ps), (2, "sp2", sp2_ps)):
            for tied in (True, False):
                monolithic_ref, te, llm = _monolithic_and_split_qwen3(tied=tied)

                # Reference: packed forward + backward + AdamW step on the full 2-conversation batch.
                with use_parallel_state("ref_sp1"):
                    ref_batch = MainCollator()(_build_align_packed_features())
                    ref_out = monolithic_ref(
                        input_ids=ref_batch["input_ids"],
                        attention_mask=ref_batch["attention_mask"],
                        position_ids=ref_batch["position_ids"],
                        cu_seq_lens_q=ref_batch["cu_seq_lens_q"],
                        cu_seq_lens_k=ref_batch["cu_seq_lens_k"],
                        max_length_q=ref_batch["max_length_q"],
                        max_length_k=ref_batch["max_length_k"],
                        labels=ref_batch["labels"],
                        use_cache=False,
                    )
                    ref_loss = ref_out.loss.detach().clone()
                    ref_out.loss.backward()
                    ref_grads = {
                        name: param.grad.detach().clone()
                        for name, param in monolithic_ref.named_parameters()
                        if param.grad is not None
                    }
                    ref_opt = torch.optim.AdamW(monolithic_ref.parameters(), lr=1e-2, weight_decay=0.01, foreach=False)
                    ref_opt.step()
                    ref_stepped = {name: param.detach().clone() for name, param in monolithic_ref.named_parameters()}

                # Wrap SeedOmni modules with FSDP2 over ps.fsdp_mesh (dp_shard_sp).
                with use_parallel_state(ps_name):
                    fully_shard(te.embed_tokens, mesh=ps.fsdp_mesh, reshard_after_forward=True)
                    fully_shard(te, mesh=ps.fsdp_mesh)
                    for layer in llm.language_model.layers:
                        fully_shard(layer, mesh=ps.fsdp_mesh, reshard_after_forward=True)
                    fully_shard(llm, mesh=ps.fsdp_mesh)

                    all_convs = _build_align_conversation_list()
                    if sp_size == 1:
                        # dp_size=2, sp_size=1: each DP rank takes one conversation; their
                        # supervised token counts are [1, 3], so the token-weighted DP mean
                        # over fsdp_group matches the packed 2-sample reference.
                        local_convs = [all_convs[rank]]
                    else:
                        # dp_size=1, sp_size=2: both SP ranks receive the same packed 2-sample
                        # batch (total length 7 -> padded to 8 across SP=2).
                        local_convs = all_convs

                    loss = _run_qwen3_omni_graph_step(te, llm, local_convs)
                    loss.backward()

                    omni_grads = _collect_omni_as_source_dict(te, llm, grads=True)
                    te_opt = torch.optim.AdamW(te.parameters(), lr=1e-2, weight_decay=0.01, foreach=False)
                    llm_opt = torch.optim.AdamW(llm.parameters(), lr=1e-2, weight_decay=0.01, foreach=False)
                    te_opt.step()
                    llm_opt.step()
                    omni_stepped = _collect_omni_as_source_dict(te, llm, grads=False)

                loss_match = torch.allclose(loss.detach(), ref_loss, atol=1e-5, rtol=1e-5)
                grad_mismatches = [
                    key
                    for key, ref_g in ref_grads.items()
                    if not torch.allclose(omni_grads[key], ref_g, atol=1e-4, rtol=1e-4)
                ]
                step_mismatches = [
                    key
                    for key, ref_w in ref_stepped.items()
                    if not torch.allclose(omni_stepped[key], ref_w, atol=1e-4, rtol=1e-4)
                ]
                mode_key = f"sp{sp_size}_{'tied' if tied else 'untied'}"
                reports[mode_key] = {
                    "loss_match": loss_match,
                    "loss": float(loss.detach()),
                    "ref_loss": float(ref_loss),
                    "grad_mismatches": grad_mismatches,
                    "step_mismatches": step_mismatches,
                }
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_qwen3_omni_graph_matches_monolithic_qwen3_under_fsdp2_and_sp(tmp_path: Path):
    """Full Qwen3 SeedOmni graph (encode -> llm -> decode) under FSDP2 at SP=1 and
    SP=2 (with odd sequence length 7 requiring SP padding) matches monolithic
    Qwen3ForCausalLM on loss, every parameter gradient, and AdamW parameter updates."""
    import json

    import torch.multiprocessing as mp

    out_dir = tmp_path / "fsdp2_sp_align"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _qwen3_fsdp2_sp_align_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert set(reports) == {"sp1_tied", "sp1_untied", "sp2_tied", "sp2_untied"}
        for mode, report in reports.items():
            assert report["loss_match"], (rank, mode, report)
            assert not report["grad_mismatches"], (rank, mode, report)
            assert not report["step_mismatches"], (rank, mode, report)
