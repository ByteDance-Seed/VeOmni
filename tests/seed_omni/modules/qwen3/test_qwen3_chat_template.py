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
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import (
    Qwen3ChatTemplate,
    Qwen3VLChatTemplate,
)
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


def test_qwen3_text_preprocessor_from_pretrained_applies_enable_image_override(tmp_path, monkeypatch):
    import veomni.models.seed_omni.modules.qwen3.text_encoder.processing as qwen3_text_processing

    Qwen3TextEncoderConfig(enable_image=False).save_pretrained(str(tmp_path))
    monkeypatch.setattr(qwen3_text_processing, "build_tokenizer", lambda module_path: FakeTokenizer())

    default_pre = Qwen3TextEncoderPreprocessor.from_pretrained(str(tmp_path))
    assert isinstance(default_pre._chat_template, Qwen3ChatTemplate)
    assert not isinstance(default_pre._chat_template, Qwen3VLChatTemplate)

    override_pre = Qwen3TextEncoderPreprocessor.from_pretrained(str(tmp_path), config_overrides={"enable_image": True})
    assert isinstance(override_pre._chat_template, Qwen3VLChatTemplate)


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
