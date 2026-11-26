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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core module unit tests and end-to-end training-graph smoke tests for DeepSeek-V4.1-Flash."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import DeepseekV41LlmAccelerated
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig, shrink_layer_schedule
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
    DeepseekV41Linear,
    DeepseekV41QuantizedExperts,
    dense_sparse_attn,
    make_grouped_linear,
    make_linear,
)
from veomni.models.seed_omni.modules.deepseek_v41.mtp.accelerated import DeepseekV41MtpAccelerated
from veomni.models.seed_omni.modules.deepseek_v41.mtp.configuration import DeepseekV41MtpConfig
from veomni.models.seed_omni.modules.deepseek_v41.ngram.accelerated.accelerated import DeepseekV41NgramAccelerated
from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import DeepseekV41Ngram, pack_ngram_inputs
from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.accelerated.accelerated import (
    DeepseekV41TextEncoderAccelerated,
)
from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.chat_template import DeepseekV41ChatTemplate
from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.configuration import DeepseekV41TextEncoderConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.accelerated.accelerated import (
    DeepseekV41VisionEncoderAccelerated,
)
from veomni.models.seed_omni.modules.deepseek_v41.vision.configuration import DeepseekV41VisionEncoderConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.processing import image_token_types
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from veomni.ops import VeomniOp


CONFIG_ROOT = Path("configs/seed_omni/DeepSeek/deepseek_v41_flash")
TRAIN_GRAPH_PATH = CONFIG_ROOT / "graph_train.yaml"
TRAIN_GRAPH_NO_NGRAM_MTP_PATH = CONFIG_ROOT / "deepgemm/graph_no_ngram_mtp.yaml"

HIDDEN = 64
VOCAB = 64
PAD = 3
N_LAYERS = 2
ENGRAM_HEADS = 2
ENGRAM_HEAD_DIM = 8
MAX_NGRAM = 4
HASH_VOCAB = 64
TABLE_ROWS = 1024

VISION = "deepseek_v41_vision"
NGRAM = "deepseek_v41_ngram"
TEXT = "deepseek_v41_text_encoder"
LLM = "deepseek_v41_llm"
MTP = "deepseek_v41_mtp"

_DUMMY_GENERATION = {
    "infer_und": {
        "initial": "step",
        "states": {
            "step": {
                "body": [{"from": TEXT, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }
}


def eager_sparse_mqa_target_fwd(q, kv, topk_idxs, lse, sm_scale=None):
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    kv_len = kv.shape[1]
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    safe = topk_idxs.clamp(0, max(kv_len - 1, 0)).long()
    gathered = kv[:, None, :, :].expand(-1, q.shape[1], -1, -1)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, kv.shape[-1]))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * sm_scale
    contrib = torch.exp2(scores * math.log2(math.e) - lse.float().unsqueeze(-1))
    contrib = contrib.masked_fill(~valid.unsqueeze(2), 0.0)
    return contrib.sum(dim=2)


def _patch_cpu_attention(monkeypatch) -> None:
    monkeypatch.setattr(
        "veomni.models.seed_omni.modules.deepseek_v41.llm.modeling.sparse_attn_tilelang",
        dense_sparse_attn,
    )
    monkeypatch.setattr(
        "veomni.models.seed_omni.modules.deepseek_v41.llm.modeling.sparse_mqa_target_fwd",
        eager_sparse_mqa_target_fwd,
    )
    monkeypatch.setattr(
        "veomni.models.seed_omni.modules.deepseek_v41.vision.modeling.USE_FLASH_ATTN_VARLEN",
        False,
    )


def tiny_llm_config(**overrides) -> DeepseekV41LlmConfig:
    schedule = shrink_layer_schedule(N_LAYERS, n_draft=0)
    schedule.pop("kept_layer_ids", None)
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=48,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=6,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=4,
        rms_norm_eps=1e-20,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        max_position_embeddings=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=1,
        engram_max_ngram_size=MAX_NGRAM,
        engram_n_heads=ENGRAM_HEADS,
        engram_head_dim=ENGRAM_HEAD_DIM,
        dspark_block_size=0,
        dspark_noise_token_id=31,
        dsa_indexer_loss=True,
        weight_quantization=None,
        image_token_id=4,
        ops_implementation={
            "mhc_implementation": "eager",
            "moe_implementation": "eager",
            "rms_norm_implementation": "eager",
            "gated_mlp_implementation": "eager",
            "linear_implementation": "eager",
            "kv_round_implementation": "eager",
            "dsa_indexer_implementation": "eager",
        },
        rope_scaling={
            "rope_type": "yarn",
            "factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 256,
        },
        **schedule,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def tiny_ngram_config(layer_ids) -> DeepseekV41NgramConfig:
    return DeepseekV41NgramConfig(
        layer_ids=list(layer_ids),
        num_embeddings=[TABLE_ROWS] * len(layer_ids),
        max_ngram_size=MAX_NGRAM,
        n_heads=ENGRAM_HEADS,
        head_dim=ENGRAM_HEAD_DIM,
        vocab_size=VOCAB,
        hash_vocab_size=HASH_VOCAB,
        compressed_vocab_size=32,
        pad_token_id=PAD,
        freeze=True,
    )


def tiny_vision_config() -> DeepseekV41VisionEncoderConfig:
    return DeepseekV41VisionEncoderConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        patch_size=4,
        downsample_ratio=1,
        max_image_tokens=16,
        min_pixels=16,
        out_hidden_size=HIDDEN,
        ops_implementation={"attn_implementation": "sdpa", "rms_norm_implementation": "eager"},
    )


def tiny_text_config() -> DeepseekV41TextEncoderConfig:
    return DeepseekV41TextEncoderConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        tie_word_embeddings=False,
        dspark_block_size=0,
        dspark_noise_token_id=31,
        pad_token_id=PAD,
        image_token_id=4,
        ops_implementation={"cross_entropy_loss_implementation": "chunk_loss"},
    )


def build_toy_modules(
    *,
    with_ngram: bool = True,
    dspark: bool = False,
    n_trainable_layers: int | None = None,
    freeze_mtp: bool = False,
):
    overrides: dict = {}
    if dspark:
        schedule = shrink_layer_schedule(N_LAYERS, n_draft=1)
        schedule.pop("kept_layer_ids", None)
        overrides.update(schedule)
        overrides.update(
            dspark_block_size=5,
            dspark_n_routed_experts=4,
            dspark_num_experts_per_tok=2,
            emit_draft_hiddens=True,
        )
    if not with_ngram:
        overrides["engram_layer_ids"] = []
    if n_trainable_layers is not None:
        overrides["n_trainable_layers"] = n_trainable_layers
    llm_config = tiny_llm_config(**overrides) if overrides else tiny_llm_config()
    torch.manual_seed(0)

    text = DeepseekV41TextEncoderAccelerated(tiny_text_config())
    text._chat_template = DeepseekV41ChatTemplate
    text.loss_function = ForCausalLMLoss

    llm = DeepseekV41LlmAccelerated(llm_config)
    llm.freeze_model()
    modules: dict = {
        VISION: DeepseekV41VisionEncoderAccelerated(tiny_vision_config()),
        TEXT: text,
        LLM: llm,
    }
    if with_ngram:
        ngram = DeepseekV41NgramAccelerated(tiny_ngram_config(llm_config.engram_layer_ids))
        ngram.token_map.copy_(torch.arange(VOCAB) % ngram.config.compressed_vocab_size)
        ngram.freeze_model()
        modules[NGRAM] = ngram
    if dspark:
        mtp_cfg = DeepseekV41MtpConfig(
            **{k: v for k, v in llm_config.to_dict().items() if k != "model_type"},
            freeze=freeze_mtp,
        )
        mtp = DeepseekV41MtpAccelerated(mtp_cfg)
        mtp.freeze_model()
        modules[MTP] = mtp
    return modules


def build_toy_omni(modules) -> OmniModel:
    if MTP in modules:
        edges = yaml.safe_load(TRAIN_GRAPH_PATH.read_text())
    elif NGRAM not in modules:
        edges = yaml.safe_load(TRAIN_GRAPH_NO_NGRAM_MTP_PATH.read_text())
    else:
        edges = [e for e in yaml.safe_load(TRAIN_GRAPH_PATH.read_text()) if e["from"] != MTP and e["to"] != MTP]
        edges.insert(-1, {"from": LLM, "to": f"{TEXT}.decode"})
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in modules},
        training_graphs={"default": edges},
        generation_graphs=_DUMMY_GENERATION,
        infer_type="infer_und",
    )
    return OmniModel(config, modules)


def toy_conversation() -> list[list[ConversationItem]]:
    user_ids = torch.tensor([7, 8, 9, 10], dtype=torch.long)
    asst_ids = torch.tensor([11, 12, 13, 14], dtype=torch.long)
    user = ConversationItem(
        type="text",
        value=user_ids,
        role="user",
        meta={"input_ids": user_ids.clone(), "labels": torch.full_like(user_ids, -100)},
    )
    assistant = ConversationItem(
        type="text",
        value=asst_ids,
        role="assistant",
        meta={"input_ids": asst_ids.clone(), "labels": asst_ids.clone()},
    )
    dummy_img = ConversationItem(
        type="image",
        value=torch.zeros(1, 3, 4, 4),
        role="user",
        is_dummy=True,
        meta={"_omni_vit_grid": [1, 1], "_omni_token_types": image_token_types(1, 1)},
    )
    return [[user, assistant, dummy_img]]


class TestEndToEndTrainGraph:
    def test_five_module_graph_runs_forward_backward_and_step(self, monkeypatch):
        """Full 5-module DAG (vision + ngram + text + llm + mtp): CE + indexer KL + MTP loss."""
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(with_ngram=True, dspark=True)
        model = build_toy_omni(modules)
        model.train()

        ngram_params = list(modules[NGRAM].parameters())
        assert ngram_params and all(not p.requires_grad for p in ngram_params)
        mtp_params = [p for n, p in modules[MTP].named_parameters() if n.startswith("model.mtp.")]
        assert mtp_params and all(p.requires_grad for p in mtp_params)

        trainable = [p for p in model.parameters() if p.requires_grad]
        probe = next(p for n, p in model.named_parameters() if n.startswith(f"{LLM}.") and p.requires_grad)
        before = probe.detach().clone()
        opt = torch.optim.AdamW(trainable, lr=1e-3)

        out = model({"conversation_list": toy_conversation()})
        loss = out["loss"]
        assert torch.isfinite(loss) and loss.requires_grad
        assert {f"{TEXT}.decode.ce_loss", f"{TEXT}.decode.mtp_loss", f"{LLM}.forward.indexer_kl_loss"} <= set(
            out["losses"]
        )
        torch.testing.assert_close(loss, sum(out["losses"].values()))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None for p in ngram_params)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mtp_params)
        opt.step()
        assert not torch.equal(probe.detach(), before)

    def test_three_module_no_ngram_mtp_graph_trains(self, monkeypatch):
        """3-module DAG (`deepgemm/graph_no_ngram_mtp.yaml`): vision + text_encoder + llm."""
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(with_ngram=False, dspark=False)
        model = build_toy_omni(modules)
        model.train()

        out = model({"conversation_list": toy_conversation()})
        assert torch.isfinite(out["loss"])
        assert set(out["losses"]) == {f"{TEXT}.decode.ce_loss", f"{LLM}.forward.indexer_kl_loss"}
        out["loss"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in modules[LLM].parameters())

    def test_freeze_n_trainable_layers_and_frozen_mtp_skip(self, monkeypatch):
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(with_ngram=True, dspark=True, n_trainable_layers=1, freeze_mtp=True)
        called = []
        orig = modules[MTP].model.forward_train
        modules[MTP].model.forward_train = lambda *a, **kw: (called.append(True), orig(*a, **kw))[1]

        model = build_toy_omni(modules)
        model.train()
        out = model({"conversation_list": toy_conversation()})
        out["loss"].backward()

        assert called == []
        frozen = [p for n, p in modules[LLM].named_parameters() if n.startswith("model.layers.1.")]
        trained = [p for n, p in modules[LLM].named_parameters() if n.startswith("model.layers.0.")]
        assert frozen and all(p.grad is None for p in frozen)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained)

    def test_parallel_plans_resolve_on_llm_mtp_and_ngram(self):
        modules = build_toy_modules(with_ngram=True, dspark=True)
        llm_names = {n for n, _ in modules[LLM].named_parameters()}
        for pattern in modules[LLM].get_parallel_plan().extra_parallel_plan["ep"]:
            if pattern.startswith("model.mtp."):
                continue
            assert pattern.replace("*", "0") in llm_names
        mtp_names = {n for n, _ in modules[MTP].named_parameters()}
        assert "model.mtp.0.mlp.experts.gate_up_proj" in mtp_names
        ngram_names = {n for n, _ in modules[NGRAM].named_parameters()}
        for pattern in modules[NGRAM].get_parallel_plan().extra_parallel_plan["emb"]:
            assert pattern in ngram_names


class TestLinearFactoryAndOpsBinding:
    def test_plain_training_linears_and_quantized_fail_fast(self):
        lin = make_linear(16, 8, bias=False, quant_dtype=None)
        assert isinstance(lin, torch.nn.Linear) and not isinstance(lin, DeepseekV41Linear)
        grouped = make_grouped_linear(16, 8, n_groups=2, quant_dtype=None)
        assert isinstance(grouped, torch.nn.Linear)

        q_lin = DeepseekV41Linear(32, 32, quant_dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="pre-quantized"):
            q_lin(torch.randn(2, 32))

        q_cfg = tiny_llm_config()
        experts = DeepseekV41QuantizedExperts(q_cfg, quant_dtype=torch.int8)
        with pytest.raises(ValueError, match="pre-quantized"):
            experts(torch.randn(2, HIDDEN), torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2))

    def test_module_level_ops_implementation_binds_llm_and_vision_ops(self):
        llm_cfg = tiny_llm_config(
            ops_implementation={
                "mhc_implementation": "eager",
                "moe_implementation": "eager",
                "rms_norm_implementation": "eager",
                "gated_mlp_implementation": "eager",
                "linear_implementation": "eager",
                "kv_round_implementation": "eager",
            }
        )
        llm = DeepseekV41LlmAccelerated(llm_cfg)
        layer0 = llm.model.layers[0]
        assert isinstance(layer0.veomni_mhc_post, VeomniOp) and layer0.veomni_mhc_post.impl == "eager"
        assert isinstance(layer0.mlp.experts.veomni_moe, VeomniOp) and layer0.mlp.experts.veomni_moe.impl == "eager"
        assert isinstance(layer0.mlp.shared_experts.veomni_gated_mlp, VeomniOp)
        assert isinstance(layer0.self_attn.veomni_wq_a, VeomniOp) and layer0.self_attn.veomni_wq_a.impl == "eager"
        assert isinstance(layer0.input_layernorm.veomni_rms_norm, VeomniOp)

        vision = DeepseekV41VisionEncoderAccelerated(tiny_vision_config())
        v_block0 = vision.vision.blocks[0]
        assert isinstance(v_block0.attn.veomni_attn, VeomniOp) and v_block0.attn.veomni_attn.impl == "veomni_sdpa"
        assert isinstance(v_block0.norm1.veomni_rms_norm, VeomniOp)


class TestNgramPackingAndCP:
    def test_sharded_image_span_keeps_full_length(self):
        text_a = ConversationItem(
            type="text", value=torch.tensor([5, 6]), role="user", meta={"input_ids": torch.tensor([5, 6])}
        )
        text_b = ConversationItem(
            type="text", value=torch.tensor([7]), role="user", meta={"input_ids": torch.tensor([7])}
        )
        sharded = ConversationItem(type="image", value=torch.randn(3, HIDDEN), role="user", meta={_OMNI_TOKEN_LEN: 8})
        full = ConversationItem(type="image", value=torch.randn(8, HIDDEN), role="user")

        row_sharded = pack_ngram_inputs([[text_a, sharded, text_b]], torch.device("cpu"), PAD)
        row_full = pack_ngram_inputs([[text_a, full, text_b]], torch.device("cpu"), PAD)
        for got, want in zip(row_sharded, row_full, strict=True):
            assert torch.equal(got, want)
        assert row_sharded[0].shape == (1, 11)

    def test_cp_tail_rank_beyond_valid_len_returns_exact_local_len_zeros(self):
        model = DeepseekV41Ngram(tiny_ngram_config([0]))
        input_ids = torch.tensor([[5, 6]], dtype=torch.long)
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # cp_size=4, valid_len=2 -> full_len=4, local_len=1; rank 3 covers [3, 4) > valid_len=2.
        out = model(input_ids=input_ids, token_mask=token_mask, cp_size=4, cp_rank=3)
        assert len(out["ngram_embeds"]) == 1
        assert out["ngram_embeds"][0].shape == (1, 1, ENGRAM_HEADS * (MAX_NGRAM - 1), ENGRAM_HEAD_DIM)
        assert torch.count_nonzero(out["ngram_embeds"][0]) == 0


class TestReviewRegressions:
    def test_vision_freeze_aligner_independent_and_forward_pre_kwargs_and_dummy_filter(self):
        cfg = tiny_vision_config()
        cfg.freeze = False
        cfg.freeze_aligner = True
        cfg.enable_vision = True
        vis = DeepseekV41VisionEncoderAccelerated(cfg)
        vis.freeze_model()
        assert all(p.requires_grad for p in vis.vision.parameters())
        assert not any(p.requires_grad for p in vis.aligner.parameters())
        assert (
            not vis.image_start.requires_grad
            and not vis.image_end.requires_grad
            and not vis.image_newline.requires_grad
        )

        from veomni.models.seed_omni.modules.deepseek_v41.vision.processing import _OMNI_TOKEN_TYPES, _OMNI_VIT_GRID

        grid = [2, 2]
        patches = torch.zeros(4, 3, cfg.patch_size, cfg.patch_size)
        types = image_token_types(1, 1)
        real_item = ConversationItem(
            type="image",
            value=patches,
            role="user",
            is_dummy=False,
            meta={_OMNI_VIT_GRID: grid, _OMNI_TOKEN_TYPES: types},
        )
        dummy_item = ConversationItem(
            type="image",
            value=patches,
            role="user",
            is_dummy=True,
            meta={_OMNI_VIT_GRID: grid, _OMNI_TOKEN_TYPES: types},
        )
        # Accepts extra multisource keys (ds_idx/source_name) and skips dummy items when real images exist.
        out = vis.forward_pre(
            conversation_list=[[real_item], [dummy_item]],
            ds_idx=torch.tensor([0, 1]),
            source_name="mix",
        )
        assert out["grids"] == [grid]
        assert len(vis._visual_output_slots) == 1 and vis._visual_output_slots[0].item is real_item

    def test_reentrant_gradient_checkpointing_rejected_on_llm_and_mtp(self):
        llm = DeepseekV41LlmAccelerated(tiny_llm_config())
        with pytest.raises(ValueError, match="reentrant"):
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        mtp_schedule = shrink_layer_schedule(N_LAYERS, n_draft=1)
        mtp_schedule.pop("kept_layer_ids", None)
        mtp_cfg = DeepseekV41MtpConfig(
            **(
                tiny_llm_config(
                    dspark_block_size=2,
                    dspark_markov_rank=16,
                    dspark_n_routed_experts=4,
                    dspark_num_experts_per_tok=2,
                    **mtp_schedule,
                ).to_dict()
            )
        )
        mtp = DeepseekV41MtpAccelerated(mtp_cfg)
        with pytest.raises(ValueError, match="reentrant"):
            mtp.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        mtp.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def test_indexer_ops_selection_candidate_mask_zero_width_and_ratio_validation(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            DeepseekV41Attention,
            DeepseekV41Indexer,
            SharedAttentionState,
            build_compression_plans,
        )

        # Mapping-form OpsSelection normalizes to its string name.
        cfg = tiny_llm_config(
            ops_implementation={
                "mhc_implementation": "eager",
                "moe_implementation": "eager",
                "rms_norm_implementation": "eager",
                "gated_mlp_implementation": "eager",
                "linear_implementation": "eager",
                "kv_round_implementation": "eager",
                "dsa_indexer_implementation": {"name": "tilelang"},
            }
        )
        idx = DeepseekV41Indexer(cfg, layer_idx=1)
        assert idx.dsa_indexer_implementation == "tilelang"

        # Rejected non-candidate slots become -1 / -inf when fewer than topk candidates survive.
        cfg_cand = tiny_llm_config(
            num_hidden_layers=3,
            compress_ratios=[2, 1, 1],
            kv_source_layer_ids=[0, 1],
            index_source_layer_ids=[0, 1, 2],
            candidate_source_layer_id=1,
            engram_layer_ids=[0],
            dspark_target_layer_ids=[0, 1, 2],
            index_topk=4,
            candidate_block_size=1,
            candidate_topk_blocks=1,
        )
        idx_cand = DeepseekV41Indexer(cfg_cand, layer_idx=2)
        assert idx_cand.uses_candidates is True
        seq_len = 8
        pos = torch.arange(seq_len).unsqueeze(0)
        plan = build_compression_plans(cfg_cand, pos, [(0, seq_len)])[cfg_cand.compress_ratios[2]]
        shared = SharedAttentionState(candidates=torch.zeros(1, seq_len, plan.compressed_len, dtype=torch.bool))
        shared.candidates[:, :, 0] = True
        hidden = torch.randn(1, seq_len, cfg_cand.hidden_size)
        q_res = torch.randn(1, seq_len, cfg_cand.q_lora_rank)
        index_k = torch.randn(1, plan.compressed_len, cfg_cand.index_head_dim)
        cos = torch.ones(1, seq_len, cfg_cand.qk_rope_head_dim)
        sin = torch.zeros(1, seq_len, cfg_cand.qk_rope_head_dim)
        topk_idxs, student = idx_cand(hidden, q_res, index_k, plan, shared, cos, sin, return_scores=True)
        # Only slot 0 is a valid candidate; all other slots in topk must be -1 / -inf.
        assert ((topk_idxs == -1) | (topk_idxs == 0)).all()
        assert torch.isneginf(student[topk_idxs == -1]).all()

        # _indexer_loss handles width == 0 without negative-zero slicing.
        attn = DeepseekV41Attention(cfg_cand, layer_idx=0)
        q = torch.randn(1, 2, cfg_cand.num_attention_heads, cfg_cand.head_dim)
        kv = torch.randn(1, 2, cfg_cand.head_dim)
        lse = torch.zeros(1, 2, cfg_cand.num_attention_heads)
        terms = attn._indexer_loss(q, kv, torch.zeros(1, 2, 2, dtype=torch.int32), lse, torch.empty(1, 2, 0))
        assert terms.kl.item() == 0.0 and terms.uniform.item() == 0.0

        # Mismatched compress_ratio between consumer and KV source fails fast.
        with pytest.raises(ValueError, match="compress_ratio"):
            tiny_llm_config(
                num_hidden_layers=3,
                compress_ratios=[2, 4, 4],
                kv_source_layer_ids=[0],
                index_source_layer_ids=[0],
                candidate_source_layer_id=0,
                engram_layer_ids=[0],
                dspark_target_layer_ids=[0, 1, 2],
            )

    def test_ngram_prime_overflow_empty_dequant_fused_lookup_and_converter_guards(self):
        from veomni.models.seed_omni.modules.deepseek_v41.ngram.modeling import (
            _DEFAULT_ALL_TO_ALL_APPLY,
            AllToAllEmbedding,
            NgramTableDequant,
            NgramTablePadding,
            dequantize_rows,
        )

        # Bound method equality keeps fused lookup active when not monkeypatched.
        assert not (AllToAllEmbedding.apply != _DEFAULT_ALL_TO_ALL_APPLY)

        # Empty FP8 rows dequantize to empty BF16 without dividing by zero.
        empty_rows = torch.empty((0, 128), dtype=torch.float8_e4m3fn)
        empty_scales = torch.empty((0, 1), dtype=torch.float32)
        out = dequantize_rows(empty_rows, empty_scales)
        assert out.shape == (0, 128) and out.dtype == torch.bfloat16

        # Prime bucket sum exceeding logical num_embeddings fails fast even if floor & padded rows pass.
        small_cfg = DeepseekV41NgramConfig(
            layer_ids=[0],
            num_embeddings=[384],
            make_table_rows_divisible_by=1024,
            max_ngram_size=MAX_NGRAM,
            n_heads=ENGRAM_HEADS,
            head_dim=ENGRAM_HEAD_DIM,
            vocab_size=VOCAB,
            hash_vocab_size=HASH_VOCAB,
            compressed_vocab_size=VOCAB,
            pad_token_id=PAD,
        )
        assert small_cfg.padded_num_embeddings == [1024]
        with pytest.raises(ValueError, match="prime buckets"):
            DeepseekV41Ngram(small_cfg)

        # Truncated checkpoint (< logical_rows) is rejected.
        pad_conv = NgramTablePadding(padded_num_embeddings=[64], num_embeddings=[40])
        with pytest.raises(ValueError, match="outside the expected"):
            pad_conv.convert("tables.0.weight", torch.zeros(32, 16, dtype=torch.bfloat16))

        # Unpaired FP8 table weights are flagged by can_handle_tensor; BF16 weights are not.
        deq_conv = NgramTableDequant(padded_num_embeddings=[64], num_embeddings=[40])
        assert deq_conv.can_handle_tensor("tables.0.weight", torch.zeros(40, 128, dtype=torch.float8_e4m3fn))
        assert not deq_conv.can_handle_tensor("tables.0.weight", torch.zeros(40, 128, dtype=torch.bfloat16))

    def test_build_configs_enables_emit_draft_hiddens_when_mtp_present(self):
        from veomni.models.seed_omni.modules.deepseek_v41.convert_model import _build_configs

        mtp_schedule = shrink_layer_schedule(N_LAYERS, n_draft=1)
        mtp_schedule.pop("kept_layer_ids", None)
        hf_cfg = {
            "text_config": tiny_llm_config(dspark_block_size=2, **mtp_schedule).to_dict()
            | {
                "engram_num_embeddings": [256],
                "engram_vocab_size": 32,
                "engram_compressed_vocab_size": 32,
                "engram_pad_token_id": 0,
            },
            "vision_config": tiny_vision_config().to_dict(),
        }
        hf_cfg["text_config"].pop("emit_draft_hiddens", None)
        _, _, _, llm_cfg, _ = _build_configs(hf_cfg)
        assert llm_cfg.emit_draft_hiddens is True
