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

"""Step 5 toy OmniModel: four random-init modules, one training-graph step.

The released backbone's prefill attention is TileLang on SM90, so a CPU (and
any non-SM90) box cannot run the production kernel. The dense gather that
decode already uses is the same arithmetic, so the toy step swaps it in and
keeps the rest of the graph — vision dummy, frozen n-gram tables, indexer KL
folded as ``_loss``, CE head, AdamW — identical to a real train step.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import DeepseekV41LlmAccelerated
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig, shrink_layer_schedule
from veomni.models.seed_omni.modules.deepseek_v41.llm.decode import dense_sparse_attn
from veomni.models.seed_omni.modules.deepseek_v41.ngram.accelerated.accelerated import DeepseekV41NgramAccelerated
from veomni.models.seed_omni.modules.deepseek_v41.ngram.configuration import DeepseekV41NgramConfig
from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.accelerated.accelerated import (
    DeepseekV41TextEncoderAccelerated,
)
from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.chat_template import DeepseekV41ChatTemplate
from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.configuration import DeepseekV41TextEncoderConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.accelerated.accelerated import (
    DeepseekV41VisionEncoderAccelerated,
)
from veomni.models.seed_omni.modules.deepseek_v41.vision.configuration import DeepseekV41VisionEncoderConfig
from veomni.models.seed_omni.modules.deepseek_v41.vision.processing import _SOURCE, image_token_types
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.ops.kernels.cross_entropy import ForCausalLMLoss
from veomni.utils.constants import IGNORE_INDEX


CONFIG_ROOT = Path("configs/seed_omni/DeepSeek/deepseek_v41_flash")
TRAIN_GRAPH_PATH = CONFIG_ROOT / "train/graph_train.yaml"

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
    """CPU stand-in for the TileLang teacher: head-summed ``exp2(score - lse)``."""
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
    """Use the dense gather for prefill so the toy step does not need TileLang/SM90.

    Production training calls ``sparse_attn_tilelang``. Decode already has a
    gather-and-softmax that matches it, which is what this smoke needs: a
    backward through the same graph, not a kernel check.
    """
    monkeypatch.setattr(
        "veomni.models.seed_omni.modules.deepseek_v41.llm.attention.sparse_attn_tilelang",
        dense_sparse_attn,
    )
    monkeypatch.setattr(
        "veomni.models.seed_omni.modules.deepseek_v41.llm.attention.sparse_mqa_target_fwd",
        eager_sparse_mqa_target_fwd,
    )
    # flash_attn is importable on this box even when the tensors stay on CPU.
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
    )


def build_toy_modules(
    *,
    n_trainable_layers=None,
    dspark: bool = False,
    n_layers: int = N_LAYERS,
    trainable_layer_ids=None,
    freeze_mtp=None,
):
    overrides: dict = {}
    n_draft = 1 if dspark else 0
    if dspark or n_layers != N_LAYERS:
        schedule = shrink_layer_schedule(n_layers, n_draft=n_draft)
        schedule.pop("kept_layer_ids", None)
        overrides.update(schedule)
    if dspark:
        overrides.update(
            dspark_block_size=5,
            dspark_n_routed_experts=4,
            dspark_num_experts_per_tok=2,
        )
    if n_trainable_layers is not None:
        overrides["n_trainable_layers"] = n_trainable_layers
    if trainable_layer_ids is not None:
        overrides["trainable_layer_ids"] = list(trainable_layer_ids)
    if freeze_mtp is not None:
        overrides["freeze_mtp"] = freeze_mtp
    llm_config = tiny_llm_config(**overrides) if overrides else tiny_llm_config()
    torch.manual_seed(0)
    ngram = DeepseekV41NgramAccelerated(tiny_ngram_config(llm_config.engram_layer_ids))
    ngram.token_map.copy_(torch.arange(VOCAB) % ngram.config.compressed_vocab_size)
    ngram.freeze_model()

    text = DeepseekV41TextEncoderAccelerated(tiny_text_config())
    text._chat_template = DeepseekV41ChatTemplate
    # HF's default ``ForCausalLMLoss`` requires materialized logits. Training
    # decode takes the fused linear+CE contract (``logits=None`` + hidden/weights).
    text.loss_function = ForCausalLMLoss

    llm = DeepseekV41LlmAccelerated(llm_config)
    llm.freeze_model()
    return {
        VISION: DeepseekV41VisionEncoderAccelerated(tiny_vision_config()),
        NGRAM: ngram,
        TEXT: text,
        LLM: llm,
    }


def build_toy_omni(modules) -> OmniModel:
    edges = yaml.safe_load(TRAIN_GRAPH_PATH.read_text())
    if isinstance(edges, dict):
        edges = edges["training_graph"]
    config = OmniConfig(
        modules={name: {"subfolder": name} for name in modules},
        training_graph=edges,
        generation_graphs=_DUMMY_GENERATION,
        infer_type="infer_und",
    )
    return OmniModel(config, modules)


def dummy_image_item() -> ConversationItem:
    return ConversationItem(
        type="image",
        value=torch.zeros(1, 3, 4, 4),
        role="dummy",
        source=_SOURCE,
        meta={
            "_omni_vit_grid": [1, 1],
            "_omni_token_types": image_token_types(1, 1),
        },
    )


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
    return [[user, assistant, dummy_image_item()]]


class TestToyTrainStep:
    def test_four_modules_run_backward_and_adamw(self, monkeypatch):
        """Random-init graph: CE + indexer KL, frozen n-gram, one AdamW step."""
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules()
        model = build_toy_omni(modules)
        model.train()

        ngram_params = list(modules[NGRAM].parameters())
        assert ngram_params and all(not p.requires_grad for p in ngram_params)

        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable
        probe = next(p for n, p in model.named_parameters() if n.startswith(f"{LLM}.") and p.requires_grad)
        before = probe.detach().clone()
        opt = torch.optim.AdamW(trainable, lr=1e-3)

        out = model({"conversation_list": toy_conversation()})
        loss = out["loss"]
        losses = out["losses"]
        assert torch.isfinite(loss)
        assert loss.requires_grad
        assert f"{TEXT}.decode" in losses
        assert f"{LLM}.forward" in losses, "indexer KL must fold in as `_loss` or the indexer has no gradient"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for param in ngram_params:
            assert param.grad is None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
        opt.step()
        assert not torch.equal(probe.detach(), before)

    def test_parallel_plans_still_resolve_on_the_toy_tree(self):
        """EP / emb plans must match the shrunk module, not only the 40-layer one."""
        modules = build_toy_modules()
        llm_names = {name for name, _ in modules[LLM].named_parameters()}
        for pattern in modules[LLM].get_parallel_plan().extra_parallel_plan["ep"]:
            mtp_on = any(name.startswith("model.mtp.") for name in llm_names)
            if pattern.startswith("model.mtp.") and not mtp_on:
                continue
            concrete = pattern.replace("*", "0")
            assert concrete in llm_names, f"{pattern} matches nothing on the toy backbone"

        ngram_names = {name for name, _ in modules[NGRAM].named_parameters()}
        for pattern in modules[NGRAM].get_parallel_plan().extra_parallel_plan["emb"]:
            assert pattern in ngram_names, f"{pattern} matches nothing on the toy n-gram tables"


class TestFirstNLayersFreeze:
    def test_unset_leaves_the_stack_trainable(self):
        llm = DeepseekV41LlmAccelerated(tiny_llm_config())
        llm.freeze_model()
        assert all(p.requires_grad for p in llm.parameters())

    def test_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="n_trainable_layers"):
            tiny_llm_config(n_trainable_layers=N_LAYERS + 1)

    def test_later_layers_and_their_engram_freeze(self):
        llm = DeepseekV41LlmAccelerated(tiny_llm_config(n_trainable_layers=1))
        llm.freeze_model()
        by_layer = {0: [], 1: []}
        for name, param in llm.named_parameters():
            if name.startswith("model.layers.0."):
                by_layer[0].append(param.requires_grad)
            elif name.startswith("model.layers.1."):
                by_layer[1].append(param.requires_grad)
        assert by_layer[0] and all(by_layer[0])
        assert by_layer[1] and not any(by_layer[1])
        assert all(p.requires_grad for n, p in llm.named_parameters() if n.startswith("model.engram.0."))
        assert all(not p.requires_grad for n, p in llm.named_parameters() if n.startswith("model.engram.1."))
        assert all(not p.requires_grad for n, p in llm.named_parameters() if n.startswith("model.norm."))

    def test_frozen_tail_gets_no_grad(self, monkeypatch):
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(n_trainable_layers=1)
        model = build_toy_omni(modules)
        model.train()
        out = model({"conversation_list": toy_conversation()})
        out["loss"].backward()
        frozen = [p for n, p in modules[LLM].named_parameters() if n.startswith("model.layers.1.")]
        trained = [p for n, p in modules[LLM].named_parameters() if n.startswith("model.layers.0.")]
        assert frozen and all(p.grad is None for p in frozen)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained)


class TestDsparkTrainStep:
    def test_shrunk_stack_freezes_mtp(self):
        llm = DeepseekV41LlmAccelerated(
            tiny_llm_config(
                **{
                    **{k: v for k, v in shrink_layer_schedule(N_LAYERS, n_draft=1).items() if k != "kept_layer_ids"},
                    "dspark_block_size": 5,
                    "dspark_n_routed_experts": 4,
                    "n_trainable_layers": 1,
                }
            )
        )
        llm.freeze_model()
        mtp = [p for n, p in llm.named_parameters() if n.startswith("model.mtp.")]
        assert mtp and all(not p.requires_grad for p in mtp)

    def test_full_parameter_trains_mtp(self, monkeypatch):
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(dspark=True)
        mtp = [p for n, p in modules[LLM].named_parameters() if n.startswith("model.mtp.")]
        assert mtp and all(p.requires_grad for p in mtp)
        model = build_toy_omni(modules)
        model.train()
        out = model({"conversation_list": toy_conversation()})
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mtp)

    def test_mtp_ep_plan_resolves(self):
        modules = build_toy_modules(dspark=True)
        llm_names = {name for name, _ in modules[LLM].named_parameters()}
        assert "model.mtp.0.mlp.experts.gate_up_proj" in llm_names
        assert "model.mtp.0.mlp.experts.down_proj" in llm_names

    def test_frozen_mtp_skips_the_train_forward(self, monkeypatch):
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(dspark=True, n_trainable_layers=1)
        called = []
        orig = modules[LLM].model.forward_dspark_train

        def _wrapped(*args, **kwargs):
            called.append(True)
            return orig(*args, **kwargs)

        modules[LLM].model.forward_dspark_train = _wrapped
        model = build_toy_omni(modules)
        model.train()
        out = model({"conversation_list": toy_conversation()})
        assert torch.isfinite(out["loss"])
        assert called == []

    def test_packed_dspark_does_not_read_the_future(self):
        schedule = shrink_layer_schedule(N_LAYERS, n_draft=1)
        schedule.pop("kept_layer_ids", None)
        torch.manual_seed(0)
        llm = DeepseekV41LlmAccelerated(
            tiny_llm_config(
                **schedule,
                dspark_block_size=5,
                dspark_n_routed_experts=4,
                dspark_num_experts_per_tok=2,
            )
        )
        llm.train()
        batch, seq_len = 1, 6
        embeds = torch.randn(batch, seq_len, HIDDEN)
        drafts = [torch.randn(batch, seq_len, HIDDEN) for _ in llm.config.dspark_target_layer_ids]
        positions = torch.arange(seq_len).unsqueeze(0)
        clean = llm.model.forward_dspark_train(embeds, positions, drafts)
        dirty_embeds = embeds.clone()
        dirty_embeds[:, -1] = 1.0e3
        dirty_drafts = [hidden.clone() for hidden in drafts]
        for hidden in dirty_drafts:
            hidden[:, -1] = 1.0e3
        dirty = llm.model.forward_dspark_train(dirty_embeds, positions, dirty_drafts)
        torch.testing.assert_close(clean[0][:, :-1], dirty[0][:, :-1], atol=1e-4, rtol=1e-4)

    def test_packed_dspark_matches_decode_slot_zero(self):
        """Packed row ``j`` is decode's slot 0 after verified token ``j-1``, Markov residual included.

        One-slot blocks make the comparison exact: with more slots, decode's
        slot 0 also attends the block's noise tokens, which teacher forcing
        does not have.
        """
        from veomni.models.seed_omni.modules.deepseek_v41.llm.decode import DeepseekV41DecodeCache

        schedule = shrink_layer_schedule(N_LAYERS)
        schedule.pop("kept_layer_ids", None)
        torch.manual_seed(0)
        llm = DeepseekV41LlmAccelerated(
            tiny_llm_config(**schedule, dspark_block_size=1, dspark_n_routed_experts=4, dspark_num_experts_per_tok=2)
        ).eval()
        backbone = llm.model
        lengths = [9, 4]
        seq_len = sum(lengths)
        embeds = torch.randn(1, seq_len, HIDDEN)
        drafts = [torch.randn(1, seq_len, HIDDEN) for _ in llm.config.dspark_target_layer_ids]
        positions = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
        input_ids = torch.randint(0, VOCAB, (1, seq_len))
        with torch.no_grad():
            packed, markov_bias = backbone.forward_dspark_train(embeds, positions, drafts, input_ids=input_ids)
            markov = backbone.mtp[-1].markov_head
            expected_bias = torch.nn.functional.linear(
                torch.nn.functional.embedding(input_ids, markov.embed.weight), markov.head.weight
            )
            torch.testing.assert_close(markov_bias, expected_bias)

            start = 0
            for length in lengths:
                for j in range(1, length):
                    cache = DeepseekV41DecodeCache.empty(llm.config)
                    if j > 1:
                        backbone.seed_dspark_windows(
                            [d[:, start : start + j - 1] for d in drafts],
                            torch.arange(j - 1).unsqueeze(0),
                            cache.dspark,
                        )
                    verified = backbone.mtp[0].project_main([d[:, start + j - 1 : start + j] for d in drafts])
                    spec = backbone.forward_spec(
                        embeds[:, start + j : start + j + 1], position=j - 1, caches=cache.dspark, main_x=verified
                    )
                    torch.testing.assert_close(packed[:, start + j], spec[:, 0], atol=1e-5, rtol=1e-5)
                start += length

    def test_packed_dspark_enters_every_stage_through_call(self):
        """FSDP2 unshards a stage in its pre-forward hook; a bypassed ``__call__`` sees DTensor shards."""
        schedule = shrink_layer_schedule(N_LAYERS)
        schedule.pop("kept_layer_ids", None)
        llm = DeepseekV41LlmAccelerated(
            tiny_llm_config(**schedule, dspark_block_size=5, dspark_n_routed_experts=4, dspark_num_experts_per_tok=2)
        )
        llm.train()
        entered = []
        for stage in llm.model.mtp:
            stage.register_forward_pre_hook(lambda module, _args: entered.append(module.stage_id))
        batch, seq_len = 1, 6
        drafts = [torch.randn(batch, seq_len, HIDDEN) for _ in llm.config.dspark_target_layer_ids]
        draft, markov_bias = llm.model.forward_dspark_train(
            torch.randn(batch, seq_len, HIDDEN),
            torch.arange(seq_len).unsqueeze(0),
            drafts,
            input_ids=torch.randint(0, VOCAB, (batch, seq_len)),
        )
        assert entered == list(range(len(llm.model.mtp)))
        assert draft.shape == (batch, seq_len, HIDDEN)
        assert markov_bias.shape == (batch, seq_len, VOCAB)

    def test_dspark_stages_checkpoint_with_the_same_grads(self):
        """Each stage's MoE holds as much as a decoder layer's until backward, so it is checkpointed too."""
        schedule = shrink_layer_schedule(N_LAYERS)
        schedule.pop("kept_layer_ids", None)
        torch.manual_seed(0)
        config = tiny_llm_config(
            **schedule, dspark_block_size=5, dspark_n_routed_experts=4, dspark_num_experts_per_tok=2
        )
        plain = DeepseekV41LlmAccelerated(config).train()
        checkpointed = DeepseekV41LlmAccelerated(config).train()
        checkpointed.load_state_dict(plain.state_dict())
        checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        assert all(stage.gradient_checkpointing for stage in checkpointed.model.mtp)

        batch, seq_len = 1, 6
        embeds = torch.randn(batch, seq_len, HIDDEN)
        drafts = [torch.randn(batch, seq_len, HIDDEN) for _ in config.dspark_target_layer_ids]
        positions = torch.arange(seq_len).unsqueeze(0)
        input_ids = torch.randint(0, VOCAB, (batch, seq_len))
        grads = []
        for llm in (plain, checkpointed):
            leaves = [embeds.clone().requires_grad_(), *(d.clone().requires_grad_() for d in drafts)]
            draft, markov_bias = llm.model.forward_dspark_train(leaves[0], positions, leaves[1:], input_ids=input_ids)
            (draft.float().square().mean() + markov_bias.float().square().mean()).backward()
            mtp = {n: p.grad for n, p in llm.named_parameters() if n.startswith("model.mtp.") and p.grad is not None}
            grads.append((mtp, [leaf.grad for leaf in leaves]))
        assert grads[0][0].keys() == grads[1][0].keys() and grads[0][0]
        for name, grad in grads[0][0].items():
            torch.testing.assert_close(grads[1][0][name], grad, msg=name)
        for ref, got in zip(grads[0][1], grads[1][1]):
            torch.testing.assert_close(got, ref)

    def test_decode_pre_masks_rows_decode_never_drafts_from(self):
        """DSpark labels are the causal t+1 labels minus sample starts and vision rows."""
        te = DeepseekV41TextEncoderAccelerated(tiny_text_config())

        def _part(kind: str, length: int, labels: list[int] | None = None, *, dspark: bool = True):
            meta = {} if labels is None else {"labels": torch.tensor(labels)}
            if dspark:
                meta["dspark_hidden"] = torch.randn(length, HIDDEN)
                meta["dspark_markov_bias"] = torch.randn(length, VOCAB)
            return ConversationItem(type=kind, value=torch.randn(length, HIDDEN), role="assistant", meta=meta)

        packed = te.decode_pre(
            conversation_list=[
                [_part("text", 3, [1, 2, 3])],
                [_part("image", 4), _part("text", 2, [4, 5])],
            ]
        )
        assert packed["shift_labels"].tolist() == [2, 3, IGNORE_INDEX, 4, 5, IGNORE_INDEX]
        assert packed["mtp_labels"].tolist() == [IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX, 5, IGNORE_INDEX]
        assert packed["dspark_hidden_states"].shape == (6, HIDDEN)
        assert packed["dspark_markov_bias"].shape == (6, VOCAB)

        no_draft = te.decode_pre(conversation_list=[[_part("text", 3, [1, 2, 3], dspark=False)]])
        assert "mtp_labels" not in no_draft
        assert "dspark_hidden_states" not in no_draft

    def test_decode_dspark_ce_adds_the_markov_residual(self, monkeypatch):
        """Draft CE scores ``lm_head(h) + markov`` on the DSpark labels, on top of the main loss."""
        from veomni.models.seed_omni.modules.base.text_encoder.accelerated.accelerated import (
            TrainingMixin as BaseTrainingMixin,
        )

        def _fake_decode(self, hidden_states=None, labels=None, shift_labels=None, **kwargs):
            del self, hidden_states, labels, shift_labels, kwargs
            return {"loss": torch.tensor(1.0), "logits": None}

        monkeypatch.setattr(BaseTrainingMixin, "decode", _fake_decode)
        te = DeepseekV41TextEncoderAccelerated(tiny_text_config())
        torch.manual_seed(0)
        draft = torch.randn(5, HIDDEN)
        markov_bias = torch.randn(5, VOCAB)
        mtp_labels = torch.tensor([IGNORE_INDEX, 3, 7, IGNORE_INDEX, 5])
        out = te.decode(
            hidden_states=torch.randn(5, HIDDEN),
            shift_labels=torch.tensor([2, 3, 7, 4, 5]),
            dspark_hidden_states=draft,
            dspark_markov_bias=markov_bias,
            mtp_labels=mtp_labels,
        )
        keep = mtp_labels != IGNORE_INDEX
        expected = torch.nn.functional.cross_entropy(
            (te._project(draft) + markov_bias)[keep].float(), mtp_labels[keep]
        )
        torch.testing.assert_close(out["loss"], 1.0 + expected)

        with pytest.raises(RuntimeError, match="dspark_markov_bias"):
            te.decode(
                hidden_states=torch.randn(5, HIDDEN),
                shift_labels=torch.tensor([2, 3, 7, 4, 5]),
                dspark_hidden_states=draft,
                mtp_labels=mtp_labels,
            )


def _has_grad(module, prefix: str) -> bool:
    params = [p for n, p in module.named_parameters() if n.startswith(prefix)]
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


def _all_frozen(module, prefix: str) -> bool:
    params = [p for n, p in module.named_parameters() if n.startswith(prefix)]
    return bool(params) and all(not p.requires_grad for p in params)


class TestTrainableLayerMatrix:
    """Partial freeze must work for prefix layers, MTP-target layers, and MTP itself."""

    def test_cannot_set_both_selectors(self):
        with pytest.raises(ValueError, match="n_trainable_layers and trainable_layer_ids"):
            tiny_llm_config(n_trainable_layers=1, trainable_layer_ids=[0])

    def test_mtp_target_ids_stay_unfrozen_when_named(self):
        llm = DeepseekV41LlmAccelerated(
            tiny_llm_config(
                **{
                    **{k: v for k, v in shrink_layer_schedule(N_LAYERS, n_draft=1).items() if k != "kept_layer_ids"},
                    "dspark_block_size": 5,
                    "dspark_n_routed_experts": 4,
                    "trainable_layer_ids": [1],
                    "freeze_mtp": False,
                }
            )
        )
        llm.freeze_model()
        assert _all_frozen(llm, "model.layers.0.")
        assert all(p.requires_grad for n, p in llm.named_parameters() if n.startswith("model.layers.1."))
        assert all(p.requires_grad for n, p in llm.named_parameters() if n.startswith("model.mtp."))

    @pytest.mark.parametrize(
        "label, kwargs, train_layers, train_mtp",
        [
            ("first_two_freeze_mtp", {"n_trainable_layers": 2}, {0, 1}, False),
            ("both_layers_and_mtp", {"trainable_layer_ids": [0, 1], "freeze_mtp": False}, {0, 1}, True),
            ("encoder_only", {"trainable_layer_ids": [0]}, {0}, False),
            ("decoder_mtp_layer", {"trainable_layer_ids": [1]}, {1}, False),
            ("decoder_and_mtp", {"trainable_layer_ids": [1], "freeze_mtp": False}, {1}, True),
            ("mtp_only", {"trainable_layer_ids": [], "freeze_mtp": False}, set(), True),
        ],
    )
    def test_two_layer_dspark_train_step(self, monkeypatch, label, kwargs, train_layers, train_mtp):
        del label
        _patch_cpu_attention(monkeypatch)
        modules = build_toy_modules(dspark=True, **kwargs)
        model = build_toy_omni(modules)
        model.train()
        out = model({"conversation_list": toy_conversation()})
        assert torch.isfinite(out["loss"]), out
        out["loss"].backward()
        llm = modules[LLM]
        for layer_idx in range(N_LAYERS):
            prefix = f"model.layers.{layer_idx}."
            if layer_idx in train_layers:
                assert _has_grad(llm, prefix), f"layer {layer_idx} should train"
            else:
                assert _all_frozen(llm, prefix), f"layer {layer_idx} should be frozen"
                frozen = [p for n, p in llm.named_parameters() if n.startswith(prefix)]
                assert all(p.grad is None for p in frozen)
        if train_mtp:
            assert _has_grad(llm, "model.mtp.")
        else:
            assert _all_frozen(llm, "model.mtp.")

    def test_four_layer_last_two_are_the_mtp_targets(self, monkeypatch):
        """4-layer CED: first two are encoder-ish; last two are DSpark targets.

        ``shrink_layer_schedule(4, n_draft=1)`` sets ``dspark_target_layer_ids``
        to ``[1, 2, 3]``, so training ``[2, 3]`` plus MTP is the 'keep the MTP
        layers' counterpart of ``n_trainable_layers: 2``.
        """
        _patch_cpu_attention(monkeypatch)
        schedule = shrink_layer_schedule(4, n_draft=1)
        assert schedule["dspark_target_layer_ids"] == [1, 2, 3]
        modules = build_toy_modules(
            dspark=True,
            n_layers=4,
            trainable_layer_ids=[2, 3],
            freeze_mtp=False,
        )
        model = build_toy_omni(modules)
        model.train()
        out = model({"conversation_list": toy_conversation()})
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        llm = modules[LLM]
        assert _all_frozen(llm, "model.layers.0.")
        assert _all_frozen(llm, "model.layers.1.")
        assert _has_grad(llm, "model.layers.2.")
        assert _has_grad(llm, "model.layers.3.")
        assert _has_grad(llm, "model.mtp.")


class TestQatOpsSlotBinding:
    def test_bind_on_accelerated_module_enables_dense_and_expert_qat(self):
        """YAML ``qat_implementation`` is a no-op unless both slots live here.

        ``_bind_veomni_ops`` only scans the OmniModule's Python module. Dense
        GEMM / KV / indexer QAT is in ``quantization.py``; routed-expert QAT
        closes over the reused V4 generated slot. Both have to be attributes
        of ``accelerated.py`` or the 8xH20 job trains in bf16 while thinking
        it is doing QAT.
        """
        from veomni.arguments.arguments_types import OpsImplementationConfig
        from veomni.models.auto import _bind_veomni_ops
        from veomni.models.seed_omni.modules.deepseek_v41.llm import quantization as qat
        from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated import accelerated as acc
        from veomni.models.seed_omni.modules.deepseek_v41.llm.reuse import veomni_qat_implementation as expert_slot

        dense_slot = qat.veomni_qat_implementation
        dense_before, expert_before = dense_slot.value, expert_slot.value
        try:
            ops = OpsImplementationConfig()
            ops.qat_implementation = "fp8_blockwise"
            assert _bind_veomni_ops(acc, ops)
            assert qat.qat_enabled()
            assert expert_slot.value == "fp8_blockwise"
            assert dense_slot is not expert_slot
        finally:
            dense_slot._value = dense_before
            expert_slot._value = expert_before
