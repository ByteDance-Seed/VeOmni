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

"""DeepSeek-V4.1 llm under sequence parallelism against the unsharded forward.

Gloo on CPU, with the dense gather standing in for the TileLang kernels as the
toy train step does. The schedule keeps every layer kind -- window-only, the
two KV sources at ratios 2 and 1, the candidate source, the consumers reusing
its top-k -- plus Engram, the indexer KL and two DSpark stages. The packed row
is 29 tokens in three samples, so both group sizes pad, and a shard boundary
falls inside a sample: windows and compression groups straddle ranks.

The comparison is the one training sees. Each rank's loss is the replicated
downstream loss on the gathered output plus its own KL share; FSDP averages
gradients over the group, so the rank-summed gradient divided by the group
size must equal the unsharded gradient, and the rank-mean loss the unsharded
loss.
"""

import math
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig


SAMPLE_LENGTHS = (13, 4, 12)
HIDDEN = 64
INDEXER_COEF = 0.5


@torch.no_grad()
def eager_sparse_mqa_target_fwd(q, kv, topk_idxs, lse, sm_scale=None):
    """CPU stand-in for the TileLang teacher: head-summed ``exp2(score - lse)``.

    Gradient-free like the forward-only kernel. A differentiable teacher would
    send ``0 * inf`` back through the KL's masked slots.
    """
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


def tiny_config() -> DeepseekV41LlmConfig:
    # Layer 2 reuses layer 1's ratio-2 history and top-k; layer 4 indexes
    # layer 3's ratio-1 history inside layer 3's candidate blocks.
    schedule = dict(
        num_hidden_layers=6,
        num_nextn_predict_layers=2,
        compress_ratios=[0, 2, 2, 1, 1, 0, 0, 0],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4],
        candidate_source_layer_id=3,
        engram_layer_ids=[1, 4],
        dspark_target_layer_ids=[3, 4, 5],
    )
    return DeepseekV41LlmConfig(
        vocab_size=64,
        dspark_noise_token_id=31,
        image_token_id=4,
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
        rms_norm_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        # More candidate slots than index_topk, as released (16384 vs 512). At
        # fewer, a row short of candidates pads its top-k with merely-visible
        # groups picked by tie order, which depends on the compressed width.
        candidate_topk_blocks=4,
        candidate_block_size=2,
        max_position_embeddings=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_shared_experts=1,
        dspark_n_routed_experts=4,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
        dsa_indexer_loss=True,
        dsa_indexer_loss_coef=INDEXER_COEF,
        weight_quantization=None,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 256,
        },
        **schedule,
    )


def _inputs(config):
    generator = torch.Generator().manual_seed(1)
    total = sum(SAMPLE_LENGTHS)
    embeds = torch.randn(1, total, HIDDEN, generator=generator)
    positions = torch.cat([torch.arange(n) for n in SAMPLE_LENGTHS]).unsqueeze(0)
    cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    ngram = [
        torch.randn(1, total, cols, config.engram_head_dim, generator=generator) * 0.1 for _ in config.engram_layer_ids
    ]
    image_mask = torch.zeros(1, total, dtype=torch.bool)
    image_mask[:, 18:23] = True
    weights = torch.randn(1, total, HIDDEN, generator=generator)
    input_ids = torch.randint(0, config.vocab_size, (1, total), generator=generator)
    return embeds, positions, ngram, image_mask, weights, input_ids


def _loss(outputs, hidden, dspark, markov, weights):
    from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import fold_indexer_kl_loss

    return (
        (hidden * weights).sum()
        + (dspark * weights.flip(1)).sum()
        + (markov * markov.detach().sin()).sum()
        + fold_indexer_kl_loss(outputs, INDEXER_COEF)
    )


def _worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        from veomni.models.seed_omni.modules.deepseek_v41.llm import attention
        from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import shard_packed_row
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Llm
        from veomni.models.seed_omni.modules.deepseek_v41.llm.reuse import dense_sparse_attn
        from veomni.models.seed_omni.modules.deepseek_v41.mtp.configuration import DeepseekV41MtpConfig
        from veomni.models.seed_omni.modules.deepseek_v41.mtp.modeling import DeepseekV41Mtp

        attention.sparse_attn_tilelang = dense_sparse_attn
        attention.sparse_mqa_target_fwd = eager_sparse_mqa_target_fwd

        config = tiny_config()
        config.emit_draft_hiddens = True
        mtp_config = DeepseekV41MtpConfig(**{k: v for k, v in config.to_dict().items() if k != "model_type"})
        torch.manual_seed(0)
        model = DeepseekV41Llm(config).float().train()
        mtp = DeepseekV41Mtp(mtp_config).float().train()
        assert len(mtp.model.mtp) == 2
        embeds, positions, ngram, image_mask, weights, input_ids = _inputs(config)

        reference_embeds = embeds.clone().requires_grad_(True)
        reference = model(
            inputs_embeds=reference_embeds,
            position_ids=positions,
            ngram_embeds=ngram,
            image_mask=image_mask,
        )
        reference_mtp = mtp(
            inputs_embeds=reference_embeds,
            position_ids=positions,
            draft_hiddens=reference["draft_hiddens"],
            image_mask=image_mask,
            dspark_input_ids=input_ids,
        )
        reference_loss = _loss(
            reference,
            reference["hidden_states"],
            reference_mtp["dspark_hidden_states"],
            reference_mtp["dspark_markov_bias"],
            weights,
        )
        reference_loss.backward()
        reference_grads = {
            **{f"llm.{name}": p.grad.clone() for name, p in model.named_parameters() if p.grad is not None},
            **{f"mtp.{name}": p.grad.clone() for name, p in mtp.named_parameters() if p.grad is not None},
        }
        model.zero_grad(set_to_none=True)
        mtp.zero_grad(set_to_none=True)

        sp_embeds = embeds.clone().requires_grad_(True)
        local_embeds, local_positions, local_ngram, local_mask, shard = shard_packed_row(
            sp_embeds, positions, ngram, image_mask, dist.group.WORLD, rank, world_size
        )
        assert shard.full_len % world_size == 0
        outputs = model(
            inputs_embeds=local_embeds,
            position_ids=local_positions,
            ngram_embeds=local_ngram,
            image_mask=local_mask,
            sequence_shard=shard,
        )
        mtp_outputs = mtp(
            inputs_embeds=local_embeds,
            position_ids=local_positions,
            draft_hiddens=outputs["draft_hiddens"],
            image_mask=local_mask,
            sequence_shard=shard,
            dspark_input_ids=input_ids,
        )
        hidden = shard.gather(outputs["hidden_states"]).narrow(1, 0, shard.valid_len)
        dspark = shard.gather(mtp_outputs["dspark_hidden_states"]).narrow(1, 0, shard.valid_len)
        markov = shard.gather(mtp_outputs["dspark_markov_bias"]).narrow(1, 0, shard.valid_len)
        loss = _loss(outputs, hidden, dspark, markov, weights)
        loss.backward()

        torch.testing.assert_close(hidden, reference["hidden_states"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(dspark, reference_mtp["dspark_hidden_states"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(markov, reference_mtp["dspark_markov_bias"], rtol=1e-4, atol=1e-5)

        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(mean_loss / world_size, reference_loss.detach(), rtol=1e-5, atol=1e-5)

        input_grad = sp_embeds.grad.clone()
        dist.all_reduce(input_grad)
        torch.testing.assert_close(input_grad / world_size, reference_embeds.grad, rtol=1e-4, atol=1e-5)

        sp_grads = {
            **{f"llm.{name}": p.grad for name, p in model.named_parameters() if p.grad is not None},
            **{f"mtp.{name}": p.grad for name, p in mtp.named_parameters() if p.grad is not None},
        }
        assert sp_grads.keys() == reference_grads.keys()
        indexer_grads = [name for name in sp_grads if ".indexer." in name]
        assert indexer_grads, "the indexer KL reached no indexer parameter"
        for name, grad in sp_grads.items():
            dist.all_reduce(grad)
            torch.testing.assert_close(
                grad / world_size,
                reference_grads[name],
                rtol=1e-4,
                atol=1e-5,
                msg=lambda msg, name=name: f"{name}: {msg}",
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo required")
@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_forward_and_gradients_match_the_unsharded_row(world_size: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        mp.spawn(_worker, args=(world_size, init_file), nprocs=world_size, join=True)


def _skip_sp_gather_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
        from veomni.models.seed_omni.modules.deepseek_v41.llm import attention
        from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import DeepseekV41LlmAccelerated
        from veomni.models.seed_omni.modules.deepseek_v41.llm.reuse import dense_sparse_attn
        from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.accelerated.accelerated import (
            DeepseekV41TextEncoderAccelerated,
        )
        from veomni.models.seed_omni.modules.deepseek_v41.text_encoder.configuration import (
            DeepseekV41TextEncoderConfig,
        )
        from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem

        attention.sparse_attn_tilelang = dense_sparse_attn
        attention.sparse_mqa_target_fwd = eager_sparse_mqa_target_fwd

        def make_conversations():
            gen = torch.Generator().manual_seed(7)
            convs = []
            for slen in SAMPLE_LENGTHS:
                ids = torch.randint(0, 64, (slen,), generator=gen)
                labels = ids.clone()
                labels[:3] = -100
                convs.append(
                    [
                        ConversationItem(
                            type="text",
                            value=ids,
                            role="assistant",
                            meta={
                                "input_ids": ids.clone(),
                                "labels": labels,
                                "attention_mask": torch.ones(slen, dtype=torch.long),
                                _OMNI_TOKEN_LEN: slen,
                            },
                        )
                    ]
                )
            return convs

        def run_pipeline(skip_gather: bool):
            clear_parallel_state()
            _init_parallel_state(
                dp_size=1,
                dp_replicate_size=1,
                dp_shard_size=1,
                tp_size=1,
                pp_size=1,
                cp_size=world_size,
                ulysses_size=1,
                device_type="cpu",
                skip_sp_gather_between_modules=skip_gather,
                name="base",
            )
            from veomni.models.seed_omni.modules.deepseek_v41.mtp.accelerated import DeepseekV41MtpAccelerated
            from veomni.models.seed_omni.modules.deepseek_v41.mtp.configuration import DeepseekV41MtpConfig

            config = tiny_config()
            config.emit_draft_hiddens = True
            mtp_config = DeepseekV41MtpConfig(**{k: v for k, v in config.to_dict().items() if k != "model_type"})
            te_cfg = DeepseekV41TextEncoderConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                pad_token_id=None,
                tie_word_embeddings=False,
            )
            torch.manual_seed(0)
            te = DeepseekV41TextEncoderAccelerated(te_cfg).float().train()
            llm = DeepseekV41LlmAccelerated(config).float().train()
            mtp = DeepseekV41MtpAccelerated(mtp_config).float().train()

            convs = make_conversations()
            cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
            gen = torch.Generator().manual_seed(11)
            total = sum(SAMPLE_LENGTHS)
            ngram = [
                torch.randn(1, total, cols, config.engram_head_dim, generator=gen) * 0.1
                for _ in config.engram_layer_ids
            ]

            enc_in = te.pre_forward(method="encode", conversation_list=convs)
            enc_out = te.encode(**enc_in)
            enc_post = te.post_forward(method="encode", **enc_out)

            llm_in = llm.pre_forward(
                method="forward",
                conversation_list=enc_post["conversation_list"],
                ngram_embeds=ngram,
            )
            llm_out = llm.forward(**llm_in)
            llm_post = llm.post_forward(method="forward", **llm_out)

            mtp_in = mtp.pre_forward(
                method="forward",
                conversation_list=llm_post["conversation_list"],
                dspark_inputs=llm_post["dspark_inputs"],
            )
            mtp_out = mtp.forward(**mtp_in)
            mtp_post = mtp.post_forward(method="forward", **mtp_out)

            if skip_gather:
                # Verify conversation_list text items hold only this rank's 2D shard slice
                assert all(item.value.dim() == 2 for conv in mtp_post["conversation_list"] for item in conv)
                assert sum(item.value.size(0) for conv in mtp_post["conversation_list"] for item in conv) < sum(
                    SAMPLE_LENGTHS
                )

            dec_in = te.pre_forward(method="decode", conversation_list=mtp_post["conversation_list"])
            dec_out = te.decode(**dec_in)
            dec_post = te.post_forward(method="decode", **dec_out)

            total_loss = (
                dec_post["_loss"]["ce_loss"] + dec_post["_loss"]["mtp_loss"] + llm_post["_loss"]["indexer_kl_loss"]
            )
            total_loss.backward()
            te_grads = {f"te.{k}": p.grad.clone() for k, p in te.named_parameters() if p.grad is not None}
            llm_grads = {f"llm.{k}": p.grad.clone() for k, p in llm.named_parameters() if p.grad is not None}
            mtp_grads = {f"mtp.{k}": p.grad.clone() for k, p in mtp.named_parameters() if p.grad is not None}
            for g in (*te_grads.values(), *llm_grads.values(), *mtp_grads.values()):
                dist.all_reduce(g)
            mean_loss = total_loss.detach().clone()
            dist.all_reduce(mean_loss)
            return mean_loss / world_size, {**te_grads, **llm_grads, **mtp_grads}

        loss_gather, grads_gather = run_pipeline(skip_gather=False)
        loss_skip, grads_skip = run_pipeline(skip_gather=True)

        torch.testing.assert_close(loss_skip, loss_gather, rtol=1e-5, atol=1e-5)
        assert grads_skip.keys() == grads_gather.keys()
        for name in grads_gather:
            torch.testing.assert_close(
                grads_skip[name] / world_size,
                grads_gather[name] / world_size,
                rtol=1e-4,
                atol=1e-5,
                msg=lambda msg, name=name: f"{name}: {msg}",
            )
    finally:
        from veomni.distributed.parallel_state import clear_parallel_state

        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo required")
def test_skip_sp_gather_between_modules_matches_full_gather() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        mp.spawn(_skip_sp_gather_worker, args=(2, init_file), nprocs=2, join=True)


def test_fold_dummy_anchors_sums_rather_than_means_on_a_zero_row_window():
    """The V4.1 FSDP dummy anchor must survive a rank whose CP window is empty.

    ``fold_dummy_anchors`` folds ``0 * mean`` of every dummy item so the vision
    tower stays in each rank's autograd graph. Under CP a dummy's rows can fall
    entirely outside one rank's window, and ``mean`` of a zero-row tensor is
    ``NaN`` (``0.0 * NaN`` is still ``NaN``) -- while the previous ``numel() > 0``
    guard dropped the anchor outright. Either way that rank never issues its
    reduce-scatter for the tower's parameters, and FSDP2 has no unused-parameter
    pass, so the run hangs in ``backward`` rather than failing. ``sum`` is 0.0
    over no rows and keeps the ``grad_fn``.
    """
    from veomni.models.seed_omni.modules.base.sp_shard import sp_valid_in_window
    from veomni.models.seed_omni.modules.deepseek_v41.llm.accelerated.accelerated import fold_dummy_anchors
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    # 4 dummy rows over 3 ranks: the last rank's window holds none of them.
    assert [sp_valid_in_window(4, rank, 3) for rank in range(3)] == [2, 2, 0]

    vision = torch.nn.Linear(8, 8)
    embeds = torch.zeros(1, 4, 8)

    empty = vision(torch.zeros(4, 8))[:0]
    assert empty.numel() == 0
    assert torch.isnan(empty.mean() * 0.0), "a zero-row mean must be the NaN this guards against"

    item = ConversationItem(type="image", value=empty, role="user", is_dummy=True)
    anchored = fold_dummy_anchors(embeds, [[item]])

    assert torch.isfinite(anchored).all()
    anchored.sum().backward()
    assert vision.weight.grad is not None, "the dummy anchor must keep the tower's params in the graph"
    assert torch.equal(vision.weight.grad, torch.zeros_like(vision.weight.grad))
