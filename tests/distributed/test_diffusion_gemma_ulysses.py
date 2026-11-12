# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Ulysses sequence parallelism over DiffusionGemma's encoder, on real ranks.

Two ranks against one, on the same packed row and the same weights. Nothing here
is observable single-process: the all-to-all inside the encoder attention, the
shard each rank claims, the hidden state each borrows from its neighbour for the
next-token shift, and the factor that keeps FSDP's *average* over the SP group
equal to one rank's gradient.

Both halves of Eq. 13 are covered, in two tests rather than one, because they
fail differently. The encoder test compares hidden states, so a wrong shard
shows up positionally; the whole-model test goes through `forward`, where the
decoder reads a cache gathered back out of the shards and a canvas that saw
the wrong context is only visible in the loss. The paths the decoder has that
*cannot* read a sharded cache are refused, and pinned here as refusals, since
their shapes line up and the number they would return looks ordinary.
"""

import os

import pytest
import torch
import torch.distributed as dist


TOY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "toy_config", "diffusion_gemma_toy")

#: Two samples and an odd total, so the sequence-parallel padding is exercised
#: rather than merely permitted: 21 tokens over 2 ranks needs one pad token, and
#: that token has to become a third segment. The reference arm is handed the same
#: prompts *without* that tail, so the comparison also says the tail is inert.
PROMPTS = (13, 8)
#: `dllm_hole_attention` sizes its query tile to divide the canvas, and 16 is the
#: narrowest tile a Triton dot takes, so this is the toy config's `canvas_length`
#: and not a free parameter.
CANVAS = 16
CANVASES = 2


def _build_row(config, device, sp_size: int):
    """One packed row, built by the shipped collator so the shapes are the real ones."""
    from veomni.data.data_collator import DiffusionGemmaCollator
    from veomni.utils.constants import IGNORE_INDEX

    features = []
    for index, prompt_tokens in enumerate(PROMPTS):
        torch.manual_seed(100 + index)
        vocab = config.text_config.vocab_size
        labels = torch.full((CANVASES, CANVAS), IGNORE_INDEX, dtype=torch.long)
        labels[:, ::2] = torch.randint(3, vocab, (CANVASES, CANVAS // 2))
        weight = torch.zeros(CANVASES, CANVAS)
        weight[:, ::2] = 2.0 / CANVAS
        features.append(
            {
                "input_ids": torch.randint(3, vocab, (prompt_tokens,)),
                "encoder_prompt_mask": torch.zeros(prompt_tokens, dtype=torch.bool),
                "decoder_input_ids": torch.randint(3, vocab, (CANVASES, CANVAS)),
                "labels": labels,
                "loss_weight": weight,
                "decoder_context_mask": torch.ones(CANVASES, prompt_tokens, dtype=torch.bool),
            }
        )

    batch = DiffusionGemmaCollator(pack_samples=True, sp_size=sp_size)(features)
    return {key: value.to(device) for key, value in batch.items()}


def _build_model(device, attn_dg_prefix: bool = True, kv_heads: int | None = None):
    """A toy model in the one configuration SP accepts, with the AR term on.

    `attn_dg_prefix` is the recipe's setting and the decoder's only sharded path:
    it is what routes `flat_seq_pack` to the hole kernel, which reads each
    canvas's visible run as an interval over a cache the attention has gathered
    back to full length. With it off, `sequence_parallel_preflight` refuses at
    model build rather than letting the decoder reach its concat-path refusal after
    allocating a dense L² mask.

    `kv_heads` overrides both KV head counts, which is how the repeat branch is
    reached: the toy config has two of them, so at sp=2 every rank gets one and
    `_kv_repeats` returns 1. One KV head makes it 2.
    """
    from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
    from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

    config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
    config.canvas_batching = "flat_seq_pack"
    config.encoder_ar_loss = True
    config._attn_implementation = "sdpa"
    config.text_config.attn_dg_prefix = attn_dg_prefix
    if kv_heads is not None:
        config.text_config.num_key_value_heads = kv_heads
        config.text_config.num_global_key_value_heads = kv_heads
    torch.manual_seed(0)
    # bfloat16 because the 25 sliding layers reach `flash_attn_varlen_func`, which has
    # no fp32 path. The comparison below is therefore on relative norms rather than
    # `assert_close`'s dtype defaults.
    return DiffusionGemmaForBlockDiffusion(config).to(device=device, dtype=torch.bfloat16).train()


def _encoder_and_ar_term(model, batch):
    """The encoder stack and the AR term, which is what SP covers today.

    `forward` is not available: it always runs the decoder, which refuses under SP.
    `encoder_ar_term` is the model's own method, so the arithmetic compared here is
    the arithmetic a training step performs.
    """
    encoder_outputs = model.model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        pack_samples=True,
        return_dict=True,
    )
    loss = model.encoder_ar_term(
        encoder_outputs.last_hidden_state,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        mm_token_type_ids=None,
        encoder_prompt_mask=batch["encoder_prompt_mask"],
        encoder_ar_row_weight=None,
        pack_samples=True,
    )
    return encoder_outputs.last_hidden_state, loss


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative Frobenius norm, which is the only scale-free measure available here.

    An elementwise tolerance cannot be set on these: the gradients span several
    orders of magnitude across 131 tensors, and the two arms reduce their sums in
    different orders and different tree shapes.
    """
    scale = expected.float().norm()
    return float((actual.float() - expected.float()).norm() / scale.clamp(min=1e-12))


def _gradients(model) -> dict:
    return {name: p.grad.detach().float().clone() for name, p in model.named_parameters() if p.grad is not None}


def _both_topologies(world_size: int):
    """A data-parallel-only topology and an Ulysses one over the same ranks.

    Two real meshes rather than a stubbed state, so the reference arm is genuinely
    a run with SP off: every rank holds the whole row, which is what one rank
    would do. `_init_parallel_state` installs only the first as the global one and
    returns both, and `use_parallel_state` is the supported way to swap. Neither
    claims a registry name: a second call under the same name hands back the first.
    """
    from veomni.distributed.parallel_state import _init_parallel_state

    replicated = _init_parallel_state(dp_size=world_size, dp_shard_size=world_size, ulysses_size=1, name=None)
    sharded = _init_parallel_state(dp_size=1, dp_shard_size=1, ulysses_size=world_size, name=None)
    return replicated, sharded


def _equivalence_worker():
    from veomni.distributed.parallel_state import get_parallel_state, use_parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(world_size)

    # The reference: SP off, so this rank runs the whole row -- the same computation
    # a single-GPU run performs. Its row carries no alignment tail, so what the two
    # arms are asked to agree on includes the tail contributing nothing.
    model = _build_model(device)
    reference_batch = _build_row(model.config, device, sp_size=1)
    full_hidden, full_loss = _encoder_and_ar_term(model, reference_batch)
    full_loss.backward()
    full_hidden, full_loss, full_grads = full_hidden.detach().float(), float(full_loss.detach()), _gradients(model)
    del model

    with use_parallel_state(sharded):
        state = get_parallel_state()
        assert state.sp_enabled and state.sp_size == world_size, (
            f"[Rank {rank}] the topology did not come up: sp_size={state.sp_size}"
        )
        model = _build_model(device)
        batch = _build_row(model.config, device, sp_size=world_size)
        hidden, loss = _encoder_and_ar_term(model, batch)
        loss.backward()
        shard_grads = _gradients(model)

    real = full_hidden.shape[1]
    total = batch["input_ids"].shape[1]
    local = total // world_size
    assert total > real, (
        f"[Rank {rank}] the collator padded a {real}-token row to {total} over {world_size} ranks, so this "
        "row does not exercise the alignment tail"
    )
    assert hidden.shape[1] == local, (
        f"[Rank {rank}] the encoder returned {hidden.shape[1]} states for a {total}-token row over "
        f"{world_size} ranks; SP is not sharding the sequence"
    )

    # 1. The forward. Attention gathers the row back to full length and splits the
    # heads, and every head's arithmetic is then the arithmetic of the unsharded
    # run -- so this is expected to be near-exact, not merely close. The rest of the
    # stack is token-wise. Only the real tokens are compared; the tail has no
    # counterpart in the reference row, and nothing downstream reads it.
    #
    # The bound is stated in bf16's own units. One ulp is a relative 2**-8, and
    # what the two arms produce is that: most states agree bit for bit and the
    # rest by a single ulp, spread over the shard rather than concentrated at the
    # token whose predecessor lives on another rank. Two ulps of slack, so a real
    # error has to be several times the representation to hide in it.
    ulp = 2**-8
    start, stop = rank * local, min((rank + 1) * local, real)
    error = _relative_error(hidden.float()[:, : stop - start], full_hidden[:, start:stop])
    assert error < 2 * ulp, (
        f"[Rank {rank}] the sharded encoder states differ from the whole row by {error:.2e}, more than "
        f"two bf16 ulps ({2 * ulp:.1e})"
    )

    # 2. The loss value. A sum over tokens, all-reduced, so it should equal the
    # unsharded sum -- including the token at each shard's start, whose predecessor
    # lives on the rank before. Dropping those would show up here as a deficit of
    # `sp_size - 1` tokens' worth. An alignment tail that reached the AR term would
    # show up as a surplus, since the reference row does not have one.
    # The bound is tight because the two arms add the same terms: the row holds 19
    # supervised tokens, so losing the one token per shard that has to be borrowed
    # would move this by percent, and what is actually seen is 1e-6.
    deficit = abs(float(loss.detach()) - full_loss) / max(abs(full_loss), 1e-9)
    assert deficit < 1e-4, (
        f"[Rank {rank}] the sharded AR loss is {float(loss.detach())} against {full_loss} unsharded, off "
        f"by {deficit:.1e}"
    )

    # 3. The gradients, as FSDP will see them. FSDP2 shards over `dp_shard_sp` and
    # *averages* over it, so what has to match one rank's gradient is the mean of
    # the ranks' gradients -- which is what `sum_over_shards` scales the backward
    # for. Comparing the local gradient instead would pass at sp=1 and be off by
    # exactly `sp_size` here.
    assert set(shard_grads) == set(full_grads), f"[Rank {rank}] SP changed which parameters have gradients"
    worst, worst_name = 0.0, ""
    for name, expected in full_grads.items():
        averaged = shard_grads[name].clone()
        dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
        averaged /= world_size
        error = _relative_error(averaged, expected)
        if error > worst:
            worst, worst_name = error, name
    assert worst < 5e-2, f"[Rank {rank}] {worst_name} is off by {worst:.2e} after averaging over the SP group"

    # (3) is where the `sp_size` on the backward is tested, and it is enough on its
    # own: without it every rank's local gradient is its own shard's contribution,
    # so the mean over the group is `g_full / sp_size` and (3) fails by 50% at
    # sp=2. An earlier version of this test added a fourth assertion -- that one
    # rank's *un-averaged* gradient differs from the whole by more than 10% -- on
    # the theory that (3) alone could not see a missing scale. That is backwards:
    # the un-averaged comparison is off by about a factor of two either way, so it
    # passed whether or not the scale was there. `test_the_backward_carries_the_sp_size_factor`
    # pins the factor itself.
    dist.barrier()


def _whole_model_worker(return_logits: bool = True):
    """Both terms of Eq. 13, through `forward`, against the same row run whole.

    The decoder is the half this exercises. Its canvases are replicated, so what
    has to hold is that a canvas reading a cache gathered back out of `sp_size`
    slices sees what it saw when the cache was never split -- including the
    alignment tail, which the sharded arm's row carries and the reference's does
    not, and which no canvas is allowed to read.
    """
    from veomni.distributed.parallel_state import use_parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(world_size)

    model = _build_model(device)
    reference_batch = _build_row(model.config, device, sp_size=1)
    reference = model(**reference_batch, use_cache=False, return_logits=return_logits)
    reference.loss.backward()
    full = {
        "loss": float(reference.loss.detach()),
        "diffusion": float(reference.diffusion_loss.detach()),
        "ar": float(reference.encoder_ar_loss.detach()),
    }
    full_grads = _gradients(model)
    del model, reference

    with use_parallel_state(sharded):
        model = _build_model(device)
        batch = _build_row(model.config, device, sp_size=world_size)
        outputs = model(**batch, use_cache=False, return_logits=return_logits)
        outputs.loss.backward()
        shard_grads = _gradients(model)

    # The diffusion term is computed on replicated canvases, so it is not reduced
    # and every rank should already hold the whole of it. A term that had been
    # reduced would be `sp_size` times this, and one computed against a partial
    # cache would be some other number entirely.
    ulp = 2**-8
    for name, value in (
        ("diffusion_loss", float(outputs.diffusion_loss.detach())),
        ("encoder_ar_loss", float(outputs.encoder_ar_loss.detach())),
        ("loss", float(outputs.loss.detach())),
    ):
        expected = full[{"diffusion_loss": "diffusion", "encoder_ar_loss": "ar", "loss": "loss"}[name]]
        error = abs(value - expected) / max(abs(expected), 1e-9)
        assert error < 4 * ulp, f"[Rank {rank}] {name} is {value} against {expected} unsharded, off by {error:.1e}"

    assert set(shard_grads) == set(full_grads), f"[Rank {rank}] SP changed which parameters have gradients"
    worst, worst_name = 0.0, ""
    for name, expected in full_grads.items():
        averaged = shard_grads[name].clone()
        dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
        averaged /= world_size
        error = _relative_error(averaged, expected)
        if error > worst:
            worst, worst_name = error, name
    assert worst < 5e-2, (
        f"[Rank {rank}] {worst_name} is off by {worst:.2e} after averaging over the SP group, on the whole-model loss"
    )
    if not return_logits:
        assert outputs.logits is None, f"[Rank {rank}] chunked training must not materialize logits"

    # As in the encoder worker, the averaged comparison above is the one that sees a
    # missing `sp_size`; a comparison against one rank's un-averaged gradient is off
    # by roughly that factor whether or not the scale is applied, and so says nothing.
    dist.barrier()


def _repeated_kv_worker():
    """Two KV heads across four ranks, which is the branch GQA makes reachable.

    `_kv_repeats` returns 1 in every other worker here, so the arithmetic that has
    never run is the interesting one: with fewer KV heads than ranks the KV is
    duplicated to one per rank, and correctness then rests on that duplication
    landing in the same order as the contiguous split of the query heads -- rank
    r's query block has to fall inside the KV group rank r received. Reading the
    wrong group changes the numbers without changing any shape.

    Not hypothetical: the bench config has two global KV heads, so `sp_size = 4`
    already repeats.

    Four ranks and two KV heads, not two ranks and one. With a single KV head the
    duplication produces `[h0, h0]`, every rank holds the same head, and no
    ordering can be wrong -- the configuration cannot fail the test. Two heads
    repeated twice gives `[h0, h0, h1, h1]`, and ranks 2 and 3 hold a head that
    ranks 0 and 1 do not.

    Checked on the forward, and deliberately. Repeating doubles the KV head count the
    attention kernel sees, which changes its tiling and fp32 reduction order even
    though the pairwise dot products and softmax sums are unchanged -- so the
    output moves in its last bits from layer 0 rather than staying bit-exact as it
    does at `repeats == 1`. That reroutes tokens in the MoE layers (constraint 33), and backpropagation carries a rerouted token
    into every parameter *upstream* of it: measured at 3.4e-1 on
    `router.per_expert_scale` and 3.1e-1 on a `q_norm` three layers earlier. So
    gradients cannot discriminate here at any bound, and the bound below is set
    from the forward instead -- where a misaligned group is not a last-bit effect
    but a different head's output entirely. The gradient path is covered at
    `repeats == 1` by the two workers above, and the ordering itself by
    `test_a_group_stays_with_its_kv_head_when_the_kv_is_repeated`, which asserts
    it exactly on indices.
    """
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.models.transformers.diffusion_gemma.dg_sequence_parallel import _kv_repeats

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(world_size)

    kv_heads = 2
    model = _build_model(device, kv_heads=kv_heads)
    heads = model.config.text_config.num_attention_heads
    repeats = _kv_repeats(heads, kv_heads, world_size)
    assert repeats > 1, (
        f"this worker is meant to drive the repeat branch, and {heads} query heads over {kv_heads} KV "
        f"heads on {world_size} ranks does not ({repeats=})"
    )

    reference_batch = _build_row(model.config, device, sp_size=1)
    with torch.no_grad():
        full_hidden, _ = _encoder_and_ar_term(model, reference_batch)
        full_hidden = full_hidden.float()
        full_loss = float(model(**reference_batch, use_cache=False).loss)
    del model

    with use_parallel_state(sharded):
        model = _build_model(device, kv_heads=kv_heads)
        batch = _build_row(model.config, device, sp_size=world_size)
        with torch.no_grad():
            hidden, _ = _encoder_and_ar_term(model, batch)
            value = float(model(**batch, use_cache=False).loss)

    # The encoder states, on the real tokens of this rank's slice. A rank handed the
    # wrong KV copy attends to another group's keys, which is an O(1) change to
    # every state it produces -- so the bound has room for the ratio effect above
    # and none for a misalignment.
    real = int(batch["attention_mask"].sum())
    local = hidden.shape[1]
    start, stop = rank * local, min((rank + 1) * local, real)
    error = _relative_error(hidden.float()[:, : stop - start], full_hidden[:, start:stop])
    assert error < 1e-1, (
        f"[Rank {rank}] with {kv_heads} KV heads repeated {repeats} times over {world_size} ranks, the "
        f"sharded encoder states differ from the whole row by {error:.2e}"
    )

    # And the whole model, which is where `ulysses_decoder_split` repeats the cache's
    # KV as well.
    loss_error = abs(value - full_loss) / max(abs(full_loss), 1e-9)
    assert loss_error < 1e-1, (
        f"[Rank {rank}] the loss with the KV heads repeated is {value} against {full_loss} unsharded, "
        f"off by {loss_error:.1e}"
    )
    dist.barrier()


def _sum_over_shards_worker():
    """The factor itself, on arithmetic simple enough to state in closed form.

    The model-level tests can only see this factor through FSDP's average, which
    is the right thing to check but not a tight one -- it holds up to bf16 noise
    over 131 tensors. Here each rank scales the forward by `3*(rank+1)`, so the
    summed value and the per-rank backward factors are exact integers, and a
    wrong scale is a wrong integer rather than a tolerance question.
    """
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.models.transformers.diffusion_gemma.dg_sequence_parallel import sum_over_shards

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(world_size)

    with use_parallel_state(sharded):
        x = torch.tensor([2.0], device=device, requires_grad=True)
        rank_scale = 3.0 * float(rank + 1)
        summed = sum_over_shards(x * rank_scale)
        expected_forward = 3.0 * 2.0 * sum(range(1, world_size + 1))
        # Forward: every rank holds the group's total, which is what makes the
        # logged AR loss the whole row's rather than this shard's.
        assert float(summed.detach()) == pytest.approx(expected_forward), (
            f"[Rank {rank}] the forward gave {float(summed.detach())} for {world_size} ranks each "
            f"contributing 3*(rank+1)*2.0; it is not summing over the group"
        )
        summed.backward()

    # Backward: each rank hands FSDP `sp_size` times its *own* shard's derivative.
    # Averaging those locals must reproduce the derivative of the summed loss --
    # `sum_s 3*(s+1)`, not `3*sp_size` unless every shard happened to match.
    expected_whole = 3.0 * sum(range(1, world_size + 1))
    expected_local = rank_scale * world_size
    assert float(x.grad) == pytest.approx(expected_local), (
        f"[Rank {rank}] the local gradient is {float(x.grad)} where {expected_local} is needed "
        f"(`3*(rank+1)*sp_size`) so FSDP's average over {world_size} ranks preserves the whole "
        "loss's derivative"
    )
    averaged = x.grad.clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
    averaged /= world_size
    assert float(averaged) == pytest.approx(expected_whole), (
        f"[Rank {rank}] averaging the local gradients gives {float(averaged)} against the whole "
        f"loss's derivative {expected_whole}"
    )
    dist.barrier()


def _postforward_worker():
    """The trainer's reduction, which no other test here goes through.

    Every other worker calls the model directly, and that is how the shipped path
    came to be broken while all of them passed: `dllm_mean_global_loss` refused
    `sp_enabled` outright, so a run that got past the preflight died in
    `postforward` on the first micro-step. A model-level test cannot see that.

    What is asserted is the reduction's *value* against the same reduction at
    sp=1. The two `sp_size` factors it relies on -- the world-group all-reduce
    counting each sample once per SP peer, and `fsdp_size` carrying the Ulysses
    dimension -- cancel only if both are present, so getting either wrong moves
    this by exactly `sp_size`.
    """
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.trainer.dllm_trainer import DllmTrainer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    replicated, sharded = _both_topologies(world_size)

    def reduced(state, sp_size):
        """One arm's `postforward`, with the trainer reduced to what it reads."""
        with use_parallel_state(state):
            model = _build_model(device)
            batch = _build_row(model.config, device, sp_size=sp_size)
            outputs = model(**batch, use_cache=False)
            trainer = DllmTrainer.__new__(DllmTrainer)
            trainer.micro_batches_num_samples = trainer._supervised_sample_count(batch["loss_weight"])
            loss, metrics = DllmTrainer.postforward(trainer, outputs, batch)
            return float(loss.detach()), {k: float(v.detach()) for k, v in metrics.items()}

    # The reference arm still runs on two ranks, so its `fsdp_size` and its
    # all-reduced sample count are both 2 -- the same pair of factors, cancelling
    # for the ordinary reason. That is the point: SP must not change the number.
    plain_loss, plain_metrics = reduced(replicated, 1)
    sp_loss, sp_metrics = reduced(sharded, world_size)

    ulp = 2**-8
    error = abs(sp_loss - plain_loss) / max(abs(plain_loss), 1e-9)
    assert error < 8 * ulp, (
        f"[Rank {rank}] the reduced loss is {sp_loss} under sp={world_size} against {plain_loss} "
        f"without it, off by {error:.1e}. A factor of {world_size} here is the sample count or "
        "`fsdp_size` being counted once instead of twice"
    )
    for name in ("diffusion_loss", "encoder_ar_loss"):
        expected, actual = plain_metrics[name], sp_metrics[name]
        error = abs(actual - expected) / max(abs(expected), 1e-9)
        assert error < 8 * ulp, (
            f"[Rank {rank}] the reduced {name} is {actual} under sp={world_size} against {expected} "
            f"without it, off by {error:.1e}"
        )
    dist.barrier()


def _a_sharded_loss_is_still_refused_worker():
    """The default stays closed, or the flag is decoration.

    A plain dLLM computes its loss on the sharded sequence, so its sum is a shard
    and the `fsdp_size` factor has nothing to cancel against. That case must keep
    raising -- which is what makes DiffusionGemma's flag a claim about the model
    rather than a way of switching the check off.
    """
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.utils.loss_utils import dllm_mean_global_loss

    torch.cuda.set_device(dist.get_rank())
    _, sharded = _both_topologies(dist.get_world_size())

    with use_parallel_state(sharded):
        with pytest.raises(NotImplementedError, match="whole across the SP group"):
            dllm_mean_global_loss(torch.tensor(1.0), micro_batches_num_samples=1)
        # And the same call with the assertion made goes through.
        dllm_mean_global_loss(torch.tensor(1.0), micro_batches_num_samples=1, sum_spans_sp_group=True)
    dist.barrier()


def _recompute_worker(use_reentrant: bool = False):
    """The same run with gradient checkpointing on, which moves the collectives.

    Recompute replays a checkpointed layer from inside the autograd engine, on
    the engine's own thread. Every all-to-all in this model therefore runs twice
    and the second time is not on the thread that built the mesh -- and
    `DeviceMesh` keeps its *flattened* dimensions in a `threading.local`, so
    resolving the `sp` dimension succeeds in the forward and raises `Mesh
    dimension 'sp' does not exist` in the backward. That is why nothing here
    reads `sp_rank` or `sp_group`, and this is the test that says so: the
    configuration below is the recipe's own (checkpointing on, which is why the
    long rows fit at all), and without the real `ulysses` dimension it does not
    reach the optimizer.

    `use_reentrant` is passed rather than left to the default, which is the only
    setting this can run: the model refuses reentrant checkpointing outright,
    because the decoder reads an encoder cache the recompute would rebuild (see
    `test_diffusion_gemma.py` for that refusal). Non-reentrant replays inside a
    `saved_tensors_hook`, still on the autograd thread, so the thread-local is
    still met -- which is the point here.
    """
    from veomni.distributed.parallel_state import use_parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(world_size)

    with use_parallel_state(sharded):
        model = _build_model(device)
        model.gradient_checkpointing_enable({"use_reentrant": use_reentrant})
        batch = _build_row(model.config, device, sp_size=world_size)
        checkpointed = model(**batch, use_cache=False)
        checkpointed.loss.backward()
        recomputed_grads = _gradients(model)
        del model

        # Against the same arm with checkpointing off, so what is compared is one
        # setting rather than SP as a whole: recompute is meant to be exact here,
        # and both arms sum over the same shards in the same order.
        model = _build_model(device)
        plain = model(**batch, use_cache=False)
        plain.loss.backward()
        plain_grads = _gradients(model)

    error = abs(float(checkpointed.loss.detach()) - float(plain.loss.detach())) / max(
        abs(float(plain.loss.detach())), 1e-9
    )
    assert error < 2**-8, (
        f"[Rank {rank}] the checkpointed loss is {float(checkpointed.loss.detach())} against "
        f"{float(plain.loss.detach())} without checkpointing, off by {error:.1e}"
    )
    assert set(recomputed_grads) == set(plain_grads), f"[Rank {rank}] checkpointing changed which parameters trained"
    worst, worst_name = 0.0, ""
    for name, expected in plain_grads.items():
        error = _relative_error(recomputed_grads[name], expected)
        if error > worst:
            worst, worst_name = error, name
    assert worst < 5e-2, (
        f"[Rank {rank}] {worst_name} is off by {worst:.2e} between the recomputed backward and the "
        "one that kept its activations"
    )
    dist.barrier()


def _attn_dg_prefix_refusal_worker():
    """`attn_dg_prefix: false` is refused at model build, not in the decoder.

    An older version of this worker turned the flag off and expected the concat
    path to refuse during forward — shapes would line up on one rank's slice of
    the cache and the run would look healthy while conditioning on a fraction of
    the context. `sequence_parallel_preflight` now catches that at build time
    instead, before `_flat_seq_sdpa_mask` can allocate over the full encoder row.
    """
    from veomni.distributed.parallel_state import use_parallel_state

    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(dist.get_world_size())

    with use_parallel_state(sharded):
        with pytest.raises(ValueError, match="needs `attn_dg_prefix: true`"):
            _build_model(device, attn_dg_prefix=False)
    dist.barrier()


def _padded_mode_refusal_worker():
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.models.transformers.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
    from veomni.models.transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

    torch.cuda.set_device(dist.get_rank())
    _, sharded = _both_topologies(dist.get_world_size())

    config = DiffusionGemmaConfig.from_pretrained(TOY_CONFIG_PATH)
    config.canvas_batching = "flat_seq"
    with use_parallel_state(sharded):
        with pytest.raises(ValueError, match="needs `canvas_batching: flat_seq_pack`"):
            DiffusionGemmaForBlockDiffusion(config)
    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_two_ranks_agree_with_one_on_the_encoder_and_the_ar_term():
    from ..tools.launch_utils import torchrun

    torchrun(_equivalence_worker, world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
@pytest.mark.parametrize("return_logits", [True, False], ids=["materialized_logits", "chunked_diffusion_loss"])
def test_two_ranks_agree_with_one_on_the_whole_model(return_logits: bool):
    from functools import partial

    from ..tools.launch_utils import torchrun

    torchrun(partial(_whole_model_worker, return_logits), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 devices: two KV heads repeated twice")
def test_four_ranks_agree_with_one_when_the_kv_heads_are_repeated():
    from ..tools.launch_utils import torchrun

    torchrun(_repeated_kv_worker, world_size=4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_the_backward_carries_the_sp_size_factor():
    from ..tools.launch_utils import torchrun

    torchrun(_sum_over_shards_worker, world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_the_trainer_reduces_the_loss_to_the_same_number():
    from ..tools.launch_utils import torchrun

    torchrun(_postforward_worker, world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_a_loss_that_is_only_a_shard_is_still_refused():
    from ..tools.launch_utils import torchrun

    torchrun(_a_sharded_loss_is_still_refused_worker, world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_the_collectives_survive_being_replayed_by_recompute():
    from ..tools.launch_utils import torchrun

    torchrun(_recompute_worker, 2, False)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_attn_dg_prefix_is_refused_at_build_on_a_real_group():
    """The build-time gate, on a topology where `sp_enabled` is genuinely true.

    Worth both: the single-process suite pins the message; this one pins that
    the refusal runs after a real Ulysses mesh has been built.
    """
    from ..tools.launch_utils import torchrun

    torchrun(_attn_dg_prefix_refusal_worker, world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_a_padded_mode_refuses_to_build_on_a_real_group():
    """Same refusal the single-process tests cover, on a topology that really exists.

    Worth both: the single-process one pins the message, this one pins that
    `sp_enabled` is true when a real Ulysses mesh has been built -- the property
    every gate in the model hangs off.
    """
    from ..tools.launch_utils import torchrun

    torchrun(_padded_mode_refusal_worker, world_size=2)
