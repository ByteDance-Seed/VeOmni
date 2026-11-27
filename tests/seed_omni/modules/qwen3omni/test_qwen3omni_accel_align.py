"""Full Qwen3-Omni thinker-graph parity against monolithic ``Qwen3OmniMoeThinkerForConditionalGeneration``.

The thinker graph is ``qwen3omni_text_encoder.encode -> qwen3omni_llm ->
qwen3omni_text_encoder.decode``. This verifies that composing it under FSDP2
reproduces the monolithic model on:

- forward loss,
- every parameter gradient, and
- AdamW parameter updates,

with an odd packed sequence length of 7 tokens so SP padding and output
unpadding are exercised. Two modes are compared:

- ``sp1`` — ``dp=2, sp=1``, each rank takes one conversation, router objective
  off. With nothing batch-wide in the loss this is a true equivalence check for
  the FSDP and sharding plumbing.
- ``sp2_router`` — ``dp=1, sp=2``, both ranks hold the same packed batch, router
  objective on. This is the case the graph hook's router gather exists for.

The router objective is off in the SP=1 mode because it is the one quantity a
data-parallel split cannot reproduce: it is a product of two per-batch means, so
two ranks holding one conversation each compute a different denominator than a
single run over both (measured at up to 2.4x on the gradients). Leaving it on
there would compare two different objectives and call the difference a bug.

The audio and vision modules are absent on purpose: the alignment batch here is
text-only, so their towers never enter the graph. They have their own tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.training_utils import make_eager_ops_config
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.models.configuration_base import get_veomni_config_class
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.base.sp_shard import sp_local_len, sp_valid_in_window
from veomni.models.seed_omni.modules.qwen3omni.hf_layout import QWEN3OMNI_HF_LAYOUT
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.chat_template import Qwen3OmniChatTemplate
from veomni.models.seed_omni.modules.qwen3omni.text_encoder.configuration import Qwen3OmniTextEncoderConfig
from veomni.models.seed_omni.utils.conversation import _OMNI_TOKEN_LEN, ConversationItem
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")
_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")

_GRAPH_EDGES = [
    {"from": "qwen3omni_text_encoder.encode", "to": "qwen3omni_llm"},
    {"from": "qwen3omni_llm", "to": "qwen3omni_text_encoder.decode"},
    {"from": "qwen3omni_text_encoder.decode", "to": "end"},
]

# Packed lengths 3 + 4 = 7 tokens (odd -> exercises SP=2 padding on both
# qwen3omni_text_encoder.encode and qwen3omni_llm.forward).
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


class _FakeTokenizer:
    """Ids for the ChatML markers only; the batch below is already tokenized."""

    eos_token_id = 0
    _MARKERS = {
        "<|im_end|>": 1,
        "<|im_start|>": 2,
        "system": 3,
        "user": 4,
        "assistant": 5,
        "<|audio_start|>": 6,
        "<|audio_end|>": 7,
        "<|audio_pad|>": 8,
        "<|image_pad|>": 9,
        "<|video_pad|>": 10,
        "<tts_text_bos>": 11,
        "<tts_text_eod>": 12,
        "<tts_pad>": 13,
    }

    def convert_tokens_to_ids(self, token):
        return self._MARKERS.get(token, 0)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(c) % 50 + 14 for c in text]}

    def decode(self, ids):
        del ids
        return ""

    def __len__(self):
        return 256


def _tiny_text_config(*, router: bool) -> dict:
    return {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 128,
        "output_router_logits": router,
        "router_aux_loss_coef": 0.01 if router else 0.0,
    }


def _thinker_config(*, router: bool) -> dict:
    """The upstream builder wants a full thinker config, not only ``text_config``.

    Vision and audio sub-configs are kept at their minimal defaults because the
    alignment batch is text-only — the towers are constructed (the layout's
    config builder needs every sub-config present) but never called.
    """
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
        Qwen3OmniMoeVisionEncoderConfig,
    )

    return {
        "text_config": _tiny_text_config(router=router),
        "audio_config": Qwen3OmniMoeAudioEncoderConfig().to_dict(),
        "vision_config": Qwen3OmniMoeVisionEncoderConfig(depth=1, hidden_size=32, out_hidden_size=64).to_dict(),
        "position_id_per_seconds": 13,
    }


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


def _monolithic_and_split_qwen3omni(
    *,
    attn_implementation: str = "veomni_sdpa",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    router: bool = True,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    ThinkerConfig = get_veomni_config_class(Qwen3OmniMoeThinkerConfig)
    ops = make_eager_ops_config(attn_implementation=attn_implementation)
    thinker_cfg = ThinkerConfig(**_thinker_config(router=router), ops_implementation=ops)
    thinker_cfg.text_config._attn_implementation = ops.attn_implementation
    thinker_cfg.text_config.ops_implementation = ops.to_dict()

    torch.manual_seed(2026)
    monolithic = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_cfg).to(device=device, dtype=dtype)

    TextEncoderAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3omni_text_encoder"]()
    LlmAcc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3omni_llm"]()
    LlmConfig = OMNI_CONFIG_REGISTRY["qwen3omni_llm"]()

    te_cfg = Qwen3OmniTextEncoderConfig(
        vocab_size=thinker_cfg.text_config.vocab_size,
        hidden_size=thinker_cfg.text_config.hidden_size,
        tie_word_embeddings=False,
        lm_head_bias=False,
        ops_implementation=make_eager_ops_config(),
    )
    llm_cfg = LlmConfig(
        text_config=_tiny_text_config(router=router),
        spatial_merge_size=thinker_cfg.vision_config.spatial_merge_size,
        position_id_per_seconds=thinker_cfg.position_id_per_seconds,
        ops_implementation=make_eager_ops_config(attn_implementation=attn_implementation),
    )
    text_encoder = TextEncoderAcc(te_cfg).to(device=device, dtype=dtype)
    text_encoder._chat_template = Qwen3OmniChatTemplate(_FakeTokenizer())
    llm = LlmAcc(llm_cfg).to(device=device, dtype=dtype)

    # Route the monolithic thinker's tensors onto the two composed modules the
    # same way an offline checkpoint split does, so a mismatch here is a layout
    # bug rather than a test-only difference.
    #
    # ``Qwen3OmniMoeThinkerForConditionalGeneration`` is the *inner* half of a
    # checkpoint: its keys are ``model.embed_tokens.weight`` and ``lm_head.weight``,
    # without the ``thinker.`` prefix the outer
    # ``Qwen3OmniMoeForConditionalGeneration`` adds. The layout describes the outer
    # checkpoint, so re-add the prefix rather than teaching the layout a second
    # spelling.
    te_sd = {}
    llm_sd = {}
    for key, val in monolithic.state_dict().items():
        routed = QWEN3OMNI_HF_LAYOUT.route(f"thinker.{key}")
        if routed is None:
            continue
        mod_name, mod_key = routed
        if mod_name == "qwen3omni_text_encoder":
            te_sd[mod_key] = val.detach().clone()
        elif mod_name == "qwen3omni_llm":
            llm_sd[mod_key] = val.detach().clone()
    text_encoder.load_state_dict(te_sd, strict=True)
    llm.load_state_dict(llm_sd, strict=True)
    return monolithic, text_encoder, llm


def _run_qwen3omni_graph_step(
    text_encoder: torch.nn.Module,
    llm: torch.nn.Module,
    conversation_list: list[list[ConversationItem]],
) -> torch.Tensor:
    omni_cfg = OmniConfig(
        _module_entries={
            "qwen3omni_text_encoder": {"model_path": "qwen3omni_text_encoder"},
            "qwen3omni_llm": {"model_path": "qwen3omni_llm"},
        },
        training_graphs={"default": _GRAPH_EDGES},
        generation_graphs={},
    )
    omni = OmniModel(
        omni_cfg,
        {"qwen3omni_text_encoder": text_encoder, "qwen3omni_llm": llm},
    )
    out = omni.forward({"conversation_list": conversation_list}, node_runner=TrainNodeRunner())
    return out["loss"]


def _collect_omni_as_source_dict(
    text_encoder: torch.nn.Module,
    llm: torch.nn.Module,
    *,
    grads: bool,
) -> dict[str, torch.Tensor]:
    def _full(t: torch.Tensor) -> torch.Tensor:
        return t.full_tensor().detach().clone() if isinstance(t, DTensor) else t.detach().clone()

    out: dict[str, torch.Tensor] = {}
    for mod_name, mod in (("qwen3omni_text_encoder", text_encoder), ("qwen3omni_llm", llm)):
        for name, param in mod.named_parameters():
            src_key = QWEN3OMNI_HF_LAYOUT.source_key(mod_name, name)
            assert src_key is not None, (mod_name, name)
            target = param.grad if grads else param
            assert target is not None, (mod_name, name)
            out[src_key] = _full(target)
    return out


def _monolithic_reference(
    monolithic: torch.nn.Module,
    device: torch.device,
    *,
    collect_router: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Reference loss / grads / post-AdamW weights on the full 2-sample batch.

    Driven through ``monolithic.model`` (the bare thinker text model) rather than
    the full ``Qwen3OmniMoeThinkerForConditionalGeneration.forward``: that one
    asserts on precomputed ``image_mask`` / ``video_mask`` / ``audio_mask`` and
    re-runs the towers, which a text-only batch has no use for.

    The packed inputs come from the same production packer the graph uses, run
    with ``sp_size == 1`` so it sees the whole batch. That keeps this comparison
    about FSDP and sequence parallelism — the packer's own packing, positions and
    ``cu_seqlens`` are covered by their own tests, and rebuilding them here would
    compare two hand-written layouts instead of one contract.

    The loss recomposes what ``Qwen3OmniLlmAccelerated.forward_post`` contributes:
    the lm-head CE plus the router-balancing term, which this family computes
    inside the graph hook rather than on the head path.
    """
    from veomni.models.loss_utils import ForCausalLMLoss
    from veomni.models.seed_omni.modules.qwen3omni.llm.modeling import (
        pack_qwen3omni_conversations_for_forward,
    )

    conversations = _build_align_conversation_list()
    # Embed the text rows the way ``qwen3omni_text_encoder.encode`` would, so the
    # carrier reaches the packer in the shape it expects. Not under ``no_grad``:
    # the composed graph trains the embedding table through this lookup, so the
    # reference has to build the same gradient for it.
    for sample in conversations:
        for item in sample:
            item.value = monolithic.model.embed_tokens(item.value.to(device))

    packed = pack_qwen3omni_conversations_for_forward(
        conversations,
        device,
        monolithic.config.vision_config.spatial_merge_size,
        monolithic.config.position_id_per_seconds,
    )
    out = monolithic.model(
        inputs_embeds=packed["inputs_embeds"],
        attention_mask=None,
        position_ids=packed["position_ids"],
        use_cache=False,
        output_router_logits=collect_router,
        cu_seq_lens_q=packed["cu_seq_lens"],
        cu_seq_lens_k=packed["cu_seq_lens"],
        max_length_q=packed["max_length"],
        max_length_k=packed["max_length"],
    )

    # The decode head shifts *within* each sample and masks the boundary, so a
    # sample's last position never scores the next sample's first token. Build the
    # same labels here rather than shifting the concatenated batch, which would add
    # one cross-sample prediction per boundary.
    sample_labels = []
    boundaries = []
    total = 0
    for sample in conversations:
        concat = torch.cat([item.meta["labels"] for item in sample], dim=0).to(device)
        sample_labels.append(concat)
        total += concat.numel()
        boundaries.append(total)
    flat_labels = torch.cat(sample_labels, dim=0)
    # ``shift_labels[t]`` is the target for position ``t``: labels[t+1], with a
    # sample's final position masked out.
    shift_labels = torch.full_like(flat_labels, -100)
    shift_labels[:-1] = flat_labels[1:]
    for boundary in boundaries[:-1]:
        shift_labels[boundary - 1] = -100
    shift_labels = shift_labels.unsqueeze(0)
    logits = monolithic.lm_head(out.last_hidden_state)
    loss, _, _ = ForCausalLMLoss(
        logits=logits,
        labels=None,
        shift_labels=shift_labels,
        vocab_size=logits.size(-1),
    )
    if collect_router and out.router_logits:
        # ``monolithic.load_balancing_loss`` is the partial the model built, so it
        # already carries the ops-selected backend; calling the bare helper would
        # need that ``op`` spelled out again.
        aux = monolithic.load_balancing_loss(
            out.router_logits,
            monolithic.config.text_config.num_experts,
            monolithic.config.text_config.num_experts_per_tok,
            None,
        )
        loss = loss + aux * monolithic.config.text_config.router_aux_loss_coef

    loss_value = loss.detach().clone()
    loss.backward()
    # ``named_parameters`` and not ``state_dict``: the latter returns detached
    # tensors whose ``.grad`` is always ``None``, which would leave both comparison
    # loops below with nothing to check.
    grads = {
        name: param.grad.detach().clone() for name, param in monolithic.named_parameters() if param.grad is not None
    }
    opt = torch.optim.AdamW(monolithic.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
    opt.step()
    # Keyed by the same names, and limited to the ones that received a gradient:
    # the monolithic thinker also owns the audio and vision towers, which a
    # text-only batch never touches and which the composed modules do not have.
    params_after = dict(monolithic.named_parameters())
    stepped = {name: params_after[name].detach().clone() for name in grads}
    return loss_value, grads, stepped


def _run_alignment_matrix(
    *,
    rank: int,
    world: int,
    device_type: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_impls: tuple[str, ...],
    mp_policy: MixedPrecisionPolicy | None,
    loss_tol: tuple[float, float],
    grad_tol: tuple[float, float],
    step_tol: tuple[float, float],
) -> dict[str, dict]:
    sp1_ps = _init_parallel_state(
        dp_size=world, dp_shard_size=world, ulysses_size=1, dp_mode="fsdp2", device_type=device_type, name="sp1"
    )
    sp2_ps = _init_parallel_state(
        dp_size=1, dp_shard_size=1, ulysses_size=world, dp_mode="fsdp2", device_type=device_type, name="sp2"
    )
    sp2_skip_ps = _init_parallel_state(
        dp_size=1,
        dp_shard_size=1,
        ulysses_size=world,
        dp_mode="fsdp2",
        device_type=device_type,
        skip_sp_gather_between_modules=True,
        name="sp2_skip_gather",
    )
    _init_parallel_state(dp_size=world, ulysses_size=1, dp_mode="ddp", device_type=device_type, name="ref_sp1")

    shard_extra = {"mp_policy": mp_policy} if mp_policy is not None else {}
    loss_atol, loss_rtol = loss_tol
    grad_atol, grad_rtol = grad_tol
    step_atol, step_rtol = step_tol

    reports: dict[str, dict] = {}
    # ``(sp_size, parallel-state name, parallel state, router objective on)``.
    #
    # The router term is what forces the SP=2 case to need a gather, so it has to
    # be covered there with the objective on. Spelled out per mode rather than
    # switched globally because the objective is *not* reproducible across a
    # data-parallel split: it is a product of two per-batch means, so two ranks
    # holding one conversation each compute a different denominator than one run
    # over both. At SP=1 (`dp=2`), the router objective is therefore off; at SP=2
    # (`dp=1`), the same packed batch is sharded instead of split, so the router
    # term stays comparable and is checked with `router=True`.
    #
    # ``sp2_skip_gather`` runs the same SP=2 batch through the shard handoff
    # instead of the gather: every producer writes only this rank's window onto
    # the carrier and the backbone concatenates from there. It has to match the
    # gathering path exactly, which is the only thing that makes the cheaper
    # handoff safe to turn on.
    modes = [
        (1, "sp1", sp1_ps, False),
        (2, "sp2", sp2_ps, True),
        (2, "sp2_skip_gather", sp2_skip_ps, True),
    ]
    for attn_impl in attn_impls:
        for sp_size, ps_name, ps, router in modes:
            with use_parallel_state("ref_sp1"):
                monolithic_ref, _, _ = _monolithic_and_split_qwen3omni(
                    attn_implementation=attn_impl, device=device, dtype=dtype, router=router
                )
                ref_loss, ref_grads, ref_stepped = _monolithic_reference(monolithic_ref, device, collect_router=router)
                # The reference is keyed by ``named_parameters``, so its names carry
                # the bare ``model.`` / ``lm_head.`` spelling of the inner thinker.
                # The composed modules are keyed by HF source key, which is the
                # outer checkpoint's -- ``thinker.``-prefixed. Normalize once so a
                # mismatch means a value differs, not that a key was unreachable.
                ref_grads = {f"thinker.{k}": v for k, v in ref_grads.items()}
                ref_stepped = {f"thinker.{k}": v for k, v in ref_stepped.items()}

            _, te, llm = _monolithic_and_split_qwen3omni(
                attn_implementation=attn_impl, device=device, dtype=dtype, router=router
            )

            with use_parallel_state(ps_name):
                fully_shard(te.embed_tokens, mesh=ps.fsdp_mesh, reshard_after_forward=True, **shard_extra)
                fully_shard(te, mesh=ps.fsdp_mesh, **shard_extra)
                for layer in llm.language_model.layers:
                    fully_shard(layer, mesh=ps.fsdp_mesh, reshard_after_forward=True, **shard_extra)
                fully_shard(llm, mesh=ps.fsdp_mesh, **shard_extra)

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

                loss = _run_qwen3omni_graph_step(te, llm, local_convs)
                if ps.skip_sp_gather_between_modules:
                    # Shard handoff: every value on the carrier holds only this
                    # rank's window, so the row total is the window's real rows —
                    # not the full span, which is what the gathering modes leave.
                    rows = sum(item.value.size(0) for conv in local_convs for item in conv)
                    assert rows == sp_valid_in_window(7, rank, ps.sp_size), (ps_name, rank, rows)
                    assert sp_local_len(7, ps.sp_size) == 4
                else:
                    expected_rows = sum(int(item.meta["labels"].numel()) for conv in local_convs for item in conv)
                    assert sum(item.value.size(0) for conv in local_convs for item in conv) == expected_rows
                loss.backward()

                omni_grads = _collect_omni_as_source_dict(te, llm, grads=True)
                te_opt = torch.optim.AdamW(te.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
                llm_opt = torch.optim.AdamW(llm.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
                te_opt.step()
                llm_opt.step()
                omni_stepped = _collect_omni_as_source_dict(te, llm, grads=False)

            # Every quantity is comparable in both modes, because the router term is
            # the one thing a data-parallel split cannot reproduce: it is a product
            # of two per-batch means, so two ranks holding one conversation each
            # compute a different denominator than one run over both. That mode
            # therefore runs with the objective off, which leaves nothing batch-wide
            # in the loss and makes SP=1 a true equivalence check for the FSDP and
            # SP plumbing. SP=2 shards the same packed batch instead of splitting
            # it, so it runs with the objective on — the case the graph hook's
            # gather exists for.
            loss_match = torch.allclose(loss.detach().float(), ref_loss.float(), atol=loss_atol, rtol=loss_rtol)
            # Guard against a comparison that checked nothing. The reference is read
            # off live parameters, and an empty or unreachable key set would make
            # both loops below pass trivially — which is exactly how this file once
            # "passed" while comparing nothing at all.
            assert ref_grads, "the reference produced no gradients, so nothing would be compared"
            assert set(ref_grads) == set(omni_grads), (
                "gradient key sets differ; one side would compare vacuously",
                sorted(set(ref_grads) - set(omni_grads))[:5],
                sorted(set(omni_grads) - set(ref_grads))[:5],
            )
            assert set(ref_stepped) == set(omni_stepped), "weight key sets differ"

            # Relative, not absolute. These are gradients of a tiny random model, so
            # many entries sit around 1e-5 and an absolute tolerance would call
            # float-noise differences failures while letting a genuinely wrong
            # large gradient through. ``rtol`` is the meaningful yardstick, with a
            # small absolute floor so an all-zero pair still compares.
            def _close(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> bool:
                return torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)

            grad_mismatches = [
                key for key, ref_g in ref_grads.items() if not _close(omni_grads[key], ref_g, grad_rtol, grad_atol)
            ]
            step_mismatches = [
                key for key, ref_w in ref_stepped.items() if not _close(omni_stepped[key], ref_w, step_rtol, step_atol)
            ]
            suffix = "_router" if router else ""
            mode_key = f"{ps_name}{suffix}" if len(attn_impls) == 1 else f"{attn_impl}_{ps_name}{suffix}"
            reports[mode_key] = {
                "loss_match": loss_match,
                "router": router,
                "loss": float(loss.detach()),
                "ref_loss": float(ref_loss),
                "grad_mismatches": grad_mismatches,
                "step_mismatches": step_mismatches,
            }
    return reports


def _qwen3omni_fsdp2_sp_align_cpu_worker(rank: int, rendezvous: str, out_dir: str) -> None:
    world = 2
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        reports = _run_alignment_matrix(
            rank=rank,
            world=world,
            device_type="cpu",
            device=torch.device("cpu"),
            dtype=torch.float32,
            attn_impls=("veomni_sdpa",),
            mp_policy=None,
            loss_tol=(1e-5, 1e-5),
            grad_tol=(1e-4, 1e-4),
            step_tol=(1e-4, 1e-4),
        )
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


def _qwen3omni_fsdp2_sp_align_accelerator_worker(rank: int, rendezvous: str, out_dir: str) -> None:
    world = 2
    device_type = get_device_type()
    get_torch_device().set_device(rank)
    device = torch.device(f"{device_type}:{rank}")
    dist.init_process_group(get_dist_comm_backend(), init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        has_fa2 = IS_NPU_AVAILABLE or importlib.util.find_spec("flash_attn") is not None
        attn_impls = ("flash_attention_2", "veomni_sdpa") if has_fa2 else ("veomni_sdpa",)
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        reports = _run_alignment_matrix(
            rank=rank,
            world=world,
            device_type=device_type,
            device=device,
            dtype=torch.bfloat16,
            attn_impls=attn_impls,
            mp_policy=mp_policy,
            loss_tol=(5e-3, 5e-3),
            grad_tol=(2e-2, 2e-2),
            step_tol=(1e-2, 1e-2),
        )
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
def test_qwen3omni_thinker_graph_matches_monolithic_under_fsdp2_and_sp(tmp_path: Path):
    """Thinker graph under FSDP2 at SP=1 and SP=2 matches the monolithic thinker
    on loss, every parameter gradient, and AdamW parameter updates."""
    out_dir = tmp_path / "fsdp2_sp_align_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _qwen3omni_fsdp2_sp_align_cpu_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert set(reports) == {"sp1", "sp2_router", "sp2_skip_gather_router"}
        for mode, report in reports.items():
            assert report["loss_match"], (rank, mode, report)
            assert not report["grad_mismatches"], (rank, mode, report)
            assert not report["step_mismatches"], (rank, mode, report)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < 2,
    reason="requires >=2 CUDA or NPU devices",
)
def test_qwen3omni_thinker_graph_matches_monolithic_on_accelerator(tmp_path: Path):
    """Same comparison in BF16 over real NCCL/HCCL collectives with the fused
    attention backends, at SP=1 and SP=2."""
    out_dir = tmp_path / "fsdp2_sp_align_acc"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _qwen3omni_fsdp2_sp_align_accelerator_worker,
        args=(str(out_dir / "rendezvous"), str(out_dir)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        reports = json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        assert reports
        for mode, report in reports.items():
            assert report["loss_match"], (rank, mode, report)
            assert not report["grad_mismatches"], (rank, mode, report)
            assert not report["step_mismatches"], (rank, mode, report)
