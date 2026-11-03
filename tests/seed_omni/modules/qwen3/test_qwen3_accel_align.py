"""Full Qwen3 SeedOmni training-graph parity against monolithic ``Qwen3ForCausalLM``.

Verifies that the composed ``qwen3_text_encoder.encode -> qwen3_llm -> qwen3_text_encoder.decode``
graph under FSDP2 matches the monolithic ``Qwen3ForCausalLM`` reference on:
- forward loss,
- every parameter gradient, and
- AdamW parameter updates,
across ``SP=1`` (``dp=2, sp=1``) and ``SP=2`` (``dp=1, sp=2`` with an odd packed
sequence length of 7 tokens that exercises SP padding + output unpadding), for
both tied and untied output heads, on CPU/Gloo (FP32) as well as real GPU/NPU
accelerators (BF16 + FlashAttention-2 / SDPA over NCCL/HCCL).
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
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as HFQwen3Config

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.data_collator import MainCollator
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
from veomni.models.configuration_base import get_veomni_config_class
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.accelerated.utils.executor import TrainNodeRunner
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3.hf_layout import QWEN3_HF_LAYOUT
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import Qwen3ChatTemplate
from veomni.models.seed_omni.modules.qwen3.text_encoder.configuration import Qwen3TextEncoderConfig
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


class _FakeTokenizer:
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        del token
        return 1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


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


def _monolithic_and_split_qwen3(
    tied: bool,
    *,
    attn_implementation: str = "veomni_sdpa",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_npu import Qwen3ForCausalLM
    else:
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3ForCausalLM

    Qwen3Config = get_veomni_config_class(HFQwen3Config)
    ops = make_eager_ops_config(attn_implementation=attn_implementation)
    hf_cfg = Qwen3Config(
        **_tiny_qwen3_cfg(),
        tie_word_embeddings=tied,
        ops_implementation=ops,
    )
    hf_cfg._attn_implementation = hf_cfg.ops_config.attn_implementation

    torch.manual_seed(2026)
    monolithic = Qwen3ForCausalLM(hf_cfg).to(device=device, dtype=dtype)

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
        ops_implementation=make_eager_ops_config(attn_implementation=attn_implementation),
    )
    text_encoder = Qwen3TextEncoderAcc(te_cfg).to(device=device, dtype=dtype)
    text_encoder._chat_template = Qwen3ChatTemplate(_FakeTokenizer())
    llm = Qwen3LlmAcc(llm_cfg).to(device=device, dtype=dtype)

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
        dp_size=world,
        dp_shard_size=world,
        ulysses_size=1,
        dp_mode="fsdp2",
        device_type=device_type,
        name="sp1",
    )
    sp2_ps = _init_parallel_state(
        dp_size=1,
        dp_shard_size=1,
        ulysses_size=world,
        dp_mode="fsdp2",
        device_type=device_type,
        name="sp2",
    )
    _init_parallel_state(dp_size=world, ulysses_size=1, dp_mode="ddp", device_type=device_type, name="ref_sp1")

    shard_extra = {"mp_policy": mp_policy} if mp_policy is not None else {}
    loss_atol, loss_rtol = loss_tol
    grad_atol, grad_rtol = grad_tol
    step_atol, step_rtol = step_tol

    reports: dict[str, dict] = {}
    for attn_impl in attn_impls:
        for sp_size, ps_name, ps in ((1, "sp1", sp1_ps), (2, "sp2", sp2_ps)):
            for tied in (True, False):
                monolithic_ref, te, llm = _monolithic_and_split_qwen3(
                    tied=tied,
                    attn_implementation=attn_impl,
                    device=device,
                    dtype=dtype,
                )

                # Reference: packed forward + backward + AdamW step on the full 2-conversation batch.
                with use_parallel_state("ref_sp1"):
                    ref_batch = MainCollator()(_build_align_packed_features())
                    ref_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ref_batch.items()}
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
                    ref_opt = torch.optim.AdamW(
                        monolithic_ref.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False
                    )
                    ref_opt.step()
                    ref_stepped = {name: param.detach().clone() for name, param in monolithic_ref.named_parameters()}

                # Wrap SeedOmni modules with FSDP2 over ps.fsdp_mesh (dp_shard_sp).
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

                    loss = _run_qwen3_omni_graph_step(te, llm, local_convs)
                    loss.backward()

                    omni_grads = _collect_omni_as_source_dict(te, llm, grads=True)
                    te_opt = torch.optim.AdamW(te.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
                    llm_opt = torch.optim.AdamW(llm.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01, foreach=False)
                    te_opt.step()
                    llm_opt.step()
                    omni_stepped = _collect_omni_as_source_dict(te, llm, grads=False)

                loss_match = torch.allclose(loss.detach().float(), ref_loss.float(), atol=loss_atol, rtol=loss_rtol)
                grad_mismatches = [
                    key
                    for key, ref_g in ref_grads.items()
                    if not torch.allclose(omni_grads[key].float(), ref_g.float(), atol=grad_atol, rtol=grad_rtol)
                ]
                step_mismatches = [
                    key
                    for key, ref_w in ref_stepped.items()
                    if not torch.allclose(omni_stepped[key].float(), ref_w.float(), atol=step_atol, rtol=step_rtol)
                ]
                mode_key = (
                    f"sp{sp_size}_{'tied' if tied else 'untied'}"
                    if len(attn_impls) == 1
                    else f"{attn_impl}_sp{sp_size}_{'tied' if tied else 'untied'}"
                )
                reports[mode_key] = {
                    "loss_match": loss_match,
                    "loss": float(loss.detach()),
                    "ref_loss": float(ref_loss),
                    "grad_mismatches": grad_mismatches,
                    "step_mismatches": step_mismatches,
                }
    return reports


def _qwen3_fsdp2_sp_align_cpu_worker(rank: int, rendezvous: str, out_dir: str) -> None:
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


def _qwen3_fsdp2_sp_align_accelerator_worker(rank: int, rendezvous: str, out_dir: str) -> None:
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
def test_qwen3_omni_graph_matches_monolithic_qwen3_under_fsdp2_and_sp(tmp_path: Path):
    """Full Qwen3 SeedOmni graph (encode -> llm -> decode) under FSDP2 at SP=1 and
    SP=2 (with odd sequence length 7 requiring SP padding) matches monolithic
    Qwen3ForCausalLM on loss, every parameter gradient, and AdamW parameter updates."""
    out_dir = tmp_path / "fsdp2_sp_align_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _qwen3_fsdp2_sp_align_cpu_worker,
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


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) or runner_device_count() < 2,
    reason="requires >=2 CUDA or NPU devices",
)
def test_qwen3_omni_graph_matches_monolithic_qwen3_on_accelerator(tmp_path: Path):
    """Full Qwen3 SeedOmni graph under FSDP2 + BF16 + FlashAttention-2 / SDPA over
    real NCCL/HCCL collectives at SP=1 and SP=2 (with odd sequence length 7) matches
    monolithic Qwen3ForCausalLM on loss, gradients, and AdamW updates."""
    out_dir = tmp_path / "fsdp2_sp_align_acc"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _qwen3_fsdp2_sp_align_accelerator_worker,
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
