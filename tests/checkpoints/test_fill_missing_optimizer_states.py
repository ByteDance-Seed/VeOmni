# Regression test for ``_fill_missing_optimizer_states`` on the FSDP2 /
# ``ep_size=1`` save+load path.
#
# Why this test exists:
#
# The existing ``test_trainer_saveload`` runs ``qwen3_moe`` /
# ``deepseek_v3`` for only 5 training steps with a small global batch,
# so every expert reliably fires and AdamW populates state for every
# parameter — the missing-state path is never exercised. The production
# bug it should have caught (qwen3.5-35B-a3b VL h100x16 at step 100:
# save hung in DCP metadata broadcast with ``TypeError: unhashable
# type: 'TensorProperties'`` on one rank and a 10-min NCCL watchdog
# timeout on the others) only shows up when that path runs with
# placeholders for missing entries.
#
# Root cause and fix layer:
#
# - ``torch.distributed.checkpoint.metadata.TensorProperties`` is a plain
#   ``@dataclass`` (no ``frozen=True``); with the default ``eq=True``
#   Python sets ``__hash__`` to None → unhashable. It is held by
#   ``TensorWriteData`` which IS ``frozen=True`` and auto-generates
#   ``__hash__`` that cascades to its fields. Hashing anything that
#   transitively contains a ``TensorProperties`` therefore raises.
# - ``dcp.save``'s metadata broadcast unpickles a dict on receiving
#   ranks; that reconstruction hashes keys → crash.
# - VeOmni installs a one-time monkey-patch at module import that gives
#   ``TensorProperties`` an explicit ``__hash__`` over its immutable
#   fields. The patch is idempotent (guarded by ``__hash__ is None``)
#   so an upstream PyTorch fix will silently win.
#
# This test pins the full save+load contract: missing AdamW state for
# ``unused.weight`` is synthesized as a zero placeholder at save (#735),
# round-trips through DCP without the unhashable bug firing, and after
# load the recovered optimizer state has ``step=0`` (faithful resume)
# — NOT the ``step=1`` that ``_init_optim_state`` would have left if
# nothing on disk overrode it.

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.metadata import TensorProperties
from torch.distributed.fsdp import fully_shard


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.launch_utils import torchrun  # noqa: E402

# Importing this module installs the TensorProperties.__hash__ patch.
from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState  # noqa: E402
from veomni.distributed.parallel_state import init_parallel_state  # noqa: E402
from veomni.utils.device import get_device_type, get_torch_device  # noqa: E402


class _TinyModel(nn.Module):
    """Two independent linear blocks. Forward only routes through ``used``.

    ``unused`` lives in the optimizer's param_groups but never sees a
    gradient, so AdamW never creates state for it — exactly the
    missing-state condition the production bug needed.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.used = nn.Linear(hidden, hidden, bias=False)
        self.unused = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x)


def _build_fsdp2_model_and_optim(device: torch.device):
    model = _TinyModel(hidden=64).to(device)
    fully_shard(model.used)
    fully_shard(model.unused)
    fully_shard(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def _run_save_load():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_type = get_device_type()
    device = torch.device(f"{device_type}:{rank}")

    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        device_type=device_type,
    )

    # The monkey-patch must already be live (loading veomni.checkpoint did
    # that at import time). Assert it explicitly: this is the property that
    # makes ``dcp.save`` not hang.
    assert TensorProperties.__hash__ is not None, (
        "TensorProperties.__hash__ patch missing — dcp.save will hang in metadata broadcast"
    )

    model, optimizer = _build_fsdp2_model_and_optim(device)

    x = torch.randn(2, 64, device=device, dtype=next(model.parameters()).dtype)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Precondition: ``unused.weight`` got no gradient → no AdamW state.
    raw_sd = OptimizerState(model, optimizer).state_dict()
    assert "unused.weight" not in raw_sd["state"], (
        "Test precondition violated: unused.weight has optimizer state, "
        "so the missing-state branch will not run. Check the model setup."
    )

    # SAVE with ``fill_missing_optimizer_states=True`` (#735's behavior):
    # ``_fill_missing_optimizer_states`` synthesizes a zero placeholder for
    # ``unused.weight`` (step=0, zero exp_avg, zero exp_avg_sq). Without the
    # TensorProperties hash patch, DCP's broadcast of the global Metadata
    # would crash here.
    tmpdir = os.environ["VEOMNI_TEST_DCP_DIR"]
    dcp.save(
        {
            "model": ModelState(model),
            "optimizer": OptimizerState(model, optimizer, fill_missing_optimizer_states=True),
        },
        checkpoint_id=tmpdir,
    )
    dist.barrier()

    # Capture pre-load snapshot for comparison.
    used_param_before = dict(model.named_parameters())["used.weight"].full_tensor().detach().cpu()
    used_state_before = {
        k: v.full_tensor().detach().cpu() if hasattr(v, "full_tensor") else v.detach().cpu()
        for k, v in optimizer.state[dict(model.named_parameters())["used.weight"]].items()
    }

    # LOAD into a FRESH model + optimizer. NOTE: strict load (no
    # ``allow_partial_load``) — the placeholders synthesized at save make
    # every expected FQN present in the checkpoint, so strict checking
    # passes naturally. This is the resume-time semantic #735 was
    # designed to preserve.
    fresh_model, fresh_optimizer = _build_fsdp2_model_and_optim(device)
    dcp.load(
        {
            "model": ModelState(fresh_model),
            "optimizer": OptimizerState(fresh_model, fresh_optimizer),
        },
        checkpoint_id=tmpdir,
    )
    dist.barrier()

    # Trained ``used.weight`` should round-trip exactly.
    used_param_after = dict(fresh_model.named_parameters())["used.weight"].full_tensor().detach().cpu()
    torch.testing.assert_close(used_param_after, used_param_before)

    # Used Adam state should also round-trip.
    used_param_in_fresh = dict(fresh_model.named_parameters())["used.weight"]
    fresh_used_state = fresh_optimizer.state.get(used_param_in_fresh, {})
    assert fresh_used_state, "fresh optimizer has no state for used.weight after load"
    for k, v_before in used_state_before.items():
        v_after = fresh_used_state[k]
        v_after_cpu = (
            v_after.full_tensor().detach().cpu() if hasattr(v_after, "full_tensor") else v_after.detach().cpu()
        )
        torch.testing.assert_close(v_after_cpu, v_before, msg=f"optimizer.state[used.weight][{k}] mismatch after load")

    # The critical fidelity check: ``unused.weight`` should resume with
    # ``step == 0`` (matching the saved placeholder), NOT ``step == 1``
    # (which is what ``_init_optim_state`` primed onto the fresh optimizer
    # before the load overwrote it).
    unused_param_in_fresh = dict(fresh_model.named_parameters())["unused.weight"]
    fresh_unused_state = fresh_optimizer.state.get(unused_param_in_fresh, {})
    assert fresh_unused_state, "fresh optimizer should have placeholder state for unused.weight after load"

    step_tensor = fresh_unused_state["step"]
    step_value = step_tensor.item() if hasattr(step_tensor, "item") else int(step_tensor)
    assert step_value == 0, (
        f"unused.weight should resume with step=0 (faithful to the saved placeholder), got step={step_value}. "
        f"This indicates _init_optim_state's primed state leaked through — the placeholder synthesis didn't run, "
        f"or the load didn't overwrite the primed state."
    )

    # The placeholder moments should also be exactly zero.
    for moment_key in ("exp_avg", "exp_avg_sq"):
        v = fresh_unused_state[moment_key]
        v_cpu = v.full_tensor().detach().cpu() if hasattr(v, "full_tensor") else v.detach().cpu()
        assert torch.all(v_cpu == 0), f"placeholder unused.weight.{moment_key} should be all zeros after load"


@pytest.mark.skipif(not get_torch_device().is_available(), reason="requires an accelerator device")
@pytest.mark.skipif(
    get_torch_device().is_available() and get_torch_device().device_count() < 2,
    reason="requires >=2 accelerator devices for FSDP2 sharding",
)
def test_dcp_save_load_with_missing_optimizer_states():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["VEOMNI_TEST_DCP_DIR"] = tmpdir
        torchrun(_run_save_load, world_size=2)


def test_tensorproperties_hash_patch_fixes_prod_error():
    """CPU-only unit proof that the ``TensorProperties.__hash__`` patch
    addresses the exact prod failure mode shown in PR #794's stack trace.

    The end-to-end FSDP2 test above is necessarily small-scale and does
    not always trigger DCP's metadata-hash code path (the prod hang was
    at qwen3.5-35B-a3b VL h100x16 — the specific dict-with-unhashable-
    keys reconstruction inside ``_unpickler.load()`` only fires above a
    certain metadata size). This test pins the minimal repro:
    ``dict({TensorProperties: value})`` raises
    ``TypeError: unhashable type: 'TensorProperties'`` without the
    patch, and succeeds with it. That is exactly the operation that
    fires inside PyTorch DCP's broadcast unpickle.
    """
    # The veomni import at the top of this file already installed the
    # patch. Verify it's active.
    assert TensorProperties.__hash__ is not None, (
        "TensorProperties.__hash__ patch missing — dcp.save will fail on the prod path"
    )

    tp_bf16 = TensorProperties(
        dtype=torch.bfloat16,
        layout=torch.strided,
        requires_grad=False,
        memory_format=torch.contiguous_format,
        pin_memory=False,
    )
    # The exact operation that fails without the patch.
    d = {tp_bf16: "x"}
    assert d[tp_bf16] == "x"

    # Hash is equality-consistent (Python's dict / set contract).
    other = TensorProperties(
        dtype=torch.bfloat16,
        layout=torch.strided,
        requires_grad=False,
        memory_format=torch.contiguous_format,
        pin_memory=False,
    )
    assert hash(tp_bf16) == hash(other), "equal TensorProperties must hash equal"
    assert tp_bf16 == other

    # Different field values produce different hashes (no collision-by-default).
    tp_fp32 = TensorProperties(
        dtype=torch.float32,
        layout=torch.strided,
        requires_grad=False,
        memory_format=torch.contiguous_format,
        pin_memory=False,
    )
    assert hash(tp_bf16) != hash(tp_fp32), "different TensorProperties should hash differently"
