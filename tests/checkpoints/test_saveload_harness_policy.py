# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints import utils
from tests.checkpoints.test_trainer_saveload import _prepare_optimizer_for_restore


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("ep_size,expected", [(1, True), (2, False), (4, False)])
def test_checkpoint_command_selects_supported_save_mode(monkeypatch, ep_size, expected):
    monkeypatch.setattr(utils, "hf_local_or_remote", lambda path: path)
    monkeypatch.setattr(utils, "resolve_ops_overrides", lambda _name: [])
    command = utils.get_checkpoint_test_command("qwen3_moe", ep_size, nproc=4)
    assert f"--train.checkpoint.save_async {expected}" in command


@pytest.mark.parametrize("multi", [False, True])
def test_checkpoint_restore_rebuilds_only_multi_optimizer(monkeypatch, multi):
    import veomni.distributed.parallel_state as parallel_state

    calls = []
    monkeypatch.setattr(parallel_state, "use_parallel_state", lambda name: nullcontext(calls.append(("mesh", name))))
    optimizer = SimpleNamespace(
        _is_multi_optimizer=multi,
    )
    runtime = SimpleNamespace(
        optimizer=optimizer,
        model_name="policy",
        model=SimpleNamespace(zero_grad=lambda **kwargs: calls.append(("model_zero_grad", kwargs))),
        _build_optimizer_checked=lambda path: calls.append(("optimizer", path)),
        _build_lr_scheduler=lambda **kwargs: calls.append(("scheduler", kwargs)),
    )
    _prepare_optimizer_for_restore(runtime, 9, "/tmp/checkpoint")
    assert calls == (
        [
            ("mesh", "policy"),
            ("model_zero_grad", {"set_to_none": True}),
            ("optimizer", "/tmp/checkpoint"),
            ("scheduler", {"total_steps": 9}),
        ]
        if multi
        else []
    )
