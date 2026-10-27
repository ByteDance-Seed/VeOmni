"""Data providers must explicitly own exact checkpoint semantics."""

from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.callbacks.global_state_callback import (
    _exact_checkpoint_contract,
    _validate_exact_loader,
    validate_resumed_prefetch_is_drained,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Contract:
    def __init__(self):
        self.calls = []

    def install(self):
        self.calls.append("install")

    def checkpoint_identity(self):
        return "test-provider-v1"

    def validate_cursor(self, trainer, cursor=None, *, require_cursor=False):
        self.calls.append((trainer, cursor, require_cursor))

    def validate_before_checkpoint(self, trainer, state=None):
        self.calls.append((trainer, state))


def _trainer(contract):
    return SimpleNamespace(
        train_dataloader=SimpleNamespace(
            _veomni_exact_checkpoint_contract=contract, state_dict=lambda: {}, load_state_dict=lambda state: None
        ),
        model=SimpleNamespace(optimizer=SimpleNamespace(_ple_file_mode_enabled=True)),
    )


def test_contract_dispatches_cursor_and_save_validation():
    contract = Contract()
    trainer = _trainer(contract)
    _exact_checkpoint_contract(trainer).install()
    _validate_exact_loader(trainer, {"cursor": 1}, require_cursor=True)
    validate_resumed_prefetch_is_drained(trainer, "step")
    assert contract.calls == ["install", (trainer, {"cursor": 1}, True), (trainer, "step")]


@pytest.mark.parametrize("contract", [None, object(), SimpleNamespace(install=lambda: None)])
def test_unsupported_data_provider_is_rejected(contract):
    with pytest.raises(RuntimeError, match="supported dataloader checkpoint contract"):
        _exact_checkpoint_contract(_trainer(contract), require_identity=True)


def test_empty_cursor_is_rejected_before_provider_validation():
    contract = Contract()
    with pytest.raises(RuntimeError, match="nonempty dataloader cursor"):
        _validate_exact_loader(_trainer(contract), {}, require_cursor=True)
    assert not contract.calls


def test_invalid_contract_identity_is_rejected():
    contract = Contract()
    contract.checkpoint_identity = lambda: ""
    with pytest.raises(RuntimeError, match="invalid identity"):
        _exact_checkpoint_contract(_trainer(contract), require_identity=True)


def test_disabled_file_profile_leaves_ordinary_loader_untouched():
    trainer = SimpleNamespace(model=None, train_dataloader=None)
    validate_resumed_prefetch_is_drained(trainer)


def test_provider_validation_failure_is_propagated():
    contract = Contract()

    def fail(trainer, state):
        raise RuntimeError("injected provider failure")

    contract.validate_before_checkpoint = fail
    with pytest.raises(RuntimeError, match="injected provider failure"):
        validate_resumed_prefetch_is_drained(_trainer(contract), "step")
