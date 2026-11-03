"""Accelerator gate for release-head loading and weighted FSDP2 backward."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.arguments import MixedPrecisionConfig
from veomni.data.decision.collator import DecisionCollator
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
from veomni.models.checkpoint.weights import init_empty_weights
from veomni.models.decision import get_decision_model, save_clef_release
from veomni.models.decision.models.clef.modeling import (
    ClefDecisionModel,
    _load_head_module,
    _reshape_scalar_parameters,
)
from veomni.trainer.decision_trainer import DecisionTrainer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


HEAD_CONFIG = {"hidden_size": 4}
_RELEASE_HEAD_SOURCE = """
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class EncodedQuestion:
    spans: tuple[tuple[int, int], ...]

class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden, input_ids, attention_mask, records, output_weight):
        outputs = []
        for batch_index, record in enumerate(records):
            questions = []
            for question in record.questions:
                values = [
                    self.projection(hidden[batch_index, start:end].mean(0)).squeeze(-1)
                    for start, end in question.option_spans
                ]
                questions.append(torch.stack(values) * self.scale)
            outputs.append(questions)
        return outputs
"""


class _ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _ToyFoundation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyBackbone()
        self.lm_head = torch.nn.Linear(4, 16, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))

    def get_metadata_collate_func(self):
        return None

    def get_position_id_func(self):
        return None


def _write_release_head(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "joint_schema_model.py").write_text(_RELEASE_HEAD_SOURCE)
    (path / "joint_head_config.json").write_text(json.dumps(HEAD_CONFIG))
    head = _load_head_module(path).JointSchemaHead(**HEAD_CONFIG)
    save_file(head.state_dict(), path / "joint_head.safetensors")
    return head


def _batch(rank, packed=False):
    samples = []
    for index in range(rank + 2 if packed else 1):
        samples.append(
            {
                "input_ids": torch.tensor([1 + rank, 2 + rank, 3 + rank, 4 + rank + index]),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "decision_meta": {
                    "question_spans": [(0, 1)],
                    "option_spans": [[(1, 2), (2, 3)]],
                    "option_ids": [("a", "b")],
                    "kinds": [1],
                    "counts": [1],
                },
                "decision_supervision": {
                    "counts": torch.tensor([2]),
                    "targets": torch.tensor([[0.2, 0.8]]),
                    "weights": torch.tensor([0.5 if rank == 0 else 2.0]),
                    "ordinal_mask": torch.tensor([False]),
                    "keys": (("a", "b"),),
                    "question_keys": ("q",),
                },
            }
        )
    return DecisionCollator()(samples)


def _build_release_model(directory, *, meta):
    context = init_empty_weights() if meta else torch.no_grad()
    with context:
        foundation = _ToyFoundation()
        head = _load_head_module(directory).JointSchemaHead(**HEAD_CONFIG)
    scalar_names = _reshape_scalar_parameters(head)
    return ClefDecisionModel(
        foundation,
        head,
        Path(directory) / "joint_head.safetensors",
        head_config=HEAD_CONFIG,
        source_dir=directory,
        scalar_head_parameters=scalar_names,
    )


def _worker(rank, directory, packed, endpoint):
    get_torch_device().set_device(rank)
    dist.init_process_group(get_dist_comm_backend(), store=endpoint.connect(), rank=rank, world_size=2)
    try:
        ps = _init_parallel_state(dp_size=2, dp_shard_size=2, device_type=get_device_type())
        model = _build_release_model(directory, meta=True)
        model = parallelize_model_fsdp2(
            model,
            weights_path=directory,
            mixed_precision=MixedPrecisionConfig(enable=False),
            init_device="meta",
            broadcast_model_weights_from_rank0=True,
            additional_weight_sources=model.get_additional_weight_sources(),
        )
        trainer = object.__new__(DecisionTrainer)
        trainer.device = torch.device(get_device_type())
        trainer.model = SimpleNamespace(parallel_state=ps)
        trainer.loss = get_decision_model("clef").build_loss("soft_ce")
        batch = _batch(rank, packed)
        trainer.prepare_loss_reduction([batch])
        loss, _, _ = trainer.postforward(model(batch), batch)
        loss.backward()
        norm = model.clip_grad_norm_(1e6)

        reference = _build_release_model(directory, meta=False).to(get_device_type())
        backbone = load_file(str(Path(directory) / "model.safetensors"))
        head = load_file(str(Path(directory) / "joint_head.safetensors"))
        head = {f"head.{name}": value.reshape(1) if value.ndim == 0 else value for name, value in head.items()}
        reference.load_state_dict(backbone | head)
        reference_batches = [_batch(index, packed) for index in range(2)]
        terms = [trainer.loss(reference(batch), batch, trainer.device) for batch in reference_batches]
        (sum(term[0] for term in terms) / sum(term[1] for term in terms)).backward()
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1e6)
        torch.testing.assert_close(norm, expected_norm, atol=1e-5, rtol=1e-5)
        expected = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            if param.grad is not None:
                torch.testing.assert_close(param.grad.full_tensor(), expected[name].grad, atol=1e-5, rtol=1e-5)
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
@pytest.mark.parametrize("packed", [False, True])
def test_release_head_split_load_and_fsdp_gradient_match_reference(tmp_path, packed):
    if get_device_type() == "cpu" or get_torch_device().device_count() < 2:
        pytest.skip("Requires two accelerator devices for FSDP2")
    torch.manual_seed(8)
    head = _write_release_head(tmp_path)
    scalar_names = _reshape_scalar_parameters(head)
    model = ClefDecisionModel(
        _ToyFoundation(),
        head,
        tmp_path / "joint_head.safetensors",
        head_config=HEAD_CONFIG,
        source_dir=tmp_path,
        scalar_head_parameters=scalar_names,
    )
    save_clef_release(tmp_path, model.state_dict(), head_config=HEAD_CONFIG, source_dir=tmp_path)
    release_head = load_file(str(tmp_path / "joint_head.safetensors"))
    release_head["scale"] = release_head["scale"].reshape(())
    save_file(release_head, tmp_path / "joint_head.safetensors")
    spawn_with_rendezvous(_worker, args=(str(tmp_path), packed), nprocs=2)
