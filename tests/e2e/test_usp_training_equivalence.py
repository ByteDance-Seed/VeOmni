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

"""Full-model training equivalence for packed USP with CP document padding.

The test runs a tiny Qwen3 CausalLM on four GPUs and compares three views of
the same two unaligned documents:

* original packed documents with no sequence parallelism;
* documents padded to ``2 * cp_size`` plus fixed tail padding, still without
  sequence parallelism;
* the production ``MainCollator`` and model path with ``cp=2, ulysses=2``.

It validates padding invariance, valid-token logits, globally reduced loss,
all parameter gradients, and one FSDP2 optimizer update.
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests
from transformers import Qwen3Config

from veomni.data.data_collator import MainCollator, PackingCollator
from veomni.distributed import parallel_state as PS
from veomni.models.auto import build_foundation_model
from veomni.utils.device import empty_cache, get_device_type, get_dist_comm_backend, get_torch_device

from ..parallel.ring._ref import ATTN_IMPL_WITH_SP
from ..parallel.ring._ref import FA_OK as _FA_OK
from ..tools.training_utils import make_eager_ops_config


try:
    _DIST_BACKEND = get_dist_comm_backend()
except Exception as exc:
    pytest.skip(f"distributed accelerator backend unavailable: {exc}", allow_module_level=True)
if not dist.is_available() or not dist.is_backend_available(_DIST_BACKEND):
    pytest.skip("distributed accelerator backend unavailable", allow_module_level=True)


CP_SIZE = 2
ULYSSES_SIZE = 2
PAD_TO_LENGTH = 32
VOCAB_SIZE = 128


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _features() -> list[dict[str, torch.Tensor]]:
    """Return fresh documents with lengths 11 and 13 and unique nonzero ids."""

    documents = (
        torch.arange(1, 12, dtype=torch.long),
        torch.arange(32, 45, dtype=torch.long),
    )
    return [
        {
            "input_ids": document.clone(),
            "attention_mask": torch.ones_like(document),
            "labels": document.clone(),
        }
        for document in documents
    ]


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _reshape_logits(output, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return output.logits.reshape(*batch["input_ids"].shape, VOCAB_SIZE)


def _full_parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Materialize each FSDP2 parameter for full-model comparison."""

    state = {}
    for name, parameter in model.named_parameters():
        assert isinstance(parameter, DTensor), f"FSDP2 parameter {name} is not a DTensor"
        state[name] = parameter.full_tensor().detach().clone()
    return state


def _assert_relative_vector_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    max_relative_l2: float,
    min_cosine: float,
) -> None:
    actual = actual.detach().float().reshape(-1)
    expected = expected.detach().float().reshape(-1)
    expected_norm = torch.linalg.vector_norm(expected)
    if expected_norm.item() == 0:
        assert torch.count_nonzero(actual).item() == 0, f"{name}: reference is zero but actual is not"
        return

    relative_l2 = torch.linalg.vector_norm(actual - expected) / expected_norm
    cosine = F.cosine_similarity(actual, expected, dim=0)
    assert relative_l2.item() <= max_relative_l2, (
        f"{name}: relative L2 error {relative_l2.item():.4f} exceeds {max_relative_l2:.4f}"
    )
    assert cosine.item() >= min_cosine, f"{name}: cosine {cosine.item():.6f} is below {min_cosine:.6f}"


class USPTrainingEquivalenceE2ETest(MultiProcessTestCase):
    @property
    def world_size(self):
        return CP_SIZE * ULYSSES_SIZE

    def setUp(self):
        super().setUp()
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _init_parallel_states(self) -> tuple[PS.ParallelState, PS.ParallelState]:
        store = dist.FileStore(self.file_name, self.world_size)
        get_torch_device().set_device(self.rank)
        dist.init_process_group(
            _DIST_BACKEND,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        os.environ["LOCAL_RANK"] = str(self.rank)

        PS.clear_parallel_state()
        usp_state = PS.init_parallel_state(
            dp_size=1,
            dp_shard_size=1,
            cp_size=CP_SIZE,
            ulysses_size=ULYSSES_SIZE,
            device_type=get_device_type(),
            name="usp_e2e",
        )
        # The reference model is intentionally replicated on every process.
        # A meshless DP-only state keeps every model/data path non-SP while
        # satisfying ParallelState's world-size invariant.
        reference_state = PS.ParallelState(
            dp_size=self.world_size,
            dp_shard_size=self.world_size,
            device_type=get_device_type(),
        )
        return reference_state, usp_state

    def _build_batches(
        self,
        reference_state: PS.ParallelState,
        usp_state: PS.ParallelState,
        device: torch.device,
    ) -> tuple[dict, dict, dict]:
        with PS.use_parallel_state(reference_state):
            unaligned_batch = MainCollator()(_features())

            # Build the full, CP-aligned reference without slicing. The actual
            # distributed batch below goes through an unmodified MainCollator;
            # this override only gives the reference the same document padding.
            aligned_collator = MainCollator(pad_to_length=PAD_TO_LENGTH)
            packing_collator = next(
                stage for stage in aligned_collator.preforward_pipeline if isinstance(stage, PackingCollator)
            )
            packing_collator.cp_size = CP_SIZE
            aligned_batch = aligned_collator(_features())

        with PS.use_parallel_state(usp_state):
            usp_batch = MainCollator(pad_to_length=PAD_TO_LENGTH)(_features())

        return tuple(_to_device(batch, device) for batch in (unaligned_batch, aligned_batch, usp_batch))

    @staticmethod
    def _broadcast_model(model: torch.nn.Module) -> None:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

    @staticmethod
    def _fully_shard_usp_model(model: torch.nn.Module, state: PS.ParallelState) -> None:
        """Apply the dense-model subset of VeOmni's production FSDP2 plan."""

        for layer in model.model.layers:
            fully_shard(layer, mesh=state.fsdp_mesh)
        fully_shard(model, mesh=state.fsdp_mesh)

    @staticmethod
    def _full_usp_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"USP gradient missing for {name}"
            assert isinstance(parameter.grad, DTensor), f"FSDP2 gradient {name} is not a DTensor"
            gradients[name] = parameter.grad.full_tensor().detach()
        return gradients

    @pytest.mark.skipif(not _FA_OK, reason="a flash-attn backend (FA2 or FA4) is required")
    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_packed_padding_full_model_training_equivalence(self):
        reference_state, usp_state = self._init_parallel_states()
        device = torch.device(f"{get_device_type()}:{self.rank}")

        torch.manual_seed(2026)
        with PS.use_parallel_state(reference_state):
            reference_model = build_foundation_model(
                config_path=_tiny_qwen3_config(),
                weights_path=None,
                torch_dtype="bfloat16",
                init_device=get_device_type(),
                ops_implementation=make_eager_ops_config(attn_implementation=ATTN_IMPL_WITH_SP),
            ).train()
        self._broadcast_model(reference_model)
        usp_model = copy.deepcopy(reference_model).train()
        with PS.use_parallel_state(usp_state):
            self._fully_shard_usp_model(usp_model, usp_state)

        unaligned_batch, aligned_batch, usp_batch = self._build_batches(
            reference_state,
            usp_state,
            device,
        )

        # 11 -> 12 and 13 -> 16 document alignment, then four fixed-tail
        # padding tokens. The distributed collator must preserve every valid
        # token exactly once across its four local shards.
        assert aligned_batch["input_ids"].shape == (1, PAD_TO_LENGTH)
        assert usp_batch["input_ids"].shape == (1, PAD_TO_LENGTH // self.world_size)
        gathered_ids = [torch.empty_like(usp_batch["input_ids"]) for _ in range(self.world_size)]
        dist.all_gather(gathered_ids, usp_batch["input_ids"], group=usp_state.sp_group)
        gathered_valid_ids = torch.cat(gathered_ids).reshape(-1)
        gathered_valid_ids = gathered_valid_ids[gathered_valid_ids != 0].sort().values
        full_valid_ids = aligned_batch["input_ids"].reshape(-1)
        full_valid_ids = full_valid_ids[full_valid_ids != 0].sort().values
        torch.testing.assert_close(gathered_valid_ids, full_valid_ids, rtol=0, atol=0)

        with PS.use_parallel_state(reference_state):
            with torch.no_grad():
                unaligned_output = reference_model(**unaligned_batch, use_cache=False)
            aligned_output = reference_model(**aligned_batch, use_cache=False)
        with PS.use_parallel_state(usp_state):
            usp_output = usp_model(**usp_batch, use_cache=False)

        unaligned_logits = _reshape_logits(unaligned_output, unaligned_batch)
        aligned_logits = _reshape_logits(aligned_output, aligned_batch)
        usp_logits = _reshape_logits(usp_output, usp_batch)

        # Per-document padding is causally after every valid token, so neither
        # valid logits nor the mean next-token loss may change.
        for unaligned_start, aligned_start, length in ((0, 0, 11), (11, 12, 13)):
            torch.testing.assert_close(
                aligned_logits[:, aligned_start : aligned_start + length],
                unaligned_logits[:, unaligned_start : unaligned_start + length],
                rtol=2e-2,
                atol=2e-2,
            )
        torch.testing.assert_close(aligned_output.loss, unaligned_output.loss, rtol=2e-2, atol=2e-2)

        # Unique nonzero token ids provide an independent mapping from each USP
        # local token back to its full aligned-reference position.
        full_ids = aligned_batch["input_ids"][0]
        local_ids = usp_batch["input_ids"][0]
        local_valid = local_ids != 0
        id_to_position = torch.full((VOCAB_SIZE,), -1, dtype=torch.long, device=device)
        full_positions = torch.arange(full_ids.numel(), device=device)
        id_to_position[full_ids[full_ids != 0]] = full_positions[full_ids != 0]
        expected_local_logits = aligned_logits[0, id_to_position[local_ids[local_valid]]]
        torch.testing.assert_close(
            usp_logits[0, local_valid],
            expected_local_logits,
            rtol=6e-2,
            atol=8e-2,
        )
        torch.testing.assert_close(usp_output.loss, aligned_output.loss, rtol=3e-2, atol=3e-2)

        reference_model.zero_grad(set_to_none=True)
        usp_model.zero_grad(set_to_none=True)
        with PS.use_parallel_state(reference_state):
            aligned_output.loss.backward()
        with PS.use_parallel_state(usp_state):
            usp_output.loss.backward()
        full_usp_gradients = self._full_usp_gradients(usp_model)

        reference_parameters = dict(reference_model.named_parameters())
        assert set(full_usp_gradients) == set(reference_parameters)
        for name, reference_parameter in reference_parameters.items():
            reference_grad = reference_parameter.grad
            assert reference_grad is not None, f"reference gradient missing for {name}"
            _assert_relative_vector_close(
                full_usp_gradients[name],
                reference_grad,
                name=f"gradient {name}",
                max_relative_l2=0.15,
                min_cosine=0.985,
            )

        reference_before = {name: parameter.detach().clone() for name, parameter in reference_parameters.items()}
        usp_before = _full_parameter_state(usp_model)
        torch.optim.SGD(reference_model.parameters(), lr=0.5).step()
        torch.optim.SGD(usp_model.parameters(), lr=0.5).step()
        usp_after = _full_parameter_state(usp_model)

        for name, reference_parameter in reference_parameters.items():
            reference_delta = reference_parameter.detach() - reference_before[name]
            usp_delta = usp_after[name] - usp_before[name]
            _assert_relative_vector_close(
                usp_delta,
                reference_delta,
                name=f"optimizer delta {name}",
                max_relative_l2=0.2,
                min_cosine=0.98,
            )

        dist.barrier()
        del reference_model, usp_model, unaligned_output, aligned_output, usp_output
        empty_cache()


if __name__ == "__main__":
    run_tests()
