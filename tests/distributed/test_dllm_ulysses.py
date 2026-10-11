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
"""Ulysses sequence parallelism under `DllmTrainer`, on real ranks.

Each test runs the same packed row twice over the same ranks: once with SP off,
where every rank holds the whole row, and once sharded across an Ulysses group.
What has to agree is what a training step consumes -- the reduced loss
`postforward` returns and the gradient FSDP would see after averaging over the
group -- so the comparison goes through the shipped collator, the model's own
flash path and the trainer's reduction, not a re-derivation of any of them.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist


LLADA_TOY = "./tests/toy_config/llada_toy"
LLADA_MOE_TOY = "./tests/toy_config/llada_moe_toy"
LLADA_V_TOY = "./tests/toy_config/llada_v_toy"
LLADA2_UNI_TOY = "./tests/toy_config/llada2_uni_toy"
#: Three samples, 21 tokens: one token of SP padding at sp=2 and three at sp=4,
#: with shard boundaries that fall inside samples.
SAMPLES = ((9, slice(4, 8), 1), (7, slice(2, 6), 2), (5, slice(1, 3), 3))
#: 22 tokens, so sp=2 adds no padding. The router counts SP padding like any token
#: (as it does `pad_to_length` padding without SP), so a padded row's aux loss is
#: not the unpadded one's.
MOE_SAMPLES = ((9, slice(4, 8), 1), (7, slice(2, 6), 2), (6, slice(1, 3), 3))


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.float().norm()
    return float((actual.float() - expected.float()).norm() / scale.clamp(min=1e-12))


def _both_topologies(world_size: int):
    # Two topologies in one process. `_init_parallel_state` caches by name, so a second
    # call under `"base"` would return the first mesh; `name=None` keeps both.
    from veomni.distributed.parallel_state import _init_parallel_state

    replicated = _init_parallel_state(dp_size=world_size, dp_shard_size=world_size, ulysses_size=1, name=None)
    sharded = _init_parallel_state(dp_size=1, dp_shard_size=1, ulysses_size=world_size, name=None)
    return replicated, sharded


def _build_model(config_path: str, device, config_kwargs=None):
    from veomni.models import build_foundation_model
    from veomni.ops import apply_ops_config

    from ..tools.training_utils import make_eager_ops_config

    apply_ops_config(make_eager_ops_config())
    torch.manual_seed(0)
    model = build_foundation_model(
        config_path,
        torch_dtype="bfloat16",
        attn_implementation="veomni_flash_attention_2_with_sp",
        init_device="cuda",
        config_kwargs=config_kwargs,
    ).to(device)
    if model.config.model_type in ("llada_moe", "llada2_uni"):
        # Spread the router logits: bf16 noise from a different GEMM row count must not
        # flip a near-tied top-k route, or the arms stop computing the same function.
        with torch.no_grad():
            for layer in model.model.layers:
                if not hasattr(layer.mlp, "experts"):
                    continue
                torch.nn.init.normal_(layer.mlp.experts.gate_up_proj, std=0.02)
                torch.nn.init.normal_(layer.mlp.experts.down_proj, std=0.02)
                torch.nn.init.normal_(layer.mlp.gate.weight, std=1.0)
    return model.train()


def _text_features(samples):
    from ..models.test_llada import _sample

    features = []
    for length, answer, seed in samples:
        feature = _sample(length, answer, seed)
        feature["answer_mask"] = (feature["labels"] != -100).long()
        features.append(feature)
    return features


def _image_features(model_type: str):
    """Three samples, one text-only, ordered so that an image straddles the sp=2 shard boundary."""
    if model_type == "llada_v":
        from ..models.test_llada_v import _features

        image, text, other_image = _features()
        return [other_image, image, text]
    from ..models.test_llada2_uni import _features

    image, text, other_image = _features()
    features = [text, image, other_image]
    generator = torch.Generator().manual_seed(0)
    for feature in features:
        if "pixel_values" in feature:
            # the toy processor emits zeros, which would give every patch the same VQ code
            feature["pixel_values"] = torch.randn(feature["pixel_values"].shape, generator=generator)
    return features


def _build_batch(model, device, features):
    """Collate through `DllmTrainer._build_collate_fn`, the collator a training run gets."""
    import types

    from veomni.trainer.dllm_trainer import DllmTrainer

    trainer = DllmTrainer.__new__(DllmTrainer)
    trainer.base = types.SimpleNamespace(
        args=types.SimpleNamespace(train=types.SimpleNamespace(pad_to_length=False)),
        model=types.SimpleNamespace(model_config=model.config),
    )
    trainer._build_collate_fn()
    batch = trainer.base.collate_fn([dict(feature) for feature in features])
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _train_step(model, device, with_aux: bool):
    """One micro-batch through the trainer's hooks; returns the reduced loss, metrics and raw grads."""
    import types

    from veomni.trainer.dllm_trainer import DllmTrainer

    model_type = model.config.model_type
    if model_type in ("llada_v", "llada2_uni"):
        features = _image_features(model_type)
    else:
        features = _text_features(MOE_SAMPLES if with_aux else SAMPLES)
    batch = _build_batch(model, device, features)
    if "image_mask" in batch and batch["image_mask"].shape != batch["input_ids"].shape:
        image_mask, chunk = batch["image_mask"][0], batch["input_ids"].shape[1]
        cuts = [b for b in range(chunk, image_mask.numel(), chunk) if image_mask[b - 1] and image_mask[b]]
        assert cuts, "no image straddles a shard boundary; the test would not cover the split"
    trainer = DllmTrainer.__new__(DllmTrainer)
    trainer.base = types.SimpleNamespace(model=types.SimpleNamespace(model_config=model.config))
    trainer.on_micro_batches_ready([batch])
    kwargs = {"output_router_logits": True} if with_aux else {}
    outputs = model(**batch, use_cache=False, **kwargs)
    loss, metrics = trainer.postforward(outputs, batch)
    loss.backward()
    grads = {name: p.grad.detach().float().clone() for name, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), {k: float(v.detach()) for k, v in metrics.items()}, grads


def _whole_step_worker(config_path: str, with_aux: bool, config_kwargs=None):
    from veomni.distributed.parallel_state import use_parallel_state

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    replicated, sharded = _both_topologies(world_size)

    with use_parallel_state(replicated):
        model = _build_model(config_path, device, config_kwargs)
        plain_loss, plain_metrics, plain_grads = _train_step(model, device, with_aux)
        del model

    with use_parallel_state(sharded):
        model = _build_model(config_path, device, config_kwargs)
        sp_loss, sp_metrics, sp_grads = _train_step(model, device, with_aux)
        del model

    # bf16 flash on both arms, reduced in different orders and tree shapes.
    tolerance = 2e-2
    loss_error = abs(sp_loss - plain_loss) / max(abs(plain_loss), 1e-9)
    assert loss_error < tolerance, (
        f"[Rank {rank}] the reduced loss is {sp_loss} at sp={world_size} against {plain_loss} without SP, "
        f"off by {loss_error:.1e}. A factor of {world_size} is the shard sum or the sample count."
    )
    if with_aux:
        aux_error = abs(sp_metrics["aux_loss"] - plain_metrics["aux_loss"]) / max(abs(plain_metrics["aux_loss"]), 1e-9)
        assert aux_error < tolerance, (
            f"[Rank {rank}] aux_loss is {sp_metrics['aux_loss']} at sp={world_size} against "
            f"{plain_metrics['aux_loss']} without SP, off by {aux_error:.1e}"
        )

    assert sp_grads.keys() == plain_grads.keys()
    worst_name, worst = None, 0.0
    for name, grad in sp_grads.items():
        # FSDP averages over the Ulysses peers; reproduce that here.
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad /= world_size
        error = _relative_error(grad, plain_grads[name])
        if error > worst:
            worst_name, worst = name, error
    assert worst < 5e-2, (
        f"[Rank {rank}] the group-averaged gradient of {worst_name} is off by {worst:.1e} at sp={world_size}"
    )
    dist.barrier()


def _load_balancing_worker():
    """The aux loss's group reduction against the unsharded loss, in fp32 so it is exact."""
    from veomni.distributed.parallel_state import use_parallel_state
    from veomni.models.transformers.llada.dllm_utils import ulysses_load_balancing_loss
    from veomni.models.transformers.llada_moe.modeling_llada_moe import load_balancing_loss_func

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    _, sharded = _both_topologies(world_size)

    num_experts, top_k, tokens, layers = 8, 2, 12 * world_size, 3
    torch.manual_seed(7)
    full = [torch.randn(tokens, num_experts, device=device, requires_grad=True) for _ in range(layers)]
    reference = load_balancing_loss_func(tuple(full), num_experts, top_k)
    reference.backward()

    shard = slice(rank * tokens // world_size, (rank + 1) * tokens // world_size)
    local = [layer.detach()[shard].clone().requires_grad_(True) for layer in full]
    with use_parallel_state(sharded):
        value = ulysses_load_balancing_loss(tuple(local), num_experts, top_k)
        value.backward()

    torch.testing.assert_close(value, reference.detach(), rtol=1e-5, atol=1e-6)
    for layer_full, layer_local in zip(full, local):
        # `SumOverShards` hands each rank sp_size times its share, for FSDP's average to undo.
        torch.testing.assert_close(layer_local.grad / world_size, layer_full.grad[shard], rtol=1e-5, atol=1e-7)
    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_llada_two_ranks_agree_with_one():
    from ..tools.launch_utils import torchrun

    torchrun(partial(_whole_step_worker, LLADA_TOY, False), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 devices")
def test_llada_four_ranks_agree_with_one_when_the_kv_heads_are_repeated():
    from ..tools.launch_utils import torchrun

    torchrun(partial(_whole_step_worker, LLADA_TOY, False, {"num_key_value_heads": 2}), world_size=4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_llada_moe_two_ranks_agree_with_one_including_the_aux_loss():
    from ..tools.launch_utils import torchrun

    torchrun(partial(_whole_step_worker, LLADA_MOE_TOY, True), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_llada_v_two_ranks_agree_with_one_when_an_image_is_split():
    from ..tools.launch_utils import torchrun

    torchrun(partial(_whole_step_worker, LLADA_V_TOY, False), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_llada2_uni_two_ranks_agree_with_one_when_an_image_is_split():
    from ..tools.launch_utils import torchrun

    torchrun(partial(_whole_step_worker, LLADA2_UNI_TOY, False), world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 devices")
def test_load_balancing_loss_reduces_over_the_group():
    from ..tools.launch_utils import torchrun

    torchrun(_load_balancing_worker, world_size=2)
