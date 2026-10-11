"""LLaDA-MoE under DllmTrainer: bidirectional packed loss, aux loss, EP plan, checkpoint fusion."""

import fnmatch
import importlib.util
import json
import os
import types

import pytest
import torch

from veomni.models.transformers.llada_moe.checkpoint_tensor_converter import LLaDAMoECheckpointTensorConverter
from veomni.trainer import dllm_trainer
from veomni.trainer.dllm_trainer import DllmTrainer

from .test_llada import REPO_ROOT, _packed_vs_separate, _sample, _toy_model


TOY_CONFIG = "./tests/toy_config/llada_moe_toy"
REAL_CKPT = "/mnt/hdfs/user_dir/omni_v2/ckpt/LLaDA-MoE-7B-A1B-Instruct-VeOmni"


def _moe(attn_implementation="eager"):
    model = _toy_model(attn_implementation, config=TOY_CONFIG)
    with torch.no_grad():
        for layer in model.model.layers:
            torch.nn.init.normal_(layer.mlp.experts.gate_up_proj, std=0.02)
            torch.nn.init.normal_(layer.mlp.experts.down_proj, std=0.02)
            torch.nn.init.normal_(layer.mlp.gate.weight, std=0.02)
    return model


def test_attention_is_bidirectional():
    model = _moe().eval()
    ids = torch.randint(0, 1000, (1, 10))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 1000
    with torch.no_grad():
        first = model(input_ids=ids).logits[0, 0]
        second = model(input_ids=changed).logits[0, 0]
    assert not torch.allclose(first, second)


def test_labels_without_loss_weight_are_refused():
    model = _moe()
    ids = torch.randint(0, 1000, (1, 4))
    with pytest.raises(ValueError, match="loss_weight"):
        model(input_ids=ids, labels=ids)


# No bf16 flash twin of this test: packing changes the GEMM row count and flips near-tied
# top-k routes (constraint 33). The varlen attention path is covered by the dense test.
def test_packed_loss_matches_separate_samples_sdpa(monkeypatch):
    separate, packed, leaked = _packed_vs_separate(monkeypatch, "sdpa", "cpu", "float32", TOY_CONFIG)
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-5)
    # the toy router mixes weakly, so the leak shows at ~1e-3 relative; still far outside the packed tolerance
    assert not torch.allclose(leaked, separate, rtol=1e-4, atol=1e-4)


def test_aux_loss_is_returned_apart_from_the_masked_loss():
    model = _moe()
    batch = {k: v.unsqueeze(0) for k, v in _sample(12, slice(3, 11), 7).items()}
    plain = model(**batch, use_cache=False)
    routed = model(**batch, use_cache=False, output_router_logits=True)
    assert plain.aux_loss is None
    assert isinstance(routed.aux_loss, torch.Tensor) and float(routed.aux_loss.detach()) > 0
    torch.testing.assert_close(routed.loss, plain.loss)
    (routed.loss + routed.aux_loss).backward()
    assert model.model.layers[0].mlp.gate.weight.grad is not None


def test_trainer_adds_weighted_aux_after_sample_normalization(monkeypatch):
    monkeypatch.setattr(dllm_trainer, "dllm_mean_global_loss", lambda s, n, sum_spans_sp_group=False: s / n)
    trainer = DllmTrainer.__new__(DllmTrainer)
    trainer.base = types.SimpleNamespace(
        model=types.SimpleNamespace(model_config=types.SimpleNamespace(router_aux_loss_coef=0.01))
    )
    trainer.micro_batches_num_samples = 4
    # two micro-batches: the step's aux terms add up to one coef * mean(aux), not to twice it
    trainer.num_micro_batches = 2
    step_loss, step_aux = 0.0, 0.0
    for aux in (3.0, 5.0):
        outputs = types.SimpleNamespace(loss=torch.tensor(4.0), aux_loss=torch.tensor(aux))
        loss, loss_dict = trainer.postforward(outputs, {})
        step_loss, step_aux = step_loss + float(loss), step_aux + float(loss_dict["aux_loss"])
    assert step_aux == pytest.approx(0.01 * 4.0)
    assert step_loss == pytest.approx(2.0 + 0.01 * 4.0)
    loss, loss_dict = trainer.postforward(types.SimpleNamespace(loss=torch.tensor(8.0), aux_loss=None), {})
    assert "aux_loss" not in loss_dict and float(loss) == 2.0


def test_ep_plan_covers_every_expert_weight_and_nothing_else():
    model = _moe()
    plan = model.get_parallel_plan().extra_parallel_plan["ep"]
    names = [name for name, _ in model.named_parameters()]
    sharded = {name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in plan)}
    assert sharded == {name for name in names if ".mlp.experts." in name}
    assert len(sharded) == 2 * model.config.num_hidden_layers


def test_checkpoint_converter_fuses_per_expert_weights():
    converter = LLaDAMoECheckpointTensorConverter(num_experts=2)
    prefix = "model.layers.0.mlp"
    tensors = {
        (e, proj): torch.randn(3, 5) if proj != "down_proj" else torch.randn(5, 3)
        for e in range(2)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    emitted = {}
    for (e, proj), tensor in tensors.items():
        out = converter.convert(f"{prefix}.experts.{e}.{proj}.weight", tensor)
        if out is not None:
            emitted[out.name] = out.tensor
    converter.finalize()
    gate_up = torch.stack([torch.cat([tensors[(e, "gate_proj")], tensors[(e, "up_proj")]]) for e in range(2)])
    torch.testing.assert_close(emitted[f"{prefix}.experts.gate_up_proj"], gate_up)
    torch.testing.assert_close(
        emitted[f"{prefix}.experts.down_proj"], torch.stack([tensors[(e, "down_proj")] for e in range(2)])
    )


def test_checkpoint_converter_refuses_an_incomplete_checkpoint():
    converter = LLaDAMoECheckpointTensorConverter(num_experts=2)
    converter.convert("model.layers.0.mlp.experts.0.down_proj.weight", torch.randn(5, 3))
    with pytest.raises(RuntimeError, match="incomplete"):
        converter.finalize()


def test_converter_self_test():
    spec = importlib.util.spec_from_file_location(
        "convert_lladamoe_to_veomni", os.path.join(REPO_ROOT, "scripts", "convert_lladamoe_to_veomni.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.self_test()


@pytest.mark.skipif(not os.path.isdir(REAL_CKPT), reason="converted LLaDA-MoE checkpoint not mounted")
def test_real_checkpoint_keys_match_the_model():
    from veomni.models.transformers.llada_moe.checkpoint_tensor_converter import (
        convert_llada_moe_fqn_to_index_mapping,
    )
    from veomni.models.transformers.llada_moe.configuration_llada_moe import LLaDAMoEConfig
    from veomni.models.transformers.llada_moe.modeling_llada_moe import LLaDAMoEForCausalLM

    with open(os.path.join(REAL_CKPT, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    config = LLaDAMoEConfig.from_pretrained(REAL_CKPT)
    with torch.device("meta"):
        model = LLaDAMoEForCausalLM(config)
    assert set(convert_llada_moe_fqn_to_index_mapping(index)) == set(model.state_dict())
