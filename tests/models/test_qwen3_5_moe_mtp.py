from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from veomni.distributed.utils import check_fqn_match
from veomni.models.checkpoint_tensor_loading import maybe_convert_checkpoint_tensor
from veomni.models.transformers.qwen3_5_moe import register_qwen3_5_moe_modeling
from veomni.models.transformers.qwen3_5_moe.parallel_plan import get_parallel_plan
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter
from veomni.utils.device import IS_NPU_AVAILABLE


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_5.generated import patched_modeling_qwen3_5_npu as dense_modeling
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_npu as modeling
else:
    from veomni.models.transformers.qwen3_5.generated import patched_modeling_qwen3_5_gpu as dense_modeling
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling


TOY_CONFIG = Path(__file__).parents[1] / "toy_config" / "qwen3_5_moe_toy"
NUM_EXPERTS = 4
HIDDEN_DIM = 8
INTERMEDIATE_DIM = 6


class _SumFusion(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add the embedding and recurrent-hidden halves of the fusion input."""
        hidden_size = hidden_states.shape[-1] // 2
        return hidden_states[..., :hidden_size] + hidden_states[..., hidden_size:]


class _EmptyRotary(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        """Return inert rotary tensors for an empty-backbone context test."""
        return torch.empty(0, device=hidden_states.device), torch.empty(0, device=hidden_states.device)


class _AddLayer(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.inputs = []

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Record the depth input and add a depth-specific marker value."""
        self.inputs.append(hidden_states.detach().clone())
        output = hidden_states + self.offset
        if kwargs.get("return_router_logits", False):
            router_logits = torch.full(
                (hidden_states.shape[0] * hidden_states.shape[1], NUM_EXPERTS),
                self.offset,
            )
            return output, router_logits
        return output


def _make_mtp_expert_key(expert: int, proj: str) -> str:
    return f"mtp.layers.0.mlp.experts.{expert}.{proj}.weight"


def _make_expert_tensor(proj: str, expert_id: int) -> torch.Tensor:
    shape = (HIDDEN_DIM, INTERMEDIATE_DIM) if proj == "down_proj" else (INTERMEDIATE_DIM, HIDDEN_DIM)
    offset = {"gate_proj": 0.0, "up_proj": 0.1, "down_proj": 0.2}[proj]
    return torch.full(shape, expert_id + offset)


def test_qwen3_5_moe_mtp_structure():
    text_config = AutoConfig.from_pretrained(TOY_CONFIG).text_config
    mtp = modeling.Qwen3_5MoeMTP(text_config)

    assert len(mtp.layers) == text_config.mtp_num_hidden_layers
    assert isinstance(mtp.layers[0], modeling.Qwen3_5MoeDecoderLayer)
    parameter_names = {name for name, _ in mtp.named_parameters()}
    assert "fc.weight" in parameter_names
    assert "layers.0.mlp.gate.weight" in parameter_names


@pytest.mark.parametrize(
    ("backend", "text_model_cls", "config_path"),
    [
        (dense_modeling, dense_modeling.Qwen3_5TextModel, Path(__file__).parents[1] / "toy_config" / "qwen3_5_toy"),
        (modeling, modeling.Qwen3_5MoeTextModel, TOY_CONFIG),
    ],
)
@pytest.mark.parametrize(("return_mtp_context", "expect_context"), [(False, False), (True, True)])
def test_qwen3_5_eval_text_model_returns_mtp_context_on_demand(
    monkeypatch, backend, text_model_cls, config_path, return_mtp_context, expect_context
):
    config = deepcopy(AutoConfig.from_pretrained(config_path).text_config)
    config.num_hidden_layers = 0
    with torch.device("meta"):
        text_model = text_model_cls(config)
    text_model.norm = torch.nn.Identity()
    text_model.rotary_emb = _EmptyRotary()
    text_model.eval()

    causal_mask = torch.ones(1, 1, 4, 4)
    monkeypatch.setattr(backend, "create_causal_mask", lambda **kwargs: causal_mask)
    outputs = text_model(
        inputs_embeds=torch.randn(1, 4, config.hidden_size),
        position_ids=torch.arange(4).unsqueeze(0),
        use_cache=False,
        return_mtp_context=return_mtp_context,
    )

    mtp_context = getattr(outputs, "mtp_context", None)
    assert (mtp_context is not None) is expect_context
    if expect_context:
        assert mtp_context["attention_mask"] is causal_mask


def test_qwen3_5_mtp_labels_advance_once_per_depth():
    feature = {"labels": torch.tensor([10, 11, 12, 13, 14])}

    dense_modeling.make_mtp_labels(feature, num_depths=3)

    expected = torch.tensor(
        [
            [12, 13, 14, -100, -100],
            [13, 14, -100, -100, -100],
            [14, -100, -100, -100, -100],
        ]
    )
    assert torch.equal(feature["mtp_labels"], expected)


@pytest.mark.parametrize(
    ("mtp_cls", "config_path"),
    [
        (dense_modeling.Qwen3_5MTP, Path(__file__).parents[1] / "toy_config" / "qwen3_5_toy"),
        (modeling.Qwen3_5MoeMTP, TOY_CONFIG),
    ],
)
def test_qwen3_5_mtp_layers_are_recurrent_prediction_depths(mtp_cls, config_path):
    config = deepcopy(AutoConfig.from_pretrained(config_path).text_config)
    config.mtp_num_hidden_layers = 2
    with torch.device("meta"):
        mtp = mtp_cls(config)

    layer_0 = _AddLayer(10.0)
    layer_1 = _AddLayer(100.0)
    mtp.pre_fc_norm_embedding = torch.nn.Identity()
    mtp.pre_fc_norm_hidden = torch.nn.Identity()
    mtp.fc = _SumFusion()
    mtp.layers = torch.nn.ModuleList([layer_0, layer_1])
    mtp.norm = torch.nn.Identity()

    inputs_embeds = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    trunk_hidden = torch.zeros_like(inputs_embeds)
    mtp_outputs = mtp(
        hidden_states=trunk_hidden,
        inputs_embeds=inputs_embeds,
        position_embeddings=(torch.empty(0), torch.empty(0)),
    )
    if isinstance(mtp, modeling.Qwen3_5MoeMTP):
        depth_hidden_states, depth_router_logits = mtp_outputs
        assert depth_router_logits is None
    else:
        depth_hidden_states = mtp_outputs

    expected_depth_0_input = torch.tensor([[[2.0], [3.0], [4.0], [0.0]]])
    expected_depth_0_output = expected_depth_0_input + 10.0
    expected_depth_1_input = expected_depth_0_output + torch.tensor([[[3.0], [4.0], [0.0], [0.0]]])
    expected_depth_1_output = expected_depth_1_input + 100.0

    assert len(depth_hidden_states) == 2
    assert torch.equal(layer_0.inputs[0], expected_depth_0_input)
    assert torch.equal(layer_1.inputs[0], expected_depth_1_input)
    assert torch.equal(depth_hidden_states[0], expected_depth_0_output)
    assert torch.equal(depth_hidden_states[1], expected_depth_1_output)


def test_qwen3_5_moe_mtp_collects_router_logits_per_depth():
    config = deepcopy(AutoConfig.from_pretrained(TOY_CONFIG).text_config)
    config.mtp_num_hidden_layers = 2
    with torch.device("meta"):
        mtp = modeling.Qwen3_5MoeMTP(config)

    mtp.pre_fc_norm_embedding = torch.nn.Identity()
    mtp.pre_fc_norm_hidden = torch.nn.Identity()
    mtp.fc = _SumFusion()
    mtp.layers = torch.nn.ModuleList([_AddLayer(10.0), _AddLayer(100.0)])
    mtp.norm = torch.nn.Identity()
    inputs_embeds = torch.tensor([[[1.0], [2.0], [3.0]]])

    depth_hidden_states, depth_router_logits = mtp(
        hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        output_router_logits=True,
    )

    assert len(depth_hidden_states) == 2
    assert len(depth_router_logits) == 2
    assert depth_router_logits[0].shape == (3, NUM_EXPERTS)
    assert torch.equal(depth_router_logits[0], torch.full((3, NUM_EXPERTS), 10.0))
    assert torch.equal(depth_router_logits[1], torch.full((3, NUM_EXPERTS), 100.0))


def test_qwen3_5_moe_mtp_router_aux_uses_layer_specific_valid_masks():
    calls = []

    def router_loss_fn(gate_logits, num_experts, top_k, attention_mask):
        calls.append((gate_logits, num_experts, top_k, attention_mask))
        return sum(logits.sum() * 0 for logits in gate_logits) + 1.25

    batch_size, num_depths, sequence_length, num_experts = 2, 2, 3, 4
    foundation_router_logits = tuple(torch.randn(batch_size * sequence_length, num_experts) for _ in range(2))
    mtp_router_logits = tuple(torch.randn(batch_size * sequence_length, num_experts) for _ in range(num_depths))
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    mtp_labels = torch.tensor(
        [
            [[4, 5, -100], [5, -100, -100]],
            [[6, -100, -100], [-100, -100, -100]],
        ]
    )

    aux_loss, combined_router_logits = modeling.compute_mtp_router_aux_loss(
        router_loss_fn,
        foundation_router_logits,
        mtp_router_logits,
        attention_mask,
        mtp_labels,
        num_experts,
        top_k=2,
    )

    expected_layer_masks = torch.tensor(
        [
            [[1, 1, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 0]],
        ]
    )
    assert aux_loss.item() == 1.25
    expected_router_logits = foundation_router_logits + mtp_router_logits
    assert all(actual is expected for actual, expected in zip(combined_router_logits, expected_router_logits))
    assert len(calls[0][0]) == 1
    assert torch.equal(calls[0][0][0], torch.cat(expected_router_logits, dim=0))
    assert calls[0][1:3] == (num_experts, 2)
    assert torch.equal(calls[0][3], expected_layer_masks.flatten(0, 1))

    eager_aux_loss, _ = modeling.compute_mtp_router_aux_loss(
        modeling.load_balancing_loss_func,
        foundation_router_logits,
        mtp_router_logits,
        attention_mask,
        mtp_labels,
        num_experts,
        top_k=2,
    )
    assert torch.isfinite(eager_aux_loss)


def test_qwen3_5_mtp_loss_flattens_depth_and_uses_all_valid_targets():
    calls = []

    def loss_fn(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"].sum() * 0 + 2.5, None, None

    hidden_states = (torch.zeros(1, 4, 3), torch.ones(1, 4, 3))
    mtp_labels = torch.tensor([[[2, 3, -100, -100], [3, -100, -100, -100]]])
    weights = torch.zeros(7, 3)

    loss = dense_modeling.compute_mtp_loss(
        loss_fn,
        hidden_states,
        mtp_labels,
        weights,
        vocab_size=7,
        custom_normalization_kwarg="preserved",
    )

    assert loss.item() == 2.5
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape == (2, 4, 3)
    assert calls[0]["labels"].shape == (2, 4)
    assert torch.equal(calls[0]["labels"], calls[0]["shift_labels"])
    assert calls[0]["num_items_in_batch"].item() == 3
    assert calls[0]["custom_normalization_kwarg"] == "preserved"


def test_qwen3_5_mtp_loss_is_differentiable_zero_without_valid_targets():
    calls = []

    def loss_fn(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"].sum() + 1.0, None, None

    hidden_state = torch.randn(1, 2, 3, requires_grad=True)
    mtp_labels = torch.full((1, 1, 2), -100)

    loss = dense_modeling.compute_mtp_loss(
        loss_fn,
        (hidden_state,),
        mtp_labels,
        torch.zeros(7, 3),
        vocab_size=7,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert calls[0]["num_items_in_batch"].item() == 1
    assert calls[0]["labels"].reshape(-1)[0].item() == 0
    assert torch.count_nonzero(hidden_state.grad).item() == 0


def test_qwen3_5_moe_mtp_outputs_keep_auxiliary_fields():
    context = {"position_ids": torch.arange(4)}
    router_logits = (torch.zeros(4, 2),)
    model_output = modeling.Qwen3_5MoeMTPContextOutput(
        last_hidden_state=torch.zeros(1, 4, 8),
        router_logits=router_logits,
        mtp_context=context,
    )
    loss_dict = {"foundation_loss": torch.tensor(1.0), "mtp_loss": torch.tensor(0.5)}
    causal_output = modeling.Qwen3_5MoeCausalLMOutputWithLogProbs(loss_dict=loss_dict)

    assert model_output.mtp_context is context
    assert model_output.router_logits is router_logits
    assert causal_output.loss_dict == loss_dict


def test_qwen3_5_moe_parallel_plan_covers_mtp_experts():
    plan = get_parallel_plan()
    patterns = plan.extra_parallel_plan["ep"]
    no_shard_patterns = plan.extra_parallel_fsdp_no_shard_module["ep"]

    for prefix in ("model.language_model.layers.0", "mtp.layers.0"):
        assert any(check_fqn_match(pattern, f"{prefix}.mlp.experts.gate_up_proj") for pattern in patterns)
        assert any(check_fqn_match(pattern, f"{prefix}.mlp.experts.down_proj") for pattern in patterns)
        assert any(check_fqn_match(pattern, f"{prefix}.mlp.experts") for pattern in no_shard_patterns)


def test_qwen3_5_moe_registry_attaches_checkpoint_converter():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")

    assert callable(model_cls._create_checkpoint_tensor_converter)
    assert callable(model_cls._convert_fqn_to_index_mapping)


def test_qwen3_5_moe_reuses_checkpoint_converter_for_mtp_experts():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=NUM_EXPERTS))
    converter = model_cls._create_checkpoint_tensor_converter(SimpleNamespace(config=config, mtp=object()))
    dispatched = {}

    assert isinstance(converter, Qwen3MoeCheckpointTensorConverter)
    assert converter.can_handle(_make_mtp_expert_key(0, "gate_proj"))

    trunk_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    trunk = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM)
    result = maybe_convert_checkpoint_tensor(trunk_key, trunk, converter)
    assert result is not None and result.name == trunk_key and result.tensor is trunk

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert_id in range(NUM_EXPERTS):
            result = maybe_convert_checkpoint_tensor(
                _make_mtp_expert_key(expert_id, proj),
                _make_expert_tensor(proj, expert_id),
                converter,
            )
            if result is not None:
                dispatched[result.name] = result.tensor

    assert converter.finalize() == []
    assert dispatched["mtp.layers.0.mlp.experts.gate_up_proj"].shape == (
        NUM_EXPERTS,
        2 * INTERMEDIATE_DIM,
        HIDDEN_DIM,
    )
    assert dispatched["mtp.layers.0.mlp.experts.down_proj"].shape == (
        NUM_EXPERTS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
    )


def test_qwen3_5_moe_checkpoint_converter_rejects_incomplete_mtp_experts():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=NUM_EXPERTS))
    converter = model_cls._create_checkpoint_tensor_converter(SimpleNamespace(config=config, mtp=object()))
    converter.convert(_make_mtp_expert_key(0, "down_proj"), _make_expert_tensor("down_proj", 0))

    with pytest.raises(RuntimeError, match="incomplete checkpoint detected"):
        converter.finalize()


def test_qwen3_5_moe_checkpoint_converter_handles_trunk_without_mtp():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=NUM_EXPERTS))
    converter = model_cls._create_checkpoint_tensor_converter(SimpleNamespace(config=config, mtp=None))
    dispatched = {}

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert_id in range(NUM_EXPERTS):
            key = f"model.language_model.layers.0.mlp.experts.{expert_id}.{proj}.weight"
            result = maybe_convert_checkpoint_tensor(key, _make_expert_tensor(proj, expert_id), converter)
            if result is not None:
                dispatched[result.name] = result.tensor

    assert isinstance(converter, Qwen3MoeCheckpointTensorConverter)
    assert converter.finalize() == []
    assert set(dispatched) == {
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    }


def test_qwen3_5_moe_reuses_checkpoint_index_mapping():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    trunk_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    mapping = {trunk_key: 1}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert_id in range(NUM_EXPERTS):
            mapping[_make_mtp_expert_key(expert_id, proj)] = expert_id + 2

    converted = model_cls._convert_fqn_to_index_mapping(mapping)

    assert converted[trunk_key] == 1
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in converted
    assert "mtp.layers.0.mlp.experts.down_proj" in converted
    assert not any("experts.0." in key for key in converted)


def test_qwen3_5_moe_conditional_generation_builds_mtp(monkeypatch):
    config = AutoConfig.from_pretrained(TOY_CONFIG)
    config.text_config.mtp_loss_weight = 0.3
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    for slot_name in (
        "veomni_rms_norm",
        "veomni_moe_experts_forward",
        "veomni_causal_lm_loss",
        "veomni_load_balancing_loss",
        "veomni_rms_norm_gated",
        "veomni_causal_conv1d",
        "veomni_chunk_gated_delta_rule",
    ):
        getattr(modeling, slot_name).bind("eager")

    with torch.device("meta"):
        model = modeling.Qwen3_5MoeForConditionalGeneration(config)

    assert isinstance(model.mtp, modeling.Qwen3_5MoeMTP)
    assert any(name.startswith("mtp.") for name, _ in model.named_parameters())
