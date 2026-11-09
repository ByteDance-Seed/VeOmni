"""Compare CLEF integration with pinned, unmodified Cloudflare release code.

The default path is fully offline: it uses the vendored release source, a
deterministic toy tokenizer, and a tiny random Qwen3.5 model. Set
``CLEF_REFERENCE_DIR`` to the pinned release snapshot to additionally validate
the released tokenizer. The 9B backbone checkpoint is never required.
"""

import copy
import json
import os
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.tiny_configs import tiny_qwen3_5_config
from tests.models.transformers.qwen.test_qwen3_5 import _build_vlm, _pin_hf_gdn_to_torch
from tests.tools.training_utils import make_eager_ops_config
from veomni.data.decision.collator import DecisionCollator, encode_request, model_features
from veomni.data.decision.typed_decisions import parse_typed_decisions_case
from veomni.distributed.parallel_state import clear_parallel_state
from veomni.models.decision import get_decision_model
from veomni.models.decision.models.clef.modeling import (
    ClefDecisionModel,
    _load_head_module,
    _reshape_scalar_parameters,
    load_local_model,
)
from veomni.models.decision.models.clef.processing import ClefPreprocessor
from veomni.models.decision.models.clef.release import save_clef_release
from veomni.models.decision.models.clef.specification import encode_clef
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
OFFICIAL_REPO = "Cloudflare/clef-flash"
OFFICIAL_REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"
LOCAL_REFERENCE_DIR = Path(__file__).parent / "clef_reference"


class ToyTokenizer:
    """Small deterministic tokenizer for offline source/model parity."""

    pad_token_id = 0
    vocab_size = 1024

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        payload = str(text).encode("utf-8")
        return [
            zlib.crc32(payload[offset : offset + 4]) % (self.vocab_size - 1) + 1
            for offset in range(0, len(payload), 4)
        ]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode(text, add_special_tokens=add_special_tokens))


@pytest.fixture(scope="module")
def official_assets():
    directory = os.environ.get("CLEF_REFERENCE_DIR")
    if directory is not None:
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    else:
        directory = LOCAL_REFERENCE_DIR
        tokenizer = ToyTokenizer()
    return Path(directory), _load_head_module(directory), tokenizer


@pytest.fixture
def rank_zero():
    """The production CPU weight loader requires an initialized rank zero."""
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


def _row():
    """Keep gold independent of VeOmni's option sorting and tensor packing."""
    return {
        "id": "official-parity",
        "state": json.dumps("The package arrived yesterday. 包裹完整。"),
        "questions": json.dumps(
            {
                "route": {
                    "type": "choice",
                    "instructions": {"en": "Choose the next action.", "zh": "选择操作"},
                    "criteria": {"z": "Escalate", "a": "Accept", "m": None},
                },
                "intact": {"type": "noul"},
                "rating": {"type": "score", "instructions": "", "criteria": ["bad", "fair", "good"]},
                "only": {"type": "choice", "criteria": {"one": "Only option"}},
                "unlabeled": {"type": "choice", "criteria": {"b": "B", "a": "A"}},
            },
            ensure_ascii=False,
        ),
        "gold": json.dumps(
            {
                "route": {"probabilities": {"z": 0.1, "a": 0.7, "m": 0.2}, "weight": 0.5},
                "intact": {"probabilities": {"true": 0.8, "false": 0.2}, "weight": 2.0},
                "rating": {"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}, "weight": 1.5},
                "only": {"probabilities": {"one": 1.0}, "weight": 7.0},
            }
        ),
    }


def _record(row):
    return {"id": row["id"], "state": json.loads(row["state"]), "questions": json.loads(row["questions"])}


def _assert_encoding(encoded, ids, meta):
    assert list(encoded.input_ids) == ids
    assert meta["counts"] == [len(encoded.questions)]
    for index, question in enumerate(encoded.questions):
        assert meta["question_spans"][index] == question.question_span
        assert tuple(meta["option_spans"][index]) == question.option_spans
        assert meta["option_ids"][index] == question.option_ids
        assert meta["kinds"][index] == question.question_type


@pytest.mark.parametrize("case", ["ordinary", "structured-state", "truncated-state", "state-limit", "schema-overflow"])
def test_encoding_and_supervision_match_official(official_assets, case):
    _, official, tokenizer = official_assets
    row = _row()
    kwargs = {}
    if case == "structured-state":
        row["state"] = json.dumps({"z": [1, "中文"], "a": {"observed": True}})
    if case in {"truncated-state", "state-limit"}:
        row["state"] = json.dumps("Long state. 很长的状态。" * 300)
        kwargs = {"max_length": 512} if case == "truncated-state" else {"max_state_tokens": 7}
    if case == "schema-overflow":
        kwargs = {"max_length": 8}
    request = parse_typed_decisions_case(row).request
    if case == "schema-overflow":
        with pytest.raises(ValueError, match="schema requires"):
            official.encode_record(tokenizer, _record(row), **kwargs)
        with pytest.raises(ValueError, match="schema requires"):
            encode_clef(request, tokenizer, **kwargs)
        return
    encoded = official.encode_record(tokenizer, _record(row), **kwargs)
    ids, meta = encode_clef(request, tokenizer, **kwargs)
    _assert_encoding(encoded, ids, meta)
    if case == "state-limit":
        return  # This lower-level option is not exposed by the trainer.
    processor = ClefPreprocessor(tokenizer, 16384, len(tokenizer))
    batch = encode_request(request, preprocessor=processor, max_request_tokens=kwargs.get("max_length", 16384))
    sample = batch
    batch = DecisionCollator()([sample])
    reference = official.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
    for key in ("input_ids", "attention_mask"):
        torch.testing.assert_close(batch[key], reference[key], rtol=0, atol=0)
    gold = json.loads(row["gold"])
    for index, question in enumerate(encoded.questions):
        target = gold.get(question.question_id)
        expected_weight = target.get("weight", 1.0) if target and len(question.option_ids) > 1 else 0.0
        assert batch["decision_supervision"]["weights"][index].item() == expected_weight
        if target:
            probabilities = target["probabilities"]
            total = sum(probabilities.values())
            expected = torch.tensor([probabilities[key] / total for key in question.option_ids])
            torch.testing.assert_close(batch["decision_supervision"]["targets"][index, : len(expected)], expected)
    padding = DecisionCollator(sample)([None])
    assert padding["decision_supervision"]["padding"] == (True,)
    assert padding["decision_supervision"]["weights"].count_nonzero() == 0


def _models(official_assets, *, layer_types=None, ops=None):
    directory, official, tokenizer = official_assets
    torch.manual_seed(42)
    config = tiny_qwen3_5_config(layer_types=layer_types or ["linear_attention", "full_attention"])
    config.text_config.vocab_size = len(tokenizer)
    config.text_config.max_position_embeddings = 2048
    # Native HF config keeps the reference independent of the VeOmni builder.
    reference_config = Qwen3_5Config.from_dict(config.to_dict())
    for sub_config in (reference_config, reference_config.text_config, reference_config.vision_config):
        sub_config._attn_implementation = "eager"
    reference_backbone = Qwen3_5ForConditionalGeneration(reference_config).float()
    _pin_hf_gdn_to_torch(reference_backbone)
    foundation = _build_vlm(copy.deepcopy(config), ops=ops).float()
    foundation.load_state_dict(reference_backbone.state_dict(), strict=True)
    head_config = dict(hidden_size=64, width=32, routing_layers=1, layers=1, heads=4, feedforward=64, dropout=0.0)
    head = official.JointSchemaHead(**head_config)
    reference = official.ClefModel(reference_backbone, head).train()
    ours_head = copy.deepcopy(head)
    scalar_names = _reshape_scalar_parameters(ours_head)
    ours = ClefDecisionModel(
        foundation,
        ours_head,
        directory / "joint_head.safetensors",
        head_config=head_config,
        source_dir=directory,
        scalar_head_parameters=scalar_names,
    ).train()
    ours.freeze_model()
    reference.language_model.model.visual.requires_grad_(False)
    processor = ClefPreprocessor.from_assets(
        tokenizer,
        ours.config,
        position_hook=ours.get_position_id_func(),
    )
    row = _row()
    batch = encode_request(parse_typed_decisions_case(row).request, preprocessor=processor)
    batch = DecisionCollator()([batch])
    # This processor emits one unpadded sequence. Eager GDN accepts only the
    # equivalent dense layout, while production FLA consumes the [0, length]
    # varlen metadata. Keep full-attention boundaries unchanged.
    batch["linear_attn_cu_seq_lens_q"] = torch.empty(0, dtype=torch.int32)
    encoded = official.encode_record(tokenizer, _record(row))
    reference_batch = official.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
    return reference, ours, batch, reference_batch, row


def _reference_loss(per_record, records, row, loss_type):
    """Apply a specified objective to official outputs; not an official training recipe."""
    gold = json.loads(row["gold"])
    terms, weights = [], []
    for logits, question in zip(per_record[0], records[0].questions, strict=True):
        label = gold.get(question.question_id)
        if label is None or len(question.option_ids) == 1:
            continue
        probabilities = label["probabilities"]
        target = logits.new_tensor([probabilities[key] / sum(probabilities.values()) for key in question.option_ids])
        ce = F.cross_entropy(logits[None], target[None], label_smoothing=0.1 if loss_type == "smooth_brier" else 0.0)
        if loss_type == "smooth_brier":
            ce = ce + F.mse_loss(logits.softmax(-1), target)
        elif loss_type == "distill_rps" and question.question_type == 2:
            ce = ce + 0.5 * F.mse_loss(logits.softmax(-1).cumsum(-1)[:-1], target.cumsum(-1)[:-1])
        weight = label.get("weight", 1.0)
        terms.append(ce * weight)
        weights.append(weight)
    return torch.stack(terms).sum() / sum(weights)


def _reference_parameters(reference):
    return {name.removeprefix("language_model."): value for name, value in reference.named_parameters()}


@pytest.mark.parametrize("loss_type", ["soft_ce", "smooth_brier", "distill_rps"])
def test_official_forward_backward_and_adamw_step(official_assets, loss_type, record_property, monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    reference, ours, batch, reference_batch, row = _models(official_assets)
    hidden = {}

    def capture(name):
        def hook(module, inputs, output):
            hidden[name] = output.last_hidden_state.detach()

        return hook

    handles = [
        reference.language_model.model.language_model.register_forward_hook(capture("reference")),
        ours.model.language_model.register_forward_hook(capture("ours")),
    ]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) for model in (reference, ours)]
    try:
        expected_logits = reference(reference_batch)
        output = ours(batch)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(hidden["ours"], hidden["reference"], atol=1e-6, rtol=1e-5)
    logits_error = 0.0
    for index, expected in enumerate(expected_logits[0]):
        actual = output.logits[index, : expected.numel()]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), atol=1e-6, rtol=1e-5)
        logits_error = max(logits_error, (actual - expected).abs().max().item())
    numerator, weight = get_decision_model("clef").build_loss(loss_type)(output, batch, "cpu")
    expected_loss = _reference_loss(expected_logits, reference_batch["records"], row, loss_type)
    # The trainer computes this denominator independently of the objective.
    torch.testing.assert_close(weight, torch.tensor(4.0), atol=0, rtol=0)
    torch.testing.assert_close(numerator, expected_loss * 4.0, atol=1e-6, rtol=1e-5)
    actual_loss = numerator / 4.0
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-5)
    for logits in expected_logits[0]:
        logits.retain_grad()
    actual_loss.backward()
    expected_loss.backward()
    reference_params = _reference_parameters(reference)
    assert reference_params.keys() == dict(ours.named_parameters()).keys()
    gradient_error = 0.0
    for name, actual in ours.named_parameters():
        expected = reference_params[name]
        assert (actual.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            expected_grad = expected.grad.reshape_as(actual.grad)
            torch.testing.assert_close(actual.grad, expected_grad, atol=1e-6, rtol=1e-4, msg=name)
            gradient_error = max(gradient_error, (actual.grad - expected_grad).abs().max().item())
    for name in ("lm_head.weight", "model.language_model.embed_tokens.weight", "head.residual_gate"):
        assert reference_params[name].grad.abs().sum() > 0, name
    # Isolate the model/optimizer comparison from loss reduction roundoff.
    # The independent objectives above agree to FP32 precision, but AdamW
    # amplifies near-zero cancellation residuals in the logit-offset bias.
    common_gradients = [
        logits.grad.detach().clone() if logits.grad is not None else torch.zeros_like(logits)
        for logits in expected_logits[0]
    ]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    repeated_reference = reference(reference_batch)[0]
    repeated_output = ours(batch)
    repeated_actual = [
        repeated_output.logits[index, : gradient.numel()] for index, gradient in enumerate(common_gradients)
    ]
    torch.autograd.backward(repeated_reference, common_gradients)
    torch.autograd.backward(repeated_actual, common_gradients)
    for name, actual in ours.named_parameters():
        expected = reference_params[name]
        assert (actual.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(
                actual.grad, expected.grad.reshape_as(actual.grad), atol=1e-6, rtol=1e-4, msg=name
            )
    norms = [torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) for model in (reference, ours)]
    torch.testing.assert_close(norms[0], norms[1], atol=1e-6, rtol=1e-5)
    for optimizer in optimizers:
        optimizer.step()
    update_error = 0.0
    for name, actual in ours.named_parameters():
        expected = reference_params[name].reshape_as(actual)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5, msg=name)
        update_error = max(update_error, (actual - expected).abs().max().item())
    metrics = dict(
        revision=OFFICIAL_REVISION,
        loss_type=loss_type,
        hidden_max_abs=(hidden["ours"] - hidden["reference"]).abs().max().item(),
        logits_max_abs=logits_error,
        loss_abs=abs(actual_loss.item() - expected_loss.item()),
        gradient_max_abs=gradient_error,
        grad_norm_abs=abs(norms[0].item() - norms[1].item()),
        parameter_max_abs=update_error,
    )
    record_property("clef_parity", json.dumps(metrics))
    print(json.dumps(metrics))


def test_exported_head_loads_in_unmodified_official_class(official_assets, tmp_path, monkeypatch, rank_zero):
    from safetensors.torch import load_file

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    _, ours, batch, reference_batch, _ = _models(official_assets)
    _, official, _ = official_assets
    scalar_values = {"prior_logit_scale": 0.6, "joint_logit_scale": -0.4, "residual_gate": 0.3}
    with torch.no_grad():
        for name, value in scalar_values.items():
            getattr(ours.head, name).fill_(value)
    save_clef_release(
        tmp_path,
        ours.state_dict(),
        head_config=ours.head_config,
        source_dir=ours.source_dir,
        model_assets=[ours.config],
        scalar_head_parameters=ours.scalar_head_parameters,
    )
    exported_head = load_file(tmp_path / "joint_head.safetensors")
    for name in scalar_values:
        assert exported_head[name].shape == torch.Size()
    head = official.JointSchemaHead(**json.loads((tmp_path / "joint_head_config.json").read_text()))
    head.load_state_dict(exported_head, strict=True)
    for name in scalar_values:
        torch.testing.assert_close(getattr(head, name), getattr(ours.head, name).reshape(()), atol=0, rtol=0)
    reloaded = load_local_model(
        tmp_path, init_device="cpu", torch_dtype="float32", ops_implementation=make_eager_ops_config()
    ).train()
    with torch.no_grad():
        features = model_features(batch)
        features["position_ids"] = features["position_ids"].transpose(0, 1).contiguous()
        hidden = ours.model.language_model(**features, use_cache=False, return_dict=True).last_hidden_state
        expected = head(
            hidden,
            reference_batch["input_ids"],
            reference_batch["attention_mask"],
            reference_batch["records"],
            ours.get_output_embeddings().weight,
        )
        output = ours(batch)
        torch.testing.assert_close(reloaded(batch).logits, output.logits)
    for index, logits in enumerate(expected[0]):
        torch.testing.assert_close(output.logits[index, : logits.numel()], logits)


@pytest.mark.parametrize(
    "backend",
    ["cpu_full", pytest.param("cuda_mixed", marks=pytest.mark.launch_plan(provider=LaunchSpec(nproc=1)))],
)
def test_packed_requests_match_independent_logits_and_gradients(official_assets, backend, monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    ops = make_eager_ops_config()
    if backend == "cuda_mixed":
        if not IS_CUDA_AVAILABLE:
            pytest.skip("Packed mixed attention requires CUDA FLA kernels")
        device = get_device_type()
        monkeypatch.setattr(getattr(torch.backends, device).matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        ops.causal_conv1d_implementation = "fla"
        ops.chunk_gated_delta_rule_implementation = "fla"
        layer_types = ["linear_attention", "full_attention"]
    else:
        layer_types = ["full_attention", "full_attention"]
        device = "cpu"
    reference, model, _, _, _ = _models(official_assets, layer_types=layer_types, ops=ops)
    del reference
    model = model.to(device)
    _, _, tokenizer = official_assets
    processor = ClefPreprocessor.from_assets(tokenizer, model.config, position_hook=model.get_position_id_func())
    first = _row()
    second = _row()
    second["state"] = json.dumps("A different and much shorter observation.")
    # Vary both question count and maximum candidate count between requests.
    questions, gold = json.loads(second["questions"]), json.loads(second["gold"])
    second["questions"] = json.dumps({"intact": questions["intact"]})
    second["gold"] = json.dumps({"intact": gold["intact"]})
    samples = [
        encode_request(parse_typed_decisions_case(row).request, preprocessor=processor) for row in (first, second)
    ]
    collator = DecisionCollator()
    packed = collator(samples)
    loss_fn = get_decision_model("clef").build_loss("soft_ce")
    separate_outputs = []
    separate_loss = 0
    total_weight = 0
    for sample in samples:
        batch = collator([sample])
        output = model(batch)
        separate_outputs.append(output.logits.detach())
        numerator, weight = loss_fn(output, batch, device)
        separate_loss = separate_loss + numerator
        total_weight = total_weight + weight
    (separate_loss / total_weight).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    output = model(packed)
    width = output.logits.shape[-1]
    expected = torch.cat(
        [F.pad(logits, (0, width - logits.shape[-1]), value=-torch.inf) for logits in separate_outputs]
    )
    torch.testing.assert_close(output.logits, expected, atol=2e-5, rtol=2e-4)
    numerator, weight = loss_fn(output, packed, device)
    torch.testing.assert_close(numerator, separate_loss.detach(), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(weight, total_weight)
    (numerator / weight).backward()
    for name, parameter in model.named_parameters():
        assert (parameter.grad is not None) == (name in gradients), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-5, rtol=2e-4, msg=name)
