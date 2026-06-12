"""Declarative reference capture plan support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import yaml

from tests.seed_omni.parity_suite.core.probes import ProbeBinding
from tests.seed_omni.parity_suite.core.spec import CaseSpec


@dataclass(frozen=True)
class CapturePlan:
    id: str
    action: str
    description: str = ""
    seed: int = 1234
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    fixture: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureOutput:
    probe: str
    path: str
    value: str


def load_capture_plan(case: CaseSpec) -> CapturePlan:
    """Load the reusable capture plan selected by a case."""

    if not case.capture:
        raise ValueError(f"{case.node_id} does not declare a capture plan.")
    path = case.source_path.with_name("capture.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = dict(data.get("defaults") or {})
    raw_plan = _raw_capture_plan(data, case.capture)
    if not isinstance(raw_plan, dict):
        raise KeyError(f"Capture plan {case.capture!r} not found in {path}.")
    default_fixture = dict(defaults.get("fixture") or {})
    plan_fixture = dict(raw_plan.get("fixture") or {})
    fixture = {**default_fixture, **plan_fixture}
    if "tolerance" in default_fixture or "tolerance" in plan_fixture:
        fixture["tolerance"] = {
            **dict(default_fixture.get("tolerance") or {}),
            **dict(plan_fixture.get("tolerance") or {}),
        }
    return CapturePlan(
        id=case.capture,
        action=str(raw_plan.get("action") or ""),
        description=str(raw_plan.get("description") or ""),
        seed=int(raw_plan.get("seed") if raw_plan.get("seed") is not None else defaults.get("seed", 1234)),
        inputs=dict(raw_plan.get("inputs") or {}),
        outputs=dict(raw_plan.get("outputs") or {}),
        fixture=fixture,
    )


def _raw_capture_plan(data: dict[str, Any], capture_id: str) -> dict[str, Any] | None:
    plans = data.get("plans") or {}
    if isinstance(plans, dict) and isinstance(plans.get(capture_id), dict):
        return dict(plans[capture_id])
    scenarios = data.get("scenarios") or {}
    if not isinstance(scenarios, dict) or not isinstance(scenarios.get(capture_id), dict):
        return None
    scenario = dict(scenarios[capture_id])
    reference = dict(scenario.get("reference") or {})
    if not reference:
        return None
    return {
        "action": reference.get("action"),
        "description": scenario.get("description", ""),
        "seed": scenario.get("seed"),
        "inputs": _resolve_template_map(data, reference.get("inputs"), "input_templates"),
        "outputs": _resolve_template_map(data, reference.get("outputs"), "output_templates"),
        "fixture": reference.get("fixture") or {},
    }


def _resolve_template_map(data: dict[str, Any], raw_value: Any, template_key: str) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise TypeError(f"`{template_key.removesuffix('_templates')}` must be a map, got {type(raw_value).__name__}")
    catalog = data.get(template_key) or {}
    if not isinstance(catalog, dict):
        raise TypeError(f"`{template_key}` must be a map, got {type(catalog).__name__}")

    resolved: dict[str, Any] = {}
    templates = raw_value.get("templates", []) or []
    if isinstance(templates, str):
        templates = [templates]
    for template_name in templates:
        template = catalog.get(str(template_name))
        if not isinstance(template, dict):
            raise KeyError(f"Template {template_name!r} not found in `{template_key}`.")
        resolved.update(template)

    values = raw_value.get("values") or {}
    if not isinstance(values, dict):
        raise TypeError(f"`values` in templated `{template_key}` must be a map, got {type(values).__name__}")
    resolved.update(values)
    for key, value in raw_value.items():
        if key not in {"templates", "values"}:
            resolved[str(key)] = value
    return resolved


def capture_output(plan: CapturePlan, probe: str) -> CaptureOutput:
    raw_output = plan.outputs.get(probe)
    if not isinstance(raw_output, dict):
        raise KeyError(f"Capture plan {plan.id!r} does not define output for probe {probe!r}.")
    path = raw_output.get("path")
    value = raw_output.get("value")
    if not isinstance(path, str) or not path:
        raise KeyError(f"Capture plan {plan.id!r} output {probe!r} is missing `path`.")
    if not isinstance(value, str) or not value:
        raise KeyError(f"Capture plan {plan.id!r} output {probe!r} is missing `value`.")
    return CaptureOutput(probe=probe, path=path, value=value)


def execute_capture_plan(
    case: CaseSpec,
    plan: CapturePlan,
    probe_bindings: dict[str, ProbeBinding],
    *,
    model: torch.nn.Module,
    config: Any,
) -> dict[str, Any]:
    """Execute a suite-supported capture plan and return fixture-shaped data."""

    if plan.action not in {"transformers_causal_lm.forward", "transformers_causal_lm.forward_backward"}:
        raise NotImplementedError(f"Unsupported capture action for {case.node_id}: {plan.action}")

    seed = int(case.case.get("seed", plan.seed))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = _tensor_from_spec(plan.inputs["input_ids"], config=config, generator=generator)
    labels = _tensor_from_spec(plan.inputs.get("labels", plan.inputs["input_ids"]), config=config, generator=generator)
    model.train(plan.action.endswith("forward_backward"))
    outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
    values = {
        "hidden_state": outputs.hidden_states[-1].detach().cpu(),
        "logits": outputs.logits.detach().cpu(),
        "greedy_token": outputs.logits[:, -1:].argmax(dim=-1).detach().cpu(),
        "loss": outputs.loss.detach().cpu().reshape(1),
    }
    values["cache_after_step"] = {
        "num_layers": getattr(config, "num_hidden_layers", 1),
        "key": [values["hidden_state"] for _ in range(getattr(config, "num_hidden_layers", 1))],
        "value": [values["hidden_state"].clone() for _ in range(getattr(config, "num_hidden_layers", 1))],
    }
    if plan.action.endswith("forward_backward"):
        if outputs.loss is None:
            raise RuntimeError(f"{case.node_id} capture plan requested backward without a loss.")
        outputs.loss.backward()
        values.update(_gradient_values(model))

    fixture = _fixture_skeleton(case, plan)
    for probe in case.probes:
        if probe not in probe_bindings:
            continue
        output = capture_output(plan, probe)
        _set_path(fixture, output.path, values[output.value])
    return fixture


def _fixture_skeleton(case: CaseSpec, plan: CapturePlan) -> dict[str, Any]:
    dtype = str(plan.fixture.get("dtype", "fp32"))
    return {
        "metadata": {
            "case_id": case.id,
            "source": "online_capture",
            "capture_plan": plan.id,
            "backend": "transformers",
            "domain": case.domain,
            "level": case.level,
            "dtype": dtype,
            "seed": int(case.case.get("seed", plan.seed)),
        },
        "prepared": {},
        "one_step": {},
        "losses": {},
        "gradients": {},
        "steps": {},
        "tolerances": {
            dtype: dict(
                plan.fixture.get(
                    "tolerance",
                    {
                        "max_abs_diff": 0.0,
                        "mean_abs_diff": 0.0,
                        "relative_l2_max": 0.0,
                        "cosine_similarity_min": 1.0,
                    },
                )
            )
        },
    }


def _tensor_from_spec(spec: dict[str, Any], *, config: Any, generator: torch.Generator) -> torch.Tensor:
    kind = spec.get("kind", "randint")
    shape = tuple(int(dim) for dim in spec.get("shape", [1, 4]))
    if kind == "randint":
        high = spec.get("high")
        if high == "config.vocab_size" or high is None:
            high = int(config.vocab_size)
        low = int(spec.get("low", 0))
        return torch.randint(low, int(high), shape, generator=generator, dtype=torch.long)
    if kind == "full":
        return torch.full(shape, int(spec.get("value", 0)), dtype=torch.long)
    raise NotImplementedError(f"Unsupported capture tensor input kind: {kind}")


def _gradient_values(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    embed_tokens = getattr(model, "embed_tokens", None)
    if embed_tokens is not None and getattr(embed_tokens, "weight", None) is not None:
        grad = embed_tokens.weight.grad
        if grad is not None:
            gradients["grad_text_embedding"] = grad.detach().cpu()
    layers = getattr(model, "layers", None)
    if layers is not None and len(layers) > 0:
        early_grad = _first_parameter_grad(layers[0])
        if early_grad is not None:
            gradients["grad_qwen_early"] = early_grad
        late_grad = _first_parameter_grad(layers[-1])
        if late_grad is not None:
            gradients["grad_qwen_late"] = late_grad
    lm_head = getattr(model, "lm_head", None)
    generation_grad = _first_parameter_grad(lm_head) if lm_head is not None else None
    if generation_grad is not None:
        gradients["grad_qwen_generation"] = generation_grad
    return gradients


def _first_parameter_grad(module: torch.nn.Module) -> torch.Tensor | None:
    for parameter in module.parameters():
        if parameter.grad is not None:
            return parameter.grad.detach().cpu()
    return None


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    cursor = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value
