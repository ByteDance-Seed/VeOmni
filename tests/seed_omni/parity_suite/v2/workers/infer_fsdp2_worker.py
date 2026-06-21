"""SeedOmni V2 FSDP2 inference smoke worker for the parity suite."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from tests.seed_omni.parity_suite.core import configure_torch_determinism, to_device  # noqa: E402
from tests.seed_omni.parity_suite.core.config.probes import ProbeCatalog, load_probe_catalog  # noqa: E402
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer  # noqa: E402
from veomni.arguments import OmniArguments, parse_omni_args  # noqa: E402
from veomni.trainer.omni_inferencer import OmniInferencer  # noqa: E402


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set for parity-suite inference FSDP2 smoke.")
    return Path(value)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))


def _count_fsdp_modules(model: torch.nn.Module) -> int:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return 0
    return sum(1 for module in model.modules() if isinstance(module, FSDPModule))


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def _fresh_request(request_kwargs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return to_device(deepcopy(dict(request_kwargs)), device)


def _inferencer_device(inferencer: OmniInferencer) -> torch.device:
    base_device = getattr(inferencer.base, "device", None)
    if isinstance(base_device, torch.device):
        return base_device
    for param in inferencer.model.parameters():
        if param.device.type != "meta":
            return param.device
    return (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def _load_probe_catalog(probes_path: Path) -> ProbeCatalog:
    return load_probe_catalog(probes_path)


def _build_whitelist(probe_names: tuple[str, ...], *, probes_path: Path) -> dict[tuple[str, str], frozenset[str]]:
    catalog = _load_probe_catalog(probes_path)
    selected = catalog.for_probe_names(probe_names)
    whitelist: dict[tuple[str, str], set[str]] = {}
    for mapping in selected:
        state = mapping.state or "prompt_encode"
        key = (state, mapping.node)
        whitelist.setdefault(key, set()).add(mapping.v2_field)
    return {key: frozenset(fields) for key, fields in whitelist.items()}


def _extract_probe_values(
    observations: Mapping[tuple[str, str], list[dict[str, Any]]],
    probe_names: tuple[str, ...],
    *,
    probes_path: Path,
) -> dict[str, Any]:
    catalog = _load_probe_catalog(probes_path)
    extracted: dict[str, Any] = {}
    for probe_name in probe_names:
        mappings = catalog.for_probe_names((probe_name,))
        if not mappings:
            continue
        mapping = mappings[0]
        state = mapping.state or "prompt_encode"
        records = observations.get((state, mapping.node), ())
        values = [record[mapping.v2_field] for record in records if mapping.v2_field in record]
        if not values:
            continue
        value = values[-1]
        if mapping.step == "all" and len(values) > 1:
            value = [item.detach().cpu() if torch.is_tensor(item) else item for item in values]
        elif torch.is_tensor(value):
            value = value.detach().cpu()
        extracted[probe_name] = value
    return extracted


def _values_are_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, list):
        return all(_values_are_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _observations_are_finite(observations: Mapping[str, Any]) -> bool:
    return all(_values_are_finite(value) for value in observations.values())


def main() -> None:
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_omni_args(OmniArguments, preload_path_fields=("model.modules", "infer.modules"))
    payload = torch.load(_env_path("VEOMNI_PARITY_INFER_FSDP2_PAYLOAD"), map_location="cpu", weights_only=False)
    output_dir = _env_path("VEOMNI_PARITY_INFER_FSDP2_OUTPUT_DIR")
    seed = int(payload.get("seed", args.infer.seed))
    args.infer.seed = seed
    require_fsdp_modules = bool(payload.get("require_fsdp_modules", True))
    probe_names = tuple(str(name) for name in payload.get("probe_names", ()) or ())
    probes_path = Path(str(payload["probes_path"]))
    request_kwargs = payload["request_kwargs"]
    generation_kwargs = dict(payload.get("generation_kwargs", {}) or {})

    inferencer = OmniInferencer(args)
    base = inferencer.base
    device = _inferencer_device(inferencer)
    configure_torch_determinism(seed)
    request = _fresh_request(request_kwargs, device)
    whitelist = _build_whitelist(probe_names, probes_path=probes_path) if probe_names else {}

    try:
        trace_buf: list[str] = []
        with torch.no_grad():
            if whitelist:
                observer_context = arm_generation_observer(whitelist)
            else:
                from contextlib import nullcontext

                observer_context = nullcontext({})
            with observer_context as observations:
                ctx = inferencer.model.generate(
                    request=request,
                    trace=trace_buf,
                    generation_kwargs=generation_kwargs,
                )
        inferencer.finalize(ctx, output_dir=str(output_dir))
        rank0_finalized = _rank() == 0 and (output_dir / "trace.txt").exists()
        extracted = _extract_probe_values(observations, probe_names, probes_path=probes_path)
        fsdp_module_count = _count_fsdp_modules(inferencer.model)
        finite_observations = _observations_are_finite(extracted) if extracted else True

        rank_metrics = {
            "rank": _rank(),
            "world_size": _world_size(),
            "fsdp_module_count": fsdp_module_count,
            "finite_observations": finite_observations,
            "trace_has_prompt_encode": any("prompt_encode" in line for line in trace_buf),
            "trace_has_image_flow": any("image_flow" in line for line in trace_buf),
            "rank0_finalized": rank0_finalized,
            "observations": _jsonable(extracted),
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(_world_size())]
        if dist.is_initialized():
            dist.all_gather_object(gathered, rank_metrics)
        else:
            gathered[0] = rank_metrics

        if _rank() == 0:
            all_pass = all(
                item is not None
                and item["finite_observations"]
                and item["trace_has_prompt_encode"]
                and item["trace_has_image_flow"]
                and (item["fsdp_module_count"] > 0 or not require_fsdp_modules)
                for item in gathered
            )
            report = {
                "all_pass": all_pass,
                "finite_observations": all(item["finite_observations"] for item in gathered if item),
                "trace_has_prompt_encode": all(item["trace_has_prompt_encode"] for item in gathered if item),
                "trace_has_image_flow": all(item["trace_has_image_flow"] for item in gathered if item),
                "rank0_finalized": rank0_finalized,
                "fsdp_module_count": fsdp_module_count,
                "expected_fsdp_module_count": 5 if require_fsdp_modules else 0,
                "require_fsdp_modules": require_fsdp_modules,
                "observations": _jsonable(extracted),
                "ranks": gathered,
            }
            _write_report(output_dir, report)
            if not all_pass:
                raise AssertionError(report)
    finally:
        if dist.is_initialized():
            base.destroy_distributed()


if __name__ == "__main__":
    main()
