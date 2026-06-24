from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import sample_named_param
from tests.seed_omni.parity_suite.core.config.probes import ProbeMapping
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.driver.observations import shifted_label_rows_from_conversation
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext
from veomni.models.seed_omni.modeling_omni import OmniModel


_LM_HEAD_SAMPLE_ROWS_KEY = "lm_head_sample_rows"
_FLOW_NOISE_KEY = "_bagel_parity_flow_noise"
_FLOW_TIMESTEP_LOGITS_KEY = "_bagel_parity_flow_timestep_logits"


class BagelParityDriver(ParityDriver):
    def runtime_sdpa_kernel_modules(self) -> tuple[Any, ...]:
        # Official BAGEL and V2 BAGEL bind attention helpers in module globals.
        # The deterministic-SDPA runtime option patches these globals with the
        # same semantic policy during reference and V2 phases.
        import tests.seed_omni.bagel.parity.reference.vendor.modeling.bagel.qwen2_navit as ref_qwen2_navit
        import tests.seed_omni.bagel.parity.reference.vendor.modeling.bagel.siglip_navit as ref_siglip_navit
        import veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling as v2_qwen2_mot
        import veomni.models.seed_omni.modules.bagel.siglip_navit.modeling as v2_siglip_navit

        return (ref_siglip_navit, ref_qwen2_navit, v2_siglip_navit, v2_qwen2_mot)

    def v2_gradient_rows(
        self,
        ctx: V2RunContext,
        batch: dict[str, Any],
        mapping: ProbeMapping,
    ) -> torch.Tensor | None:
        del ctx
        if mapping.v2_field == "train_grad_lm_head_rows":
            labels = batch.get("_bagel_train_label_ids")
            if torch.is_tensor(labels):
                return torch.unique(labels.detach().cpu()).to(dtype=torch.long)
            return shifted_label_rows_from_conversation(batch.get("conversation_list"))
        return None

    def v2_parameter_samples(
        self,
        ctx: V2RunContext,
        model: OmniModel,
        sample_context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        del ctx
        label_rows = _framework_lm_head_rows(sample_context)
        return {
            "qwen_early_q_proj": sample_named_param(
                model.get_module("bagel_qwen2_mot"),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            "lm_head_rows": sample_named_param(
                model.get_module("bagel_text_encoder"),
                "lm_head.weight",
                rows=label_rows,
            ),
            "flow_llm2vae": sample_named_param(
                model.get_module("bagel_flow_connector"),
                "llm2vae.weight",
            ),
        }

    def v2_parameter_sample_context(
        self,
        ctx: V2RunContext,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        del ctx
        label_rows = _framework_lm_head_rows(batch)
        if label_rows is None:
            return {}
        return {_LM_HEAD_SAMPLE_ROWS_KEY: label_rows.detach().cpu().to(dtype=torch.long)}

    def build_v2_request(self, ctx: V2RunContext) -> dict[str, Any]:
        request = super().build_v2_request(ctx)
        if ctx.domain == "training":
            _attach_reference_flow_noising(request, ctx.canonical, device=ctx.device)
        return request

    def v2_execution_context(
        self, ctx: V2RunContext, *, model: Any | None = None, batch: dict[str, Any] | None = None
    ):
        del model, batch
        if ctx.domain != "training" or ctx.tier != "graph":
            return nullcontext()
        return _patch_flow_noising_from_carrier_meta()


def _framework_lm_head_rows(batch: dict[str, Any]) -> torch.Tensor | None:
    rows = batch.get(_LM_HEAD_SAMPLE_ROWS_KEY)
    if torch.is_tensor(rows):
        return rows.detach().cpu().to(dtype=torch.long)
    labels = batch.get("_bagel_train_label_ids")
    if torch.is_tensor(labels):
        return torch.unique(labels.detach().cpu()).to(dtype=torch.long)
    return shifted_label_rows_from_conversation(batch.get("conversation_list"))


def _attach_reference_flow_noising(
    request: dict[str, Any],
    canonical: dict[str, Any],
    *,
    device: torch.device,
) -> None:
    train_batch = canonical.get("train_batch")
    if not isinstance(train_batch, dict):
        return
    fixed_noise = train_batch.get("fixed_noise")
    packed_timesteps = train_batch.get("packed_timesteps")
    latent_shapes = train_batch.get("patchified_vae_latent_shapes")
    conversation_list = request.get("conversation_list")
    if (
        not torch.is_tensor(fixed_noise)
        or not torch.is_tensor(packed_timesteps)
        or not isinstance(latent_shapes, list)
        or not isinstance(conversation_list, list)
    ):
        return

    image_items = [
        item
        for sample in conversation_list
        if isinstance(sample, list)
        for item in sample
        if getattr(item, "type", None) == "image" and getattr(item, "role", None) == "assistant"
    ]
    offset = 0
    for item, grid_shape in zip(image_items, latent_shapes, strict=False):
        token_count = int(grid_shape[0]) * int(grid_shape[1])
        item.meta[_FLOW_NOISE_KEY] = fixed_noise[offset : offset + token_count].to(device=device)
        item.meta[_FLOW_TIMESTEP_LOGITS_KEY] = packed_timesteps[offset : offset + token_count].to(device=device)
        offset += token_count


@contextmanager
def _patch_flow_noising_from_carrier_meta() -> Iterator[None]:
    """Make V2 flow noising consume parity-authored carrier metadata.

    Official train parity patches BAGEL's random flow noise with fixed fixture
    tensors. The V2 production flow connector should keep sampling normally, so
    this patch is scoped to the test-side flow preprocessing call.
    """

    import veomni.models.seed_omni.modules.bagel.flow_connector.modulemixin as flow_mixin

    original_preprocess = flow_mixin.preprocess_latent_embed

    def preprocess_with_fixed_noising(embed_items, *args, **kwargs):
        fixed_noise, fixed_timestep_logits = _fixed_flow_noising_from_items(
            embed_items,
            timestep_shift=float(kwargs.get("timestep_shift", 1.0)),
        )
        if fixed_noise is None or fixed_timestep_logits is None:
            return original_preprocess(embed_items, *args, **kwargs)
        with _patch_flow_random_calls(fixed_noise, fixed_timestep_logits):
            return original_preprocess(embed_items, *args, **kwargs)

    flow_mixin.preprocess_latent_embed = preprocess_with_fixed_noising
    try:
        yield
    finally:
        flow_mixin.preprocess_latent_embed = original_preprocess


def _fixed_flow_noising_from_items(
    embed_items: list[Any],
    *,
    timestep_shift: float,
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
    fixed_noise: list[torch.Tensor] = []
    fixed_timestep_logits: list[torch.Tensor] = []
    for item in embed_items:
        meta = getattr(item, "meta", {})
        noise = meta.get(_FLOW_NOISE_KEY) if isinstance(meta, dict) else None
        timestep_logits = meta.get(_FLOW_TIMESTEP_LOGITS_KEY) if isinstance(meta, dict) else None
        if torch.is_tensor(noise) and torch.is_tensor(timestep_logits):
            fixed_noise.append(noise.detach())
            fixed_timestep_logits.append(timestep_logits.detach().to(dtype=torch.float32).reshape(-1))
            continue

        noise = meta.get("noise") if isinstance(meta, dict) else None
        timestep = meta.get("timestep") if isinstance(meta, dict) else None
        if not torch.is_tensor(noise) or not torch.is_tensor(timestep):
            return None, None
        fixed_noise.append(noise.detach())
        fixed_timestep_logits.append(_timestep_to_logits(timestep.detach(), timestep_shift=timestep_shift))
    return fixed_noise, fixed_timestep_logits


@contextmanager
def _patch_flow_random_calls(
    fixed_noise: list[torch.Tensor],
    fixed_timestep_logits: list[torch.Tensor],
) -> Iterator[None]:
    original_randn_like = torch.randn_like
    original_randn = torch.randn
    noise_queue = list(fixed_noise)
    timestep_queue = list(fixed_timestep_logits)

    def fixed_randn_like(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        if not noise_queue:
            return original_randn_like(input_tensor)
        noise = noise_queue.pop(0).to(device=input_tensor.device, dtype=input_tensor.dtype)
        if noise.shape != input_tensor.shape:
            raise ValueError(
                f"BAGEL parity fixed flow noise shape mismatch: got {tuple(noise.shape)}, "
                f"expected {tuple(input_tensor.shape)}."
            )
        return noise

    def fixed_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
        if not timestep_queue:
            return original_randn(*size, **kwargs)
        logits = timestep_queue.pop(0)
        expected_shape = tuple(int(dim) for dim in size)
        if logits.shape != expected_shape:
            raise ValueError(
                f"BAGEL parity fixed timestep-logit shape mismatch: got {tuple(logits.shape)}, "
                f"expected {expected_shape}."
            )
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        return logits.to(
            device=device if device is not None else logits.device,
            dtype=dtype if dtype is not None else logits.dtype,
        )

    torch.randn_like = fixed_randn_like
    torch.randn = fixed_randn
    try:
        yield
    finally:
        torch.randn_like = original_randn_like
        torch.randn = original_randn


def _timestep_to_logits(timestep: torch.Tensor, *, timestep_shift: float) -> torch.Tensor:
    timestep = timestep.to(dtype=torch.float32).reshape(-1)
    shift = float(timestep_shift)
    if shift != 1.0:
        timestep = timestep / (shift - timestep * (shift - 1.0))
    eps = torch.finfo(torch.float32).eps
    timestep = timestep.clamp(min=eps, max=1.0 - eps)
    return torch.logit(timestep)


def create_driver(case) -> BagelParityDriver:
    return BagelParityDriver(case)
