from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core import sample_named_param
from tests.seed_omni.parity_suite.core.config.probes import ProbeMapping
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.driver.observations import shifted_label_rows_from_conversation
from veomni.models.seed_omni.modeling_omni import OmniModel


_LM_HEAD_SAMPLE_ROWS_KEY = "lm_head_sample_rows"


class BagelParityDriver(ParityDriver):
    def runtime_sdpa_kernel_modules(self) -> tuple[Any, ...]:
        # Official BAGEL and V2 BAGEL each bind sdpa_kernel in their own module.
        # The deterministic-SDPA runtime option patches both globals with the
        # same semantic policy during reference and V2 phases.
        import tests.seed_omni.bagel.parity.reference.vendor.modeling.bagel.qwen2_navit as ref_qwen2_navit
        import veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling as v2_qwen2_mot

        return (ref_qwen2_navit, v2_qwen2_mot)

    def gradient_rows(self, batch: dict[str, Any], mapping: ProbeMapping) -> torch.Tensor | None:
        if mapping.v2_field == "train_grad_lm_head_rows":
            labels = batch.get("_bagel_train_label_ids")
            if torch.is_tensor(labels):
                return torch.unique(labels.detach().cpu()).to(dtype=torch.long)
            return shifted_label_rows_from_conversation(batch.get("conversation_list"))
        return None

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        label_rows = _framework_lm_head_rows(batch)
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

    def framework_parameter_sample_context(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        label_rows = _framework_lm_head_rows(batch)
        if label_rows is None:
            return {}
        return {_LM_HEAD_SAMPLE_ROWS_KEY: label_rows.detach().cpu().to(dtype=torch.long)}


def _framework_lm_head_rows(batch: dict[str, Any]) -> torch.Tensor | None:
    rows = batch.get(_LM_HEAD_SAMPLE_ROWS_KEY)
    if torch.is_tensor(rows):
        return rows.detach().cpu().to(dtype=torch.long)
    labels = batch.get("_bagel_train_label_ids")
    if torch.is_tensor(labels):
        return torch.unique(labels.detach().cpu()).to(dtype=torch.long)
    return shifted_label_rows_from_conversation(batch.get("conversation_list"))


def create_driver(case) -> BagelParityDriver:
    return BagelParityDriver(case)
