from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core.config.probes import ProbeMapping
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.driver.observations import shifted_label_rows_from_conversation


class BagelParityDriver(ParityDriver):
    def runtime_sdpa_kernel_modules(self) -> tuple[Any, ...]:
        # Official BAGEL and V2 BAGEL each bind sdpa_kernel in their own module.
        # The deterministic-SDPA runtime option patches both globals with the
        # same semantic policy during reference and V2 phases.
        import tests.seed_omni.bagel.reference.vendor.modeling.bagel.qwen2_navit as ref_qwen2_navit
        import veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling as v2_qwen2_mot

        return (ref_qwen2_navit, v2_qwen2_mot)

    def gradient_rows(self, batch: dict[str, Any], mapping: ProbeMapping) -> torch.Tensor | None:
        if mapping.v2_field == "train_grad_lm_head_rows":
            labels = batch.get("_bagel_train_label_ids")
            if torch.is_tensor(labels):
                return torch.unique(labels.detach().cpu()).to(dtype=torch.long)
            return shifted_label_rows_from_conversation(batch.get("conversation_list"))
        return None


def create_driver(case) -> BagelParityDriver:
    return BagelParityDriver(case)
