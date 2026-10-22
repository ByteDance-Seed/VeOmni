# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import math

from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as _DeepseekV4Config,
)

from veomni.utils import logging


logger = logging.get_logger(__name__)


class DeepseekV4Config(_DeepseekV4Config):
    """DeepSeek-V4 config with Lightning Indexer KL and DSpark MTP fields.

    The objectives and DSpark draft-stage shape are configured on the model
    config rather than on the ops config: ``output_router_logits`` /
    ``router_aux_loss_coef`` are fields of the model config, folded into the
    loss in ``DeepseekV4ForCausalLM.forward`` from ``self.config``, and
    ``dsa_indexer_loss`` / ``dsa_indexer_loss_coef`` as well as the DSpark
    ``mtp.*`` fields sit beside them and are read the same way. The neighbouring
    ``dsa_indexer_implementation`` / ``dsa_attention_implementation`` stay on
    the ops config, which is implementation selection and nothing else.

    Being *declared* here is load-bearing rather than tidiness. Overrides from
    ``model.model_config`` reach the config as ``**kwargs`` to
    ``PreTrainedConfig.from_dict``, which applies only those keys the
    constructed config already answers ``hasattr`` for and drops the rest
    silently -- no error, no warning. Declaring the fields here is what makes
    both ``config.json`` entries and ``model.model_config`` YAML overrides reach
    the model.

    Validated in ``validate_build_prerequisites`` below rather than in
    ``__post_init__``: ``from_dict`` runs ``__post_init__`` on the on-disk values
    and only then ``setattr``s the overrides, so a bound checked there would see
    ``config.json`` and never the YAML that contradicts it.
    """

    dsa_indexer_loss: bool = False
    dsa_indexer_loss_coef: float = 1.0
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list[int] | None = None
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int | None = None
    dspark_num_experts_per_tok: int | None = None
    mtp_loss_coef: float = 1.0
    freeze_mtp: bool = False

    def __post_init__(self, **kwargs):
        legacy_compress_ratios = kwargs.get("compress_ratios")
        legacy_n_mtp_layers = kwargs.pop("n_mtp_layers", None)
        super().__post_init__(**kwargs)
        if legacy_n_mtp_layers is not None:
            self.num_nextn_predict_layers = legacy_n_mtp_layers
        elif (
            legacy_compress_ratios is not None
            and len(legacy_compress_ratios) > self.num_hidden_layers
            and self.dspark_block_size > 0
            and self.num_nextn_predict_layers == 1
        ):
            # Released DeepSeek-V4-Flash checkpoints ship 43+3 entries in
            # ``compress_ratios`` (3 trailing zeros for ``mtp.0..2``) while leaving
            # ``num_nextn_predict_layers`` at the upstream default of 1.
            self.num_nextn_predict_layers = len(legacy_compress_ratios) - self.num_hidden_layers

        if self.dspark_target_layer_ids is None and self.num_hidden_layers > 0:
            self.dspark_target_layer_ids = list(range(max(0, self.num_hidden_layers - 3), self.num_hidden_layers))
        elif self.dspark_target_layer_ids is not None:
            self.dspark_target_layer_ids = list(self.dspark_target_layer_ids)

    def moe_config_for_layer(self, layer_idx: int) -> tuple[int, int]:
        """Return ``(n_routed_experts, num_experts_per_tok)`` for backbone or DSpark layer ``layer_idx``."""
        if layer_idx < self.num_hidden_layers:
            return self.n_routed_experts, self.num_experts_per_tok
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_num_experts_per_tok or self.num_experts_per_tok,
        )

    def _validate_mtp_prerequisites(self) -> None:
        if not isinstance(self.freeze_mtp, bool):
            raise TypeError(f"freeze_mtp must be a bool, got {self.freeze_mtp!r} ({type(self.freeze_mtp).__name__}).")
        mtp_coef = self.mtp_loss_coef
        if isinstance(mtp_coef, bool) or not isinstance(mtp_coef, (int, float)):
            raise TypeError(f"mtp_loss_coef must be a number, got {mtp_coef!r} ({type(mtp_coef).__name__}).")
        if not math.isfinite(mtp_coef) or mtp_coef < 0:
            raise ValueError(
                f"mtp_loss_coef={mtp_coef!r} must be finite and non-negative. Use 0.0 to switch the term off."
            )
        if self.num_nextn_predict_layers > 0 and self.dspark_block_size > 0:
            if not self.dspark_target_layer_ids:
                raise ValueError("dspark_target_layer_ids must be non-empty when DSpark MTP stages are enabled.")
            for layer_idx in self.dspark_target_layer_ids:
                if (
                    isinstance(layer_idx, bool)
                    or not isinstance(layer_idx, int)
                    or not (0 <= layer_idx < self.num_hidden_layers)
                ):
                    raise ValueError(
                        f"dspark_target_layer_ids contains {layer_idx!r}, outside [0, {self.num_hidden_layers})."
                    )

    def validate_build_prerequisites(self) -> None:
        """Refuse an invalid DSpark MTP or Lightning Indexer KL configuration before weight loading."""
        from veomni.ops.config import get_ops_config

        self._validate_mtp_prerequisites()

        enabled = self.dsa_indexer_loss
        if not isinstance(enabled, bool):
            # Checked before the early return, because the failure it catches is a
            # *truthy* one: YAML quotes make ``dsa_indexer_loss: "false"`` the string
            # "false", which switches the objective on and would sail through the
            # falsiness test below on its way to a run nobody asked for. Values under
            # ``model_config`` are an untyped ``Dict`` the whole way down and nothing
            # between the YAML and here coerces them.
            raise TypeError(
                f"dsa_indexer_loss must be a bool, got {enabled!r} ({type(enabled).__name__}). "
                "Values under model.model_config are passed through as written, so quote nothing: "
                "write dsa_indexer_loss: false, not dsa_indexer_loss: 'false'."
            )
        if not enabled:
            return

        # The weight is checked before the prerequisites, not after: a user who
        # switched the term off with the coefficient has not asked for a TileLang
        # indexer, and refusing their run over the configuration of a feature they
        # just disabled would be advice about the wrong thing. Same ordering as
        # ``_indexer_loss_enabled`` in the patched forward, for the same reason.
        coef = self.dsa_indexer_loss_coef
        if isinstance(coef, bool) or not isinstance(coef, (int, float)):
            raise TypeError(
                f"dsa_indexer_loss_coef must be a number, got {coef!r} ({type(coef).__name__}). "
                "Values under model.model_config are passed through as written, so quote nothing: "
                "write dsa_indexer_loss_coef: 0.5, not dsa_indexer_loss_coef: '0.5'."
            )
        if not math.isfinite(coef) or coef < 0:
            raise ValueError(
                f"dsa_indexer_loss_coef={coef!r} must be finite and non-negative: a negative weight "
                "flips the sign of the indexer KL and trains the Lightning Indexer away from its "
                "teacher, and a non-finite one destroys every other term in the total loss. Both "
                "surface as a loss curve rather than as an error. Use 0.0 to switch the term off."
            )
        if coef == 0:
            # A legitimate way to disable the objective without editing the flag, so not
            # an error -- but a run whose config says ``dsa_indexer_loss: true`` and which
            # then reports no indexer metric at all is the sort of thing someone spends an
            # afternoon on, and this is the only place that can say so once rather than
            # once per layer per forward.
            logger.warning_rank0(
                "dsa_indexer_loss is enabled but dsa_indexer_loss_coef is 0.0, so the indexer KL is "
                "switched off entirely: no teacher is computed, the Lightning Indexer receives no "
                "gradient, and no indexer metric is reported. Set a positive coefficient to train it."
            )
            return

        kernels_config = get_ops_config()
        if kernels_config is None:
            return

        for field_name, reason in (
            ("dsa_indexer_implementation", "the eager indexer discards the per-slot scores the KL trains against"),
            (
                "dsa_attention_implementation",
                "the teacher distribution is derived from the TileLang attention's log-sum-exp",
            ),
        ):
            value = getattr(kernels_config, field_name, None)
            if value != "tilelang":
                raise ValueError(
                    f"dsa_indexer_loss requires {field_name}='tilelang', got {value!r}: {reason}. "
                    "Set it under model.ops_implementation, or switch the objective off with "
                    "dsa_indexer_loss: false / dsa_indexer_loss_coef: 0.0 under model.model_config."
                )


__all__ = ["DeepseekV4Config"]
