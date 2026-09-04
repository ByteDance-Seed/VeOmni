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
    """DeepSeek-V4, plus the two fields of the Lightning Indexer KL objective.

    The objective is a training objective, not a kernel backend, so it is
    configured the way this model's other auxiliary objective already is:
    ``output_router_logits`` / ``router_aux_loss_coef`` are fields of the model
    config, folded into the loss in ``DeepseekV4ForCausalLM.forward`` from
    ``self.config``, and ``dsa_indexer_loss`` / ``dsa_indexer_loss_coef`` sit
    beside them and are read the same way. The neighbouring
    ``dsa_indexer_implementation`` / ``dsa_attention_implementation`` stay on
    the kernels config, which is kernel selection and nothing else.

    Being *declared* here is load-bearing rather than tidiness. Overrides from
    ``model.model_config`` reach the config as ``**kwargs`` to
    ``PreTrainedConfig.from_dict``, which applies only those keys the
    constructed config already answers ``hasattr`` for and drops the rest
    silently -- no error, no warning. An undeclared ``dsa_indexer_loss: true``
    would therefore parse, launch, train the language-model objective alone and
    report no indexer metric, which is the exact silent no-op every other gate
    in this feature exists to refuse. Declaring the two fields is what makes the
    YAML reach the model at all.

    Validated in ``validate_build_prerequisites`` below rather than in a
    ``__post_init__``: ``from_dict`` runs ``__post_init__`` on the on-disk values
    and only then ``setattr``s the overrides, so a bound checked there would see
    ``config.json`` and never the YAML that contradicts it.
    """

    dsa_indexer_loss: bool = False
    dsa_indexer_loss_coef: float = 1.0

    def validate_build_prerequisites(self) -> None:
        """Refuse a Lightning Indexer KL objective this run cannot actually train.

        ``build_foundation_model`` calls this on any config that defines it, once the
        config is finished and before any rank reads a weight -- 54.8 GB for
        DeepSeek-V4-Flash, which is the cost of learning from the first forward
        instead that three lines of YAML disagree with each other.

        It lives on the config rather than in the generic builder because
        everything it knows is DeepSeek-V4's: two of its own fields, and which kernels
        those fields require. A generic model builder holding a list of model names is
        a list that the next model has to be remembered into. Here there is nothing to
        remember -- ``model.model_config`` is DeepSeek-V4's config precisely when the
        model is DeepSeek-V4.

        What it needs from outside is the installed kernels config: the objective's
        student distribution is the TileLang indexer's per-slot scores and its teacher
        is the TileLang sparse attention's log-sum-exp, and those two are kernel
        selections rather than model fields. Read off the installed singleton, not
        passed in, so that the hook stays a plain no-argument call that any config can
        implement.

        ``DeepseekV4Attention.forward`` refuses the same two on its first forward and
        that gate stays: it covers the paths that never come through
        ``build_foundation_model``, such as a test building a model straight from
        ``_from_config``.
        """
        from veomni.kernels.config import get_kernels_config

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

        kernels_config = get_kernels_config()
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
