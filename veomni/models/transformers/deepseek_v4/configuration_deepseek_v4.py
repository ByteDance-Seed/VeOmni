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

from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as _DeepseekV4Config,
)


class DeepseekV4Config(_DeepseekV4Config):
    """DeepSeek-V4, plus the two fields of the Lightning Indexer KL objective.

    The objective is a training objective, not a kernel backend, so it is
    configured the way this model's other auxiliary objective already is:
    ``output_router_logits`` / ``router_aux_loss_coef`` are fields of the model
    config, folded into the loss in ``DeepseekV4ForCausalLM.forward`` from
    ``self.config``, and ``dsa_indexer_loss`` / ``dsa_indexer_loss_coef`` sit
    beside them and are read the same way. The neighbouring
    ``dsa_indexer_implementation`` / ``dsa_attention_implementation`` stay on
    ``OpsImplementationConfig``, which is kernel selection and nothing else.

    Being *declared* here is load-bearing rather than tidiness. Overrides from
    ``model.model_config`` reach the config as ``**kwargs`` to
    ``PreTrainedConfig.from_dict``, which applies only those keys the
    constructed config already answers ``hasattr`` for and drops the rest
    silently -- no error, no warning. An undeclared ``dsa_indexer_loss: true``
    would therefore parse, launch, train the language-model objective alone and
    report no indexer metric, which is the exact silent no-op every other gate
    in this feature exists to refuse. Declaring the two fields is what makes the
    YAML reach the model at all.

    Deliberately not validated here. ``from_dict`` runs ``__post_init__`` on the
    on-disk values and only then ``setattr``s the overrides, so a bound checked
    in a ``__post_init__`` would see ``config.json`` and never the YAML that
    contradicts it. ``check_indexer_loss_prerequisites`` in ``veomni/models/auto.py``
    is the one place that sees the config after its overrides and the installed
    ``OpsImplementationConfig`` at the same time, and it runs before any rank
    reads a weight.
    """

    dsa_indexer_loss: bool = False
    dsa_indexer_loss_coef: float = 1.0


__all__ = ["DeepseekV4Config"]
