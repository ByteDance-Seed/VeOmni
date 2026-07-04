# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""DeepSpec ↔ VeOmni bridge.

DeepSpec (https://github.com/deepseek-ai) trains small *draft* models for
speculative decoding. Its training is fully *decoupled* from the target model:
a target model is run offline once to dump per-token hidden states to disk (the
"target cache"), and training then only reads that cache — the target model is
never in the training loop (it is only used at init to copy frozen embeddings /
lm_head weights).

That decoupling is what makes the integration clean: from VeOmni's point of view
a draft model is just a small ``transformers.PreTrainedModel`` trained on a
tensor dataset. This package wires the two together *without duplicating the
DeepSpec algorithm* — it imports DeepSpec's own modeling / loss / dataset code
and adapts the thin interfaces VeOmni expects:

* ``model``   — DeepSpec draft models are registered in VeOmni's
  ``MODELING_REGISTRY`` / ``MODEL_CONFIG_REGISTRY`` (see ``modeling.py``).
* ``data``    — DeepSpec's ``CacheDataset`` is exposed as a VeOmni mapping
  dataset and its ``CacheCollator`` is reused verbatim (see
  ``veomni/data/deepspec``).
* ``trainer`` — a small ``DraftModelTrainer`` subclasses VeOmni's ``BaseTrainer``
  and overrides model build (frozen embeds), data build, and the
  forward/loss/loss-reduction path so DeepSpec's own loss functions drive the
  backward pass (see ``veomni/trainer/deepspec``).

Set-up: the DeepSpec repo is not a pip package, so it must be importable. Call
``ensure_deepspec_importable()`` (done automatically by the registration and
trainer modules) which honours the ``DEEPSPEC_PATH`` env var and falls back to a
sibling ``DeepSpec/`` checkout next to the VeOmni repo.
"""

from .deepspec_path import DEEPSPEC_ENV_VAR, ensure_deepspec_importable


__all__ = [
    "DEEPSPEC_ENV_VAR",
    "ensure_deepspec_importable",
]
