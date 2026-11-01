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

"""In-tree convert for the ``fake_module_a → fake_module_b`` stand-in chain.

Upstream source is a HuggingFace ``config.json`` with ``model_type: fake_omni``
(no real weights to split). Graphs come from
``configs/seed_omni/fake_model/`` (``graph_train.yaml`` / ``graph_infer.yaml``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

from ...utils.convert_registry import OMNI_CONVERT_REGISTRY, load_family_graphs


FAKE_A = "fake_module_a"
FAKE_B = "fake_module_b"


def load_fake_omni_graphs(
    *,
    training_graph: str | Path | None = None,
    generation_graph: str | Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Read the training DAGs and generation FSMs from YAML under ``configs/seed_omni/fake_model``."""
    return load_family_graphs(
        "configs/seed_omni/fake_model",
        training="graph_train.yaml",
        generation="graph_infer.yaml",
        training_graph=training_graph,
        generation_graph=generation_graph,
    )


@OMNI_CONVERT_REGISTRY.register("fake_omni")
def register_fake_omni_convert():
    return convert_fake_omni


def convert_fake_omni(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Build the two-module stand-in chain; :func:`convert_checkpoint` writes the split directory."""
    training_graph = kwargs.pop("training_graph", None)
    generation_graph = kwargs.pop("generation_graph", None)
    del kwargs
    from .fake_module_a.configuration import FakeModuleAConfig
    from .fake_module_a.modeling import FakeModuleA
    from .fake_module_b.configuration import FakeModuleBConfig
    from .fake_module_b.modeling import FakeModuleB

    cfg_dict, _ = PretrainedConfig.get_config_dict(model_path)
    hidden_size = int(cfg_dict.get("hidden_size", 8))
    training_graphs, generation_graphs = load_fake_omni_graphs(
        training_graph=training_graph,
        generation_graph=generation_graph,
    )
    return {
        "modules": {
            FAKE_A: FakeModuleA(FakeModuleAConfig(hidden_size=hidden_size)),
            FAKE_B: FakeModuleB(FakeModuleBConfig(hidden_size=hidden_size)),
        },
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "train_type": "default",
        "infer_type": "infer_gen",
    }
