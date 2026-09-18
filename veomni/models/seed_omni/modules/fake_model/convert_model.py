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

from ...configuration_omni import OmniConfig
from ...utils.convert_registry import OMNI_CONVERT_REGISTRY


FAKE_A = "fake_module_a"
FAKE_B = "fake_module_b"

_CONFIG_RELATIVE = Path("configs") / "seed_omni" / "fake_model"
_TRAIN_GRAPH_NAME = "graph_train.yaml"
_INFER_GRAPH_NAME = "graph_infer.yaml"


def _default_config_dir() -> Path:
    """Walk up from this file to the repo's ``configs/seed_omni/fake_model``."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / _CONFIG_RELATIVE
        if (candidate / _TRAIN_GRAPH_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {_CONFIG_RELATIVE / _TRAIN_GRAPH_NAME} above {__file__}. "
        "Pass train_graph= / generation_graph= explicitly, or run convert from a VeOmni checkout."
    )


def load_family_graphs(
    *,
    train_graph: str | Path | None = None,
    generation_graph: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read the training DAG and generation FSMs from YAML under ``configs/``."""
    config_dir: Path | None = None

    def configs() -> Path:
        nonlocal config_dir
        if config_dir is None:
            config_dir = _default_config_dir()
        return config_dir

    train_path = Path(train_graph) if train_graph is not None else configs() / _TRAIN_GRAPH_NAME
    generation_path = Path(generation_graph) if generation_graph is not None else configs() / _INFER_GRAPH_NAME
    training = OmniConfig._read_graph_file(str(train_path), "training_graph")
    generation = OmniConfig._read_generation_graphs(str(generation_path))
    return training, generation


@OMNI_CONVERT_REGISTRY.register("fake_omni")
def register_fake_omni_convert():
    return convert_fake_omni


def convert_fake_omni(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Build the two-module stand-in chain; caller writes via ``save_converted_omni``."""
    train_graph = kwargs.pop("train_graph", None)
    generation_graph = kwargs.pop("generation_graph", None)
    del kwargs
    from .fake_module_a.configuration import FakeModuleAConfig
    from .fake_module_a.modeling import FakeModuleA
    from .fake_module_b.configuration import FakeModuleBConfig
    from .fake_module_b.modeling import FakeModuleB

    cfg_dict, _ = PretrainedConfig.get_config_dict(model_path)
    hidden_size = int(cfg_dict.get("hidden_size", 8))
    training_graph, generation_graphs = load_family_graphs(
        train_graph=train_graph,
        generation_graph=generation_graph,
    )
    return {
        "modules": {
            FAKE_A: FakeModuleA(FakeModuleAConfig(hidden_size=hidden_size)),
            FAKE_B: FakeModuleB(FakeModuleBConfig(hidden_size=hidden_size)),
        },
        "training_graph": training_graph,
        "generation_graphs": generation_graphs,
        "infer_type": "infer_gen",
    }
