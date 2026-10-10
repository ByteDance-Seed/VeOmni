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

"""Which modules ``ParallelPlan`` hands the parallelizer to wrap on each para mesh."""

import pytest
import torch.nn as nn
from torch.distributed.tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.models.transformers.qwen3_5_moe.parallel_plan import get_parallel_plan as qwen3_5_moe_plan


def test_every_planned_parameter_owner_is_a_para_module():
    """An owner left out would be re-sharded over the regular FSDP mesh, mixing ranks' slices."""
    assert qwen3_5_moe_plan().extra_parallel_fsdp_no_shard_module == {
        "ep": {"model.language_model.layers.*.mlp.experts", "mtp.layers.*.mlp.experts"}
    }


def test_glm_moe_dsa_plan_owns_only_the_routed_experts_of_sparse_layers():
    """The leading ``first_k_dense_replace`` layers have a dense MLP and no experts to shard."""
    from pathlib import Path

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models.auto import build_foundation_model
    from veomni.ops import apply_ops_config
    from veomni.ops.config.singleton import get_ops_config, set_ops_config

    toy_config = Path(__file__).resolve().parents[1] / "toy_config" / "glm_moe_dsa_toy"
    previous = get_ops_config()
    apply_ops_config(make_eager_ops_config())
    try:
        model = build_foundation_model(config_path=str(toy_config), weights_path=None, init_device="meta")
    finally:
        set_ops_config(previous)

    owners = model.get_parallel_plan().get_extra_parallel_fsdp_no_shard_info(model, "ep")
    sparse_layers = range(model.config.first_k_dense_replace, model.config.num_hidden_layers)
    assert sorted(owners) == [f"model.layers.{i}.mlp.experts" for i in sparse_layers]


def test_owners_of_one_module_collapse_to_one_entry():
    plan = ParallelPlan(
        extra_parallel_plan={"ep": {"layers.*.experts.gate_up_proj": Shard(0), "layers.*.experts.down_proj": Shard(0)}}
    )
    assert plan.extra_parallel_fsdp_no_shard_module == {"ep": {"layers.*.experts"}}


def test_every_owner_is_found_in_the_model():
    model = nn.Module()
    model.embed_tokens = nn.Embedding(8, 4)
    model.decoder = nn.Module()
    model.decoder.ngram = nn.Embedding(8, 4)
    model.decoder.ngram_extra = nn.Embedding(8, 4)
    plan = ParallelPlan(
        extra_parallel_plan={"emb": {"embed_tokens.weight": Shard(0), "decoder.ngram.weight": Shard(0)}}
    )

    found = plan.get_extra_parallel_fsdp_no_shard_info(model, "emb")
    assert found == {"embed_tokens": model.embed_tokens, "decoder.ngram": model.decoder.ngram}


def test_nested_owners_are_rejected():
    with pytest.raises(ValueError, match="'decoder' and 'decoder.ngram' are nested"):
        ParallelPlan(extra_parallel_plan={"emb": {"decoder.weight": Shard(0), "decoder.ngram.weight": Shard(0)}})


def test_owners_that_only_share_a_name_prefix_are_not_nested():
    plan = ParallelPlan(extra_parallel_plan={"emb": {"ngram.weight": Shard(0), "ngram_extra.weight": Shard(0)}})
    assert plan.extra_parallel_fsdp_no_shard_module == {"emb": {"ngram", "ngram_extra"}}
