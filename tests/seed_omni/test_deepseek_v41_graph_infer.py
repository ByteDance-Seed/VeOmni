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

"""The order the inference FSM actually runs the DeepSeek-V4.1 nodes in.

A generation graph is the one part of a module wiring that has no type checker
and no import to fail: a body in the wrong order yields a running FSM whose
nodes read fields nobody wrote yet. The n-gram node would hash the *previous*
token, the backbone would pack embeddings the text encoder had not written, and
neither raises anything a reader would connect to the YAML.

So the order is asserted against the real runtime rather than re-derived here.
``GenerationGraph.iter_nodes`` is the same generator inference drives, and its
feed-forward fan-in rule -- an edge into a node counts only when it appears
before that node's first use as a source -- is exactly the subtle part. These
tests would have caught the body orderings I tried before reading it.
"""

from pathlib import Path

import pytest
import yaml

from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph


GRAPH_PATH = Path("configs/seed_omni/DeepSeek/deepseek_v41_flash/infer/graph_infer.yaml")

# A bare module endpoint in a generation graph defaults to `.generate`, and the
# node pool keys on the resolved endpoint, so that is what `iter_nodes` yields.
VISION = "deepseek_v41_vision.generate"
NGRAM = "deepseek_v41_ngram.generate"
TEXT_ENCODER = "deepseek_v41_text_encoder.generate"
LLM = "deepseek_v41_llm.generate"


@pytest.fixture
def graph() -> GenerationGraph:
    return GenerationGraph(yaml.safe_load(GRAPH_PATH.read_text()))


def order(graph: GenerationGraph) -> list:
    """Node names in the order one FSM step would execute them."""
    return [node.name for node in graph.iter_nodes({})]


class TestBodyOrder:
    def test_the_prompt_is_encoded_hashed_then_prefilled(self, graph):
        assert graph.current_state_name == "prompt_encode"
        assert order(graph) == [VISION, NGRAM, TEXT_ENCODER, LLM]

    def test_the_text_encoder_runs_once_while_prefilling(self, graph):
        """It opens the body and closes it, but the closing edge is feedback.

        Running it twice here would sample a token off the prefill hidden states
        inside ``prompt_encode``, and ``text_ar`` would then sample a second one
        from the same position.
        """
        assert order(graph).count(TEXT_ENCODER) == 1

    def test_a_decode_step_samples_hashes_then_steps(self, graph):
        graph._transition_to("text_ar", {})
        assert order(graph) == [TEXT_ENCODER, NGRAM, LLM]

    def test_the_sampler_precedes_the_hash(self, graph):
        """The load-bearing order, and the one with no symptom if reversed.

        The n-gram node reads the sampled id from ``meta['input_ids']`` on the
        carrier tail. Run it before the text encoder and that key still holds the
        *previous* step's id: every n-gram would be shifted one token late, the
        lookups would be plausible, and nothing would raise.
        """
        graph._transition_to("text_ar", {})
        names = order(graph)
        assert names.index(TEXT_ENCODER) < names.index(NGRAM) < names.index(LLM)

    def test_vision_does_not_run_per_token(self, graph):
        graph._transition_to("text_ar", {})
        assert VISION not in order(graph)


class TestGraphShape:
    def test_done_is_not_declared(self):
        """The framework injects it; declaring it overrides terminal semantics."""
        assert "done" not in yaml.safe_load(GRAPH_PATH.read_text())["states"]

    def test_both_states_can_finish(self, graph):
        spec = yaml.safe_load(GRAPH_PATH.read_text())
        for name, state in spec["states"].items():
            keys = [t["condition"].get("key") for t in state["transitions"]]
            assert "text_done" in keys, f"state '{name}' has no way to reach done"

    def test_the_default_transition_is_last(self):
        """Anything after a `default` is unreachable."""
        spec = yaml.safe_load(GRAPH_PATH.read_text())
        for name, state in spec["states"].items():
            types = [t["condition"]["type"] for t in state["transitions"]]
            if "default" in types:
                assert types[-1] == "default", f"state '{name}' has transitions after its default"

    def test_every_module_in_the_graph_is_a_real_module(self, graph):
        from veomni.models.seed_omni.modules import OMNI_MODEL_REGISTRY

        graph._transition_to("prompt_encode", {})
        names = set(order(graph))
        graph._transition_to("text_ar", {})
        names |= set(order(graph))
        assert names == {VISION, NGRAM, TEXT_ENCODER, LLM}
        for name in names:
            module = name.split(".")[0]
            assert module in OMNI_MODEL_REGISTRY, f"'{module}' is not a registered OmniModule"
