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
# See the License for the specific language governing limitations
# under the License.

"""Every unit answers ``config_path``, so the loader never asks what it holds."""

from veomni.arguments import ModelArguments


class TestWhereTheLoaderReadsTheConfigFrom:
    def test_a_module_falls_back_to_its_weights_folder(self):
        # A module inside a composed checkpoint is addressed by its own subfolder,
        # so it never configures a config path separately.
        assert ModelArguments(model_path="somewhere").config_path == "somewhere"

    def test_a_whole_model_honours_a_separate_config_path(self):
        # Toy-config runs rely on this: architecture from a local json, weights
        # (and tokenizer) from somewhere else entirely.
        assert ModelArguments(model_path="weights", config_path="cfg").config_path == "cfg"

    def test_a_whole_model_falls_back_to_its_weights_path(self):
        assert ModelArguments(model_path="weights").config_path == "weights"

    def test_the_tokenizer_follows_the_config_the_base_settled(self):
        assert ModelArguments(model_path="weights", config_path="cfg").tokenizer_path == "cfg"
