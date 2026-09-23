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

from ...module_configuration_base import OmniModuleConfig


class FakeModuleAConfig(OmniModuleConfig):
    """Minimal OmniModule config used to exercise the composite save/load path."""

    model_type = "fake_module_a"

    def __init__(self, hidden_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
