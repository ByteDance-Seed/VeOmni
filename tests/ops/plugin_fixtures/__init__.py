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

"""Fake ops-backend plugin declarations for the loader tests.

Importable as ``plugin_fixtures.*`` (``tests/ops`` is put on ``sys.path`` by
``test_ops_plugin_loader.py``). Each module mimics one plugin scenario.
"""

from .kernels import FakeRMSNorm, fake_rms_norm


__all__ = ["FakeRMSNorm", "fake_rms_norm"]
