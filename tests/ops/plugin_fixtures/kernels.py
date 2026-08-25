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

"""Entry targets for the fake plugin declarations (``module:attr`` endpoints)."""


class FakeRMSNorm:
    """BackendSpec.entry target: a stand-in PER_MODEL replacement class."""


def fake_rms_norm(x, weight, eps=1e-6):  # pragma: no cover - never executed
    """KernelSpec.factory target: a stand-in functional kernel."""
    return x
