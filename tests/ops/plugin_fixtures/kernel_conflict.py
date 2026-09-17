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

"""Invalid plugin: kernel name collides with an already-registered kernel.

Loaded together with ``good`` (which registers kernel ``testkit`` for
``rms_norm/standard`` first), this must be rejected atomically.
"""

VEOMNI_PLUGIN_API_VERSION = 1

VEOMNI_OPS_BACKENDS = {
    "ops": {},
    "kernels": [
        {
            "name": "testkit",
            "op_name": "rms_norm",
            "variant": "standard",
            "factory": "plugin_fixtures.kernels:fake_rms_norm",
            "device_type": "any",
        },
    ],
}
