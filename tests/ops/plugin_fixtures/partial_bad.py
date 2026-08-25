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

"""Invalid plugin: valid ``ops`` section but broken ``kernels`` entry.

Guards atomicity — nothing from this plugin may be registered, not even the
valid part.
"""

VEOMNI_PLUGIN_API_VERSION = 1

VEOMNI_OPS_BACKENDS = {
    "ops": {
        "rms_norm": {
            "partialkit": {"entry": "plugin_fixtures.kernels:FakeRMSNorm"},
        },
    },
    "kernels": [
        {
            "name": "partialkit",
            "op_name": "rms_norm",
            "variant": "standard",
            # "factory" deliberately missing -> invalid
            "device_type": "any",
        },
    ],
}
