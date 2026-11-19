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

"""Vendored Kimi-K3 source modeling for VeOmni patchgen.

`configuration_kimi_k3.py` / `modeling_kimi_k3.py` are merged from the
HuggingFace reference repository `moonshotai/Kimi-K3`; `kda_fallback.py`
holds eager stand-ins for the fla linear-attention kernels used on NPU.
Never import this package directly at runtime - the patchgen-generated
files under `../generated/` are the runtime entry points.
"""
