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

"""Centralize compile/capture schedule selection for standard, EP and LoRA experts.

All variants share this policy so their eager-versus-fallback rules stay aligned
without querying device tensor values.
"""

import torch

from .....utils.device import get_torch_device


def must_use_shape_derived_bound() -> bool:
    """Retain the shape schedule while compiling or capture state is unknown."""
    if torch.compiler.is_compiling():
        return True
    capture_query = getattr(get_torch_device(), "is_current_stream_capturing", None)
    if not callable(capture_query):
        return True
    try:
        return capture_query() is not False
    except (NotImplementedError, RuntimeError, TypeError):
        return True
