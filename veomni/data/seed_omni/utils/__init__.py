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

"""Per-modality IO for the SeedOmni V2 data layer — one module per modality.

Each loads a source at whatever resolution it was authored in and reports what
it found, leaving every model-specific decision — resampling, ``smart_resize``,
patchify — to the module processor that owns it. That split is what keeps the
data layer model-agnostic (``docs/seed_omni/seed_omni_v2.md`` § 3).

Nothing is re-exported here, deliberately: importing one modality's module
should not execute the other two. :mod:`video` is the case that matters — it
reaches ``av`` through ``data/multimodal/audio_utils.py``, which is why
``seedomni_transform.py`` imports it behind ``is_video_audio_available()``
rather than unconditionally. A convenience re-export would run its module scope
on any import of this package and defeat that guard.
"""
