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

"""Copied or adapted official eager math for first-party models_kernel toys.

These snapshots live under ``tests/`` so consume tests do not import
``veomni.models``. ``models/`` will be deleted after the switch.

Each file states ``Source`` at the top:

- Wan: adapted from ``veomni/models/transformers/wan/modeling_wan.py`` (Wan2.1 lineage)
- Flux: adapted from ``veomni/models/transformers/flux/modeling_flux.py`` (Black Forest Labs lineage)
- MoVQGAN: copied from ``veomni/models/transformers/movqgan/`` (no upstream URL in that snapshot)
"""
