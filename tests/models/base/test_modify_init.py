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

"""``modify_init`` extras stay at the function-body indent."""

from __future__ import annotations

import ast
import textwrap

from veomni.patchgen.codegen import ModelingCodeGenerator
from veomni.patchgen.patch_spec import Patch, PatchConfig, PatchType


def _bind_handle(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_attn = object()


def test_modify_init_appends_at_function_body_indent_not_trailing_if():
    source = textwrap.dedent(
        """
        class Foo:
            def __init__(self, enabled=True):
                self.flag = False
                if enabled:
                    self.flag = True
        """
    ).lstrip()
    generator = ModelingCodeGenerator(PatchConfig(source_module="mod", target_file="out.py"))
    generator.source_code = source
    generator.source_lines = source.splitlines()
    class_node = ast.parse(source).body[0]
    patch = Patch(
        patch_type=PatchType.INIT_MODIFICATION,
        target="Foo",
        replacement=_bind_handle,
        description="Bind handle",
    )
    composed = generator._compose_modified_init(class_node, patch)
    handle_lines = [line for line in composed.splitlines() if "veomni_attn" in line]
    assert handle_lines
    assert handle_lines[0].startswith("    self.veomni_attn")
    assert not handle_lines[0].startswith("        self.veomni_attn")
