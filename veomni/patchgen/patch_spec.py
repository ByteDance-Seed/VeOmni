# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Patch Specification DSL for Modeling Code Generation.

This module provides decorators and utilities to define patches that will be
applied during code generation. The patch definitions are declarative and
are resolved at codegen time, not at runtime.

Supported patch types:
1. Class replacement: Replace an entire class with another
2. Method override: Replace a specific method of a class
3. Function replacement: Replace a module-level function
4. Additional imports: Add new imports to the generated file

Example usage:

    # veomni/models/transformers/qwen3/patches/qwen3_gpu_patches.py
    from veomni.patchgen.patch_spec import PatchConfig, replace_class, override_method, replace_function

    config = PatchConfig(
        source_module="transformers.models.qwen3.modeling_qwen3",
        target_file="patched_modeling_qwen3_gpu.py",
    )

    @config.replace_class("Qwen3RMSNorm")
    class LigerRMSNorm(nn.Module):
        # ... implementation from liger_kernel
        pass

    @config.override_method("Qwen3Attention.forward")
    def optimized_attention_forward(self, hidden_states, ...):
        # ... optimized implementation
        pass

    @config.replace_function("apply_rotary_pos_emb")
    def liger_rotary_pos_emb(q, k, cos, sin, ...):
        # ... optimized implementation
        pass
"""

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class PatchType(Enum):
    """Types of patches that can be applied."""

    CLASS_REPLACEMENT = "class_replacement"
    METHOD_OVERRIDE = "method_override"
    FUNCTION_REPLACEMENT = "function_replacement"
    ADDITIONAL_IMPORT = "additional_import"
    INIT_MODIFICATION = "init_modification"


@dataclass
class Patch:
    """Represents a single patch to be applied during code generation."""

    patch_type: PatchType
    target: str  # e.g., "Qwen3RMSNorm" or "Qwen3Attention.forward"
    replacement: Any  # The replacement class/function/method
    source_module: Optional[str] = None  # Where the replacement comes from
    description: Optional[str] = None  # Human-readable description

    # For method overrides, we may need to specify where to get the replacement
    replacement_source: Optional[str] = None  # e.g., "liger_kernel.transformers.rms_norm"


@dataclass
class ImportSpec:
    """Specification for an import to add to the generated file."""

    module: str  # e.g., "torch.nn"
    names: Optional[list[str]] = None  # e.g., ["Module", "Linear"] for "from x import y"
    alias: Optional[str] = None  # e.g., "np" for "import numpy as np"
    is_from_import: bool = True  # True for "from x import y", False for "import x"


@dataclass
class PatchConfig:
    """
    Configuration for a set of patches to be applied to a source module.

    This is the main entry point for defining patches. Create a PatchConfig
    and use its decorators to register patches.
    """

    source_module: str  # e.g., "transformers.models.qwen3.modeling_qwen3"
    target_file: str = "patched_modeling.py"  # Output filename
    description: str = ""  # Description of this patch set

    # Collected patches
    patches: list[Patch] = field(default_factory=list)
    additional_imports: list[ImportSpec] = field(default_factory=list)

    # Classes/functions to exclude from the output
    exclude: list[str] = field(default_factory=list)

    # Optional: specify HF transformers version this is based on
    transformers_version: Optional[str] = None

    def replace_class(self, target_class: str, description: Optional[str] = None):
        """
        Decorator to register a class replacement.

        Usage:
            @config.replace_class("Qwen3RMSNorm")
            class LigerRMSNorm(nn.Module):
                ...
        """

        def decorator(cls: type) -> type:
            patch = Patch(
                patch_type=PatchType.CLASS_REPLACEMENT,
                target=target_class,
                replacement=cls,
                source_module=cls.__module__ if hasattr(cls, "__module__") else None,
                description=description or f"Replace {target_class} with {cls.__name__}",
            )
            self.patches.append(patch)
            return cls

        return decorator

    def override_method(self, target_method: str, description: Optional[str] = None):
        """
        Decorator to register a method override.

        Usage:
            @config.override_method("Qwen3Attention.forward")
            def optimized_forward(self, hidden_states, ...):
                ...
        """

        def decorator(func: Callable) -> Callable:
            patch = Patch(
                patch_type=PatchType.METHOD_OVERRIDE,
                target=target_method,
                replacement=func,
                source_module=func.__module__ if hasattr(func, "__module__") else None,
                description=description or f"Override {target_method}",
            )
            self.patches.append(patch)
            return func

        return decorator

    def replace_function(self, target_func: str, description: Optional[str] = None):
        """
        Decorator to register a function replacement.

        Usage:
            @config.replace_function("apply_rotary_pos_emb")
            def liger_rotary_pos_emb(q, k, cos, sin, ...):
                ...
        """

        def decorator(func: Callable) -> Callable:
            patch = Patch(
                patch_type=PatchType.FUNCTION_REPLACEMENT,
                target=target_func,
                replacement=func,
                source_module=func.__module__ if hasattr(func, "__module__") else None,
                description=description or f"Replace {target_func} with {func.__name__}",
            )
            self.patches.append(patch)
            return func

        return decorator

    def modify_init(self, target_class: str, description: Optional[str] = None):
        """
        Decorator to register an __init__ modification.

        The decorated function should take (original_init, self, *args, **kwargs)
        and call original_init as needed.

        Usage:
            @config.modify_init("Qwen3Attention")
            def modified_init(original_init, self, config, layer_idx):
                original_init(self, config, layer_idx)
                self.custom_attr = some_value
        """

        def decorator(func: Callable) -> Callable:
            patch = Patch(
                patch_type=PatchType.INIT_MODIFICATION,
                target=target_class,
                replacement=func,
                description=description or f"Modify {target_class}.__init__",
            )
            self.patches.append(patch)
            return func

        return decorator

    def add_import(
        self, module: str, names: Optional[list[str]] = None, alias: Optional[str] = None, is_from_import: bool = True
    ):
        """
        Add an import to the generated file.

        Usage:
            config.add_import("torch.nn", names=["Module", "Linear"])
            config.add_import("numpy", alias="np", is_from_import=False)
        """
        import_spec = ImportSpec(
            module=module,
            names=names,
            alias=alias,
            is_from_import=is_from_import,
        )
        self.additional_imports.append(import_spec)

    def exclude_from_output(self, *names: str):
        """
        Exclude specific classes/functions from the generated output.

        Usage:
            config.exclude_from_output("Qwen3ForTokenClassification")
        """
        self.exclude.extend(names)

    def get_patches_for_target(self, target: str) -> list[Patch]:
        """Get all patches that apply to a specific target."""
        return [p for p in self.patches if p.target == target or p.target.startswith(f"{target}.")]

    def get_class_replacements(self) -> dict[str, Patch]:
        """Get all class replacement patches as a dict."""
        return {p.target: p for p in self.patches if p.patch_type == PatchType.CLASS_REPLACEMENT}

    def get_method_overrides(self) -> dict[str, Patch]:
        """Get all method override patches as a dict (keyed by 'ClassName.method_name')."""
        return {p.target: p for p in self.patches if p.patch_type == PatchType.METHOD_OVERRIDE}

    def get_function_replacements(self) -> dict[str, Patch]:
        """Get all function replacement patches as a dict."""
        return {p.target: p for p in self.patches if p.patch_type == PatchType.FUNCTION_REPLACEMENT}


def get_source_code(obj: Any) -> str:
    """
    Get the source code of a class or function.

    This works with both regular Python objects and dynamically defined ones.
    """
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        # Object may be dynamically created or from a C extension
        return ""


def create_patch_from_external(
    target: str,
    replacement_module: str,
    replacement_name: str,
    patch_type: PatchType = PatchType.CLASS_REPLACEMENT,
    description: Optional[str] = None,
) -> Patch:
    """
    Create a patch that references an external module.

    This is useful when you want to use a class/function from another library
    without importing it at patch definition time.

    Usage:
        patch = create_patch_from_external(
            target="Qwen3RMSNorm",
            replacement_module="liger_kernel.transformers.rms_norm",
            replacement_name="LigerRMSNorm",
        )
    """
    return Patch(
        patch_type=patch_type,
        target=target,
        replacement=None,  # Will be resolved at codegen time
        source_module=replacement_module,
        replacement_source=f"{replacement_module}.{replacement_name}",
        description=description or f"Replace {target} with {replacement_name} from {replacement_module}",
    )
