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
AST-based Code Generator for Modeling Patches.

This module provides the core code generation functionality that:
1. Loads the source HuggingFace modeling code via inspect
2. Parses it into an AST
3. Applies patches (class replacements, method overrides, function replacements)
4. Generates a self-contained output file with clear comments

The generated file will:
- Be fully self-contained (no hidden monkey patches)
- Have clear comments showing what was patched and why
- Maintain the original code structure where not patched
"""

import ast
import importlib
import inspect
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .patch_spec import Patch, PatchConfig


class CodegenError(Exception):
    """Exception raised during code generation."""

    pass


def get_module_source(module_name: str) -> str:
    """
    Get the full source code of a module.

    Args:
        module_name: Fully qualified module name (e.g., "transformers.models.qwen3.modeling_qwen3")

    Returns:
        The source code as a string.

    Raises:
        CodegenError: If the module cannot be imported or source cannot be obtained.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise CodegenError(f"Cannot import module '{module_name}': {e}") from e

    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as e:
        raise CodegenError(f"Cannot get source for module '{module_name}': {e}") from e

    return source


def get_object_source(obj: Any) -> str:
    """Get the source code of a class or function, preserving comments."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def extract_source_segment(source_lines: list[str], start_line: int, end_line: int) -> str:
    """
    Extract a segment of source code from source lines, preserving comments and blank lines.

    Also includes any leading comments and decorators that precede the definition.
    Line numbers are 1-indexed (as in AST).
    """
    # Look backwards for leading comments and blank lines
    actual_start = start_line - 1  # Convert to 0-indexed

    # Include leading comments/decorators (lines starting with # or @)
    while actual_start > 0:
        prev_line = source_lines[actual_start - 1].strip()
        if prev_line.startswith("#") or prev_line.startswith("@") or prev_line == "":
            actual_start -= 1
        else:
            break

    # Extract the segment
    segment = source_lines[actual_start:end_line]
    return "\n".join(segment)


def get_node_end_line(node: ast.AST, source_lines: list[str]) -> int:
    """
    Get the end line of an AST node.

    For nodes with end_lineno, use that. Otherwise, estimate by finding
    the next definition or end of file.
    """
    if hasattr(node, "end_lineno") and node.end_lineno is not None:
        return node.end_lineno

    # Fallback: use unparse to count lines (less accurate but works)
    try:
        unparsed = ast.unparse(node)
        return node.lineno + unparsed.count("\n")
    except Exception:
        return node.lineno + 10  # Rough estimate


class ImportCollector(ast.NodeVisitor):
    """
    Collects all import statements from an AST.
    """

    def __init__(self):
        self.imports: list[ast.stmt] = []
        self.import_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.import_names.add(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.import_names.add(name)


class ClassAndFunctionCollector(ast.NodeVisitor):
    """
    Collects all class and function definitions from an AST.
    """

    def __init__(self):
        self.classes: dict[str, ast.ClassDef] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.other_statements: list[ast.stmt] = []
        self._in_class = False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes[node.name] = node
        # Don't recurse into class body for top-level collection

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._in_class:
            self.functions[node.name] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not self._in_class:
            self.functions[node.name] = node


class MethodReplacer(ast.NodeTransformer):
    """
    AST transformer that replaces specific methods within a class.
    """

    def __init__(self, method_name: str, new_method_ast: ast.FunctionDef, comment: str = ""):
        self.method_name = method_name
        self.new_method_ast = new_method_ast
        self.comment = comment
        self.replaced = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name == self.method_name:
            self.replaced = True
            # Preserve the original method name
            self.new_method_ast.name = self.method_name
            # Add docstring comment about the patch
            if self.comment:
                # Check if there's already a docstring
                if (
                    self.new_method_ast.body
                    and isinstance(self.new_method_ast.body[0], ast.Expr)
                    and isinstance(self.new_method_ast.body[0].value, ast.Constant)
                ):
                    # Prepend patch info to existing docstring
                    existing = self.new_method_ast.body[0].value.value
                    self.new_method_ast.body[0].value.value = f"[PATCHED] {self.comment}\n\n{existing}"
            return self.new_method_ast
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        if node.name == self.method_name:
            self.replaced = True
            self.new_method_ast.name = self.method_name
            return self.new_method_ast
        return node


def parse_source_to_ast(source: str) -> ast.Module:
    """Parse Python source code to AST."""
    try:
        return ast.parse(source)
    except SyntaxError as e:
        raise CodegenError(f"Syntax error in source code: {e}") from e


def ast_to_source(node: ast.AST) -> str:
    """Convert AST node back to source code."""
    return ast.unparse(node)


def get_class_method_ast(class_node: ast.ClassDef, method_name: str) -> Optional[ast.FunctionDef]:
    """Get the AST node for a specific method in a class."""
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
            return item
    return None


def create_comment_node(comment: str) -> ast.Expr:
    """Create an AST node representing a comment (as a string expression)."""
    return ast.Expr(value=ast.Constant(value=f"# {comment}"))


class ModelingCodeGenerator:
    """
    Main code generator class that applies patches and generates output.

    This class orchestrates the entire code generation process:
    1. Load source module
    2. Parse into AST
    3. Apply patches
    4. Generate output with comments
    """

    def __init__(self, config: PatchConfig):
        self.config = config
        self.source_code: str = ""
        self.source_lines: list[str] = []
        self.source_ast: Optional[ast.Module] = None
        self.output_parts: list[str] = []

    def load_source(self) -> None:
        """Load the source module code."""
        self.source_code = get_module_source(self.config.source_module)
        self.source_lines = self.source_code.splitlines()
        self.source_ast = parse_source_to_ast(self.source_code)

    def _collect_imports(self) -> tuple[list[ast.stmt], set[str]]:
        """Collect all imports from the source AST."""
        collector = ImportCollector()
        collector.visit(self.source_ast)
        return collector.imports, collector.import_names

    def _collect_definitions(self) -> tuple[dict[str, ast.ClassDef], dict[str, ast.FunctionDef]]:
        """Collect all class and function definitions."""
        collector = ClassAndFunctionCollector()
        collector.visit(self.source_ast)
        return collector.classes, collector.functions

    def _get_transformers_version(self) -> str:
        """Get the installed transformers version."""
        try:
            import transformers

            return transformers.__version__
        except ImportError:
            return "unknown"

    def _generate_header(self) -> str:
        """Generate the file header with patch information."""
        transformers_version = self.config.transformers_version or self._get_transformers_version()

        lines = [
            "# " + "=" * 78,
            "#  AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY",
            "# " + "=" * 78,
            "#",
            f"#  Source: {self.config.source_module}",
            f"#  Based on: transformers=={transformers_version}",
            f"#  Generated: {datetime.now().isoformat()}",
            "#",
            "#  This file was generated by the modeling code generator.",
            "#  It contains a patched version of the original HuggingFace modeling code.",
            "#",
            "#  Patches applied:",
        ]

        for patch in self.config.patches:
            lines.append(f"#    - {patch.patch_type.value}: {patch.target}")
            if patch.description:
                lines.append(f"#      {patch.description}")

        lines.append("#")
        lines.append("# " + "=" * 78)
        lines.append("")

        return "\n".join(lines)

    def _transform_imports(self, import_nodes: list[ast.stmt]) -> str:
        """
        Transform and filter imports for the generated file.

        This handles:
        - Converting relative imports to absolute imports
        - Adding additional imports from the patch config
        """
        output_lines = []

        for node in import_nodes:
            if isinstance(node, ast.ImportFrom):
                # Handle relative imports - convert to absolute
                if node.level > 0:
                    # Try to resolve to absolute import
                    # The base module is self.config.source_module
                    base_parts = self.config.source_module.split(".")
                    if node.level <= len(base_parts):
                        absolute_module = ".".join(base_parts[: -node.level])
                        if node.module:
                            absolute_module = f"{absolute_module}.{node.module}"
                        new_node = ast.ImportFrom(module=absolute_module, names=node.names, level=0)
                        output_lines.append(ast_to_source(new_node))
                    else:
                        output_lines.append(ast_to_source(node))
                else:
                    output_lines.append(ast_to_source(node))
            else:
                output_lines.append(ast_to_source(node))

        # Add additional imports from config
        if self.config.additional_imports:
            output_lines.append("")
            output_lines.append("# Additional imports for patches")
            for imp in self.config.additional_imports:
                if imp.is_from_import:
                    if imp.names:
                        names_str = ", ".join(imp.names)
                        output_lines.append(f"from {imp.module} import {names_str}")
                    else:
                        output_lines.append(f"from {imp.module} import *")
                else:
                    if imp.alias:
                        output_lines.append(f"import {imp.module} as {imp.alias}")
                    else:
                        output_lines.append(f"import {imp.module}")

        return "\n".join(output_lines)

    def _apply_class_replacement(
        self,
        original_class: ast.ClassDef,
        patch: Patch,
    ) -> str:
        """
        Apply a class replacement patch.

        Returns the source code for the replaced class with comments preserved.
        """
        lines = []
        lines.append("")
        lines.append(f"# {'=' * 70}")
        lines.append(f"# [PATCHED CLASS] {original_class.name}")
        lines.append(
            f"# Original class replaced with: {patch.replacement.__name__ if patch.replacement else 'external'}"
        )
        if patch.description:
            lines.append(f"# Reason: {patch.description}")
        if patch.source_module:
            lines.append(f"# Source: {patch.source_module}")
        lines.append(f"# {'=' * 70}")

        if patch.replacement:
            # Get source of replacement class (preserves comments)
            replacement_source = get_object_source(patch.replacement)
            if replacement_source:
                # Rename the class to match original using simple text replacement
                replacement_source = textwrap.dedent(replacement_source)
                # Remove the patch decorator line if present
                source_lines = replacement_source.splitlines()
                filtered_lines = []
                for line in source_lines:
                    stripped = line.strip()
                    # Skip decorator lines that were for patch definition
                    if stripped.startswith("@") and ("replace_class" in stripped or "config." in stripped):
                        continue
                    filtered_lines.append(line)
                replacement_source = "\n".join(filtered_lines)

                # Rename the class (simple text replacement for class definition line)
                old_name = patch.replacement.__name__
                if old_name != original_class.name:
                    # Replace "class OldName" with "class NewName"
                    replacement_source = replacement_source.replace(
                        f"class {old_name}",
                        f"class {original_class.name}",
                        1,  # Only replace first occurrence
                    )
                lines.append(replacement_source)
            else:
                # Fallback: generate a placeholder
                lines.append(f"# Could not get source for {patch.replacement.__name__}")
                lines.append("# Using original class as fallback")
                end_line = get_node_end_line(original_class, self.source_lines)
                lines.append(extract_source_segment(self.source_lines, original_class.lineno, end_line))
        elif patch.replacement_source:
            # External replacement - generate import and alias
            lines.append(f"# Import from: {patch.replacement_source}")
            module, name = patch.replacement_source.rsplit(".", 1)
            lines.append(f"from {module} import {name} as {original_class.name}")
        else:
            end_line = get_node_end_line(original_class, self.source_lines)
            lines.append(extract_source_segment(self.source_lines, original_class.lineno, end_line))

        return "\n".join(lines)

    def _apply_method_override(
        self,
        class_node: ast.ClassDef,
        method_name: str,
        patch: Patch,
    ) -> tuple[ast.ClassDef, str, bool]:
        """
        Apply a method override to a class.

        Returns a tuple of (modified class AST, replacement method source with comments).
        The replacement source is preserved separately to maintain comments.
        """
        if not patch.replacement:
            return class_node, "", False

        # Get source of replacement method (preserves comments)
        replacement_source = get_object_source(patch.replacement)
        if not replacement_source:
            return class_node, "", False

        # Dedent and clean up the replacement source
        replacement_source = textwrap.dedent(replacement_source)

        # Remove the patch decorator lines if present (handles multi-line decorators)
        source_lines = replacement_source.splitlines()
        filtered_lines = []
        in_patch_decorator = False
        paren_depth = 0
        for line in source_lines:
            stripped = line.strip()
            # Start of a patch decorator
            if stripped.startswith("@") and ("override_method" in stripped or "config." in stripped):
                in_patch_decorator = True
                paren_depth = stripped.count("(") - stripped.count(")")
                # If decorator is closed on same line, we're done skipping
                if paren_depth <= 0:
                    in_patch_decorator = False
                continue
            # Continuation of multi-line decorator
            if in_patch_decorator:
                paren_depth += stripped.count("(") - stripped.count(")")
                if paren_depth <= 0:
                    in_patch_decorator = False
                continue
            filtered_lines.append(line)
        cleaned_replacement_source = "\n".join(filtered_lines)

        # Rename the function to match the target method name
        old_name = patch.replacement.__name__
        if old_name != method_name:
            cleaned_replacement_source = cleaned_replacement_source.replace(
                f"def {old_name}(",
                f"def {method_name}(",
                1,  # Only replace first occurrence
            )

        # Parse the replacement method for AST manipulation
        try:
            replacement_ast = parse_source_to_ast(replacement_source)
        except CodegenError:
            return class_node, "", False

        # Find the function definition in the parsed AST
        new_method = None
        for node in ast.walk(replacement_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_method = node
                # Remove decorators (they were for patch definition, not runtime)
                new_method.decorator_list = []
                break

        if not new_method:
            return class_node, "", False

        # Use the MethodReplacer transformer
        replacer = MethodReplacer(
            method_name, new_method, comment=patch.description or f"Replaced with {patch.replacement.__name__}"
        )
        modified_class = replacer.visit(class_node)
        applied = replacer.replaced

        # If the target method does not exist in the source class (e.g., class body is `pass`),
        # inject the method into the class so override_method still works for newer HF refactors.
        if not replacer.replaced:
            new_method.name = method_name
            if len(modified_class.body) == 1 and isinstance(modified_class.body[0], ast.Pass):
                modified_class.body = [new_method]
            else:
                modified_class.body.append(new_method)
            modified_class = ast.fix_missing_locations(modified_class)
            applied = True

        return modified_class, cleaned_replacement_source, applied

    def _apply_function_replacement(
        self,
        original_func: ast.FunctionDef,
        patch: Patch,
    ) -> str:
        """
        Apply a function replacement patch.

        Returns the source code for the replaced function with comments preserved.
        """
        lines = []
        lines.append("")
        lines.append(f"# {'=' * 70}")
        lines.append(f"# [PATCHED FUNCTION] {original_func.name}")
        if patch.description:
            lines.append(f"# Reason: {patch.description}")
        if patch.source_module:
            lines.append(f"# Source: {patch.source_module}")
        lines.append(f"# {'=' * 70}")

        if patch.replacement:
            replacement_source = get_object_source(patch.replacement)
            if replacement_source:
                replacement_source = textwrap.dedent(replacement_source)

                # Remove the patch decorator line if present
                source_lines = replacement_source.splitlines()
                filtered_lines = []
                for line in source_lines:
                    stripped = line.strip()
                    # Skip decorator lines that were for patch definition
                    if stripped.startswith("@") and ("replace_function" in stripped or "config." in stripped):
                        continue
                    filtered_lines.append(line)
                replacement_source = "\n".join(filtered_lines)

                # Rename the function if necessary (simple text replacement)
                old_name = patch.replacement.__name__
                if old_name != original_func.name:
                    # Replace "def old_name" with "def new_name"
                    replacement_source = replacement_source.replace(
                        f"def {old_name}",
                        f"def {original_func.name}",
                        1,  # Only replace first occurrence
                    )
                lines.append(replacement_source)
            else:
                lines.append("# Could not get source for replacement")
                end_line = get_node_end_line(original_func, self.source_lines)
                lines.append(extract_source_segment(self.source_lines, original_func.lineno, end_line))
        else:
            end_line = get_node_end_line(original_func, self.source_lines)
            lines.append(extract_source_segment(self.source_lines, original_func.lineno, end_line))

        return "\n".join(lines)

    def _indent_preserved_source(self, preserved_source: str, target_indent: int) -> list[str]:
        """
        Re-indent preserved source lines to target indentation.

        Args:
            preserved_source: Source text to re-indent
            target_indent: Number of spaces for the first non-empty line

        Returns:
            Re-indented lines
        """
        preserved_lines = preserved_source.splitlines()
        first_non_empty = next((line for line in preserved_lines if line.strip()), "")
        preserved_indent = len(first_non_empty) - len(first_non_empty.lstrip())

        indented_lines = []
        for line in preserved_lines:
            if line.strip():
                stripped = line[preserved_indent:] if len(line) >= preserved_indent else line.lstrip()
                indented_lines.append(" " * target_indent + stripped)
            else:
                indented_lines.append("")
        return indented_lines

    def _replace_method_body_with_preserved(self, class_source: str, method_name: str, preserved_source: str) -> str:
        """
        Replace a method in the class source with its comment-preserved version.

        Since ast.unparse() strips comments, we need to do text-based replacement
        to preserve the comments in the replacement method source.

        Args:
            class_source: The unparsed class source (no comments in method bodies)
            method_name: Name of the method to replace
            preserved_source: The source-preserved version of the method (with comments)

        Returns:
            The class source with the method body replaced
        """
        # Parse the class source to find the method boundaries
        try:
            class_ast = parse_source_to_ast(class_source)
        except CodegenError:
            return class_source

        # Find the class definition
        class_def = None
        for node in ast.walk(class_ast):
            if isinstance(node, ast.ClassDef):
                class_def = node
                break

        if not class_def:
            return class_source

        source_lines = class_source.splitlines()

        # Find class indentation from class definition line
        class_start = class_def.lineno - 1
        if class_start < len(source_lines):
            class_line = source_lines[class_start]
            class_indent = len(class_line) - len(class_line.lstrip())
        else:
            class_indent = 0

        # Find the method in the class
        method_node = None
        for item in class_def.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                method_node = item
                break

        if method_node:
            # Replace existing method body while preserving untouched class formatting.
            method_start = method_node.lineno - 1  # 0-indexed
            method_end = (
                method_node.end_lineno
                if hasattr(method_node, "end_lineno") and method_node.end_lineno
                else method_start + 1
            )
            if method_start < len(source_lines):
                original_line = source_lines[method_start]
                method_indent = len(original_line) - len(original_line.lstrip())
            else:
                method_indent = class_indent + 4

            indented_preserved_lines = self._indent_preserved_source(preserved_source, method_indent)
            new_source_lines = source_lines[:method_start] + indented_preserved_lines + source_lines[method_end:]
            return "\n".join(new_source_lines)

        # If the method does not exist, inject it while preserving class formatting.
        # Prefer replacing a single `pass` body when present.
        pass_node = next((item for item in class_def.body if isinstance(item, ast.Pass)), None)
        method_indent = class_indent + 4
        indented_preserved_lines = self._indent_preserved_source(preserved_source, method_indent)

        if pass_node is not None:
            pass_start = pass_node.lineno - 1
            pass_end = (
                pass_node.end_lineno if hasattr(pass_node, "end_lineno") and pass_node.end_lineno else pass_start + 1
            )
            new_source_lines = source_lines[:pass_start] + indented_preserved_lines + source_lines[pass_end:]
            return "\n".join(new_source_lines)

        # Otherwise append to the class body.
        new_source_lines = source_lines.copy()
        if new_source_lines and new_source_lines[-1].strip():
            new_source_lines.append("")
        new_source_lines.extend(indented_preserved_lines)
        return "\n".join(new_source_lines)

    def _generate_class_source(self, class_node: ast.ClassDef, patches: dict[str, Patch]) -> str:
        """Generate source code for a class, applying any relevant patches."""
        class_name = class_node.name

        # Check for class replacement
        if class_name in self.config.get_class_replacements():
            return self._apply_class_replacement(class_node, self.config.get_class_replacements()[class_name])

        # Check for method overrides
        method_overrides = self.config.get_method_overrides()
        modified_class = class_node

        methods_patched = []
        method_replacement_sources = {}
        for target, patch in method_overrides.items():
            if target.startswith(f"{class_name}."):
                method_name = target.split(".", 1)[1]
                modified_class, replacement_source, applied = self._apply_method_override(
                    modified_class, method_name, patch
                )
                if applied:
                    methods_patched.append(method_name)
                if applied and replacement_source:
                    method_replacement_sources[method_name] = replacement_source

        # Generate output
        lines = []
        if methods_patched:
            lines.append("")
            lines.append(f"# {'=' * 70}")
            lines.append(f"# [MODIFIED CLASS] {class_name}")
            lines.append(f"# Methods patched: {', '.join(methods_patched)}")
            lines.append(f"# {'=' * 70}")
            # Preserve original class formatting/comments for untouched methods,
            # and replace only the patched methods in-place.
            end_line = get_node_end_line(class_node, self.source_lines)
            class_source = extract_source_segment(self.source_lines, class_node.lineno, end_line)

            # Replace the unparsed method bodies with comment-preserved versions
            for method_name, preserved_source in method_replacement_sources.items():
                class_source = self._replace_method_body_with_preserved(class_source, method_name, preserved_source)

            lines.append(class_source)
        else:
            # No patches - use original source with comments preserved
            end_line = get_node_end_line(class_node, self.source_lines)
            lines.append(extract_source_segment(self.source_lines, class_node.lineno, end_line))

        return "\n".join(lines)

    def _generate_function_source(self, func_node: ast.FunctionDef) -> str:
        """Generate source code for a function, applying any relevant patches."""
        func_name = func_node.name

        # Check for function replacement
        func_replacements = self.config.get_function_replacements()
        if func_name in func_replacements:
            return self._apply_function_replacement(func_node, func_replacements[func_name])

        # No patches - use original source with comments preserved
        end_line = get_node_end_line(func_node, self.source_lines)
        return extract_source_segment(self.source_lines, func_node.lineno, end_line)

    def generate(self, output_path: Optional[Path] = None) -> str:
        """
        Generate the patched modeling code.

        Args:
            output_path: Optional path to write the output file.

        Returns:
            The generated source code as a string.
        """
        if not self.source_ast:
            self.load_source()

        output_parts = []

        # 1. Generate header
        output_parts.append(self._generate_header())

        # 2. Collect and transform imports
        import_nodes, import_names = self._collect_imports()
        output_parts.append(self._transform_imports(import_nodes))
        output_parts.append("")

        # 3. Process ALL module-level nodes in their original order
        # This preserves the exact structure of the original file
        for node in self.source_ast.body:
            # Skip import statements (already handled above)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue

            if isinstance(node, ast.ClassDef):
                if node.name not in self.config.exclude:
                    output_parts.append(self._generate_class_source(node, {}))
                    output_parts.append("")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in self.config.exclude:
                    output_parts.append(self._generate_function_source(node))
                    output_parts.append("")
            elif isinstance(node, ast.Assign):
                # Module-level assignments (like __all__, DOCSTRINGS, etc.)
                # Use source extraction to preserve any associated comments
                end_line = get_node_end_line(node, self.source_lines)
                output_parts.append(extract_source_segment(self.source_lines, node.lineno, end_line))
                output_parts.append("")
            elif isinstance(node, ast.Expr):
                # Module-level expressions (docstrings)
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    # This is a module docstring - use source extraction
                    end_line = get_node_end_line(node, self.source_lines)
                    output_parts.append(extract_source_segment(self.source_lines, node.lineno, end_line))
                    output_parts.append("")

        # 4. Join and format output
        output = "\n".join(output_parts)

        # 5. Write to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output)
            print(f"Generated: {output_path}")

        return output


def generate_from_config(config: PatchConfig, output_dir: Optional[Path] = None) -> str:
    """
    Convenience function to generate patched code from a config.

    Args:
        config: The patch configuration.
        output_dir: Optional directory to write output file.

    Returns:
        The generated source code.
    """
    generator = ModelingCodeGenerator(config)

    output_path = None
    if output_dir:
        output_path = output_dir / config.target_file

    return generator.generate(output_path)


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate patched modeling code from a patch configuration.")
    parser.add_argument(
        "patch_module",
        help="Python module containing the PatchConfig (e.g., 'veomni.models.transformers.qwen3.patches.qwen3_gpu_patches')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated files (default: sibling generated/ next to patch module)",
    )
    parser.add_argument(
        "-c",
        "--config-name",
        default="config",
        help="Name of the PatchConfig variable in the patch module (default: config)",
    )

    args = parser.parse_args()

    try:
        # Import the patch module
        patch_module = importlib.import_module(args.patch_module)
        config = getattr(patch_module, args.config_name)

        output_dir = args.output_dir
        if output_dir is None:
            module_path = Path(patch_module.__file__).resolve()
            if module_path.parent.name == "patches":
                output_dir = module_path.parent.parent / "generated"
            else:
                output_dir = Path("generated")

        if not isinstance(config, PatchConfig):
            print(f"Error: {args.config_name} is not a PatchConfig instance", file=sys.stderr)
            sys.exit(1)

        # Generate the code
        output = generate_from_config(config, output_dir)
        print("\nGeneration complete!")
        print(f"Output written to: {output_dir / config.target_file}")

    except ImportError as e:
        print(f"Error importing patch module: {e}", file=sys.stderr)
        sys.exit(1)
    except AttributeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
