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

import re
from typing import List

import torch.nn as nn


def set_module_from_path(model: nn.Module, path: str, value: any):
    attrs = path.split(".")
    if len(attrs) == 1:
        setattr(model, attrs[0], value)
    else:
        next_obj = getattr(model, attrs[0])
        set_module_from_path(next_obj, ".".join(attrs[1:]), value)


def get_module_from_path(model: nn.Module, path: str):
    attrs = path.split(".")
    if len(attrs) == 1:
        return getattr(model, attrs[0])
    else:
        next_obj = getattr(model, attrs[0])
        return get_module_from_path(next_obj, ".".join(attrs[1:]))


def is_same_module_from_path(model: nn.Module, module_name: str, path: str):
    attrs = path.split(".")
    if len(attrs) == 1:
        return getattr(model, attrs[0]).__class__.__name__ == module_name
    else:
        next_obj = getattr(model, attrs[0])
        return is_same_module_from_path(next_obj, module_name, ".".join(attrs[1:]))


def check_all_fqn_match(path_patterns: List[str], path_keys: List[str]):
    """
    Check
    """
    assert isinstance(path_patterns, list), f"path_patterns must be a list, got {type(path_patterns)}"
    assert isinstance(path_keys, (list, tuple)), f"path_keys must be a list or tuple, got {type(path_keys)}"

    if len(path_patterns) != len(path_keys):
        return False

    regex_list = []
    for pattern in path_patterns:
        regex_str = re.escape(pattern).replace(r"\*", r"(\d+)")
        regex_str = f"^{regex_str}$"
        regex_list.append((pattern, re.compile(regex_str)))

    used_patterns = set()
    expected_num = None  # the first matched number

    for key in path_keys:
        matched = False
        for p, regex in regex_list:
            if p in used_patterns:
                continue
            match = regex.match(key)
            if match:
                current_num = match.group(1)
                if expected_num is None:
                    expected_num = current_num
                elif current_num != expected_num:
                    return False
                used_patterns.add(p)
                matched = True
                break
        if not matched:
            return False

    return True


def check_any_fqn_match(path_patterns: List[str], path_key: str, return_idx: bool = False, prefix: str = None):
    assert isinstance(path_patterns, list), f"path_patterns must be a list, got {type(path_patterns)}"
    assert isinstance(path_key, str), f"path_key must be a str, got {type(path_key)}"

    if prefix:
        path_patterns = [".".join([prefix, pattern]) for pattern in path_patterns]

    regex_list = []
    for pattern in path_patterns:
        regex_str = re.escape(pattern).replace(r"\*", r"(\d+)")
        regex_str = f"^{regex_str}$"
        regex_list.append(re.compile(regex_str))

    for idx, regex in enumerate(regex_list):
        match = regex.match(path_key)
        if match:
            return idx if return_idx else True

    return -1 if return_idx else False


def check_fqn_match(fqn_pattern: str, fqn: str, prefix: str = None):
    assert isinstance(fqn_pattern, str), f"fqn_pattern must be a str, got {type(fqn_pattern)}"
    assert isinstance(fqn, str), f"fqn must be a str, got {type(fqn)}"

    if prefix:
        fqn_pattern = [".".join([prefix, pattern]) for pattern in fqn_pattern]

    regex_str = re.escape(fqn_pattern).replace(r"\*", r".*")
    regex_str = f"^{regex_str}$"
    regex = re.compile(regex_str)

    match = regex.match(fqn)

    return match


def sort_fqn_by_submodule_first(fqn_list: list[str]) -> list[str]:
    """Sort FQNs so every descendant precedes its ancestors while preserving sibling order."""
    first_index: dict[str, int] = {}
    for idx, fqn in enumerate(fqn_list):
        first_index.setdefault(fqn, idx)

    def _nearest_ancestor_index(fqn: str) -> int | None:
        if not fqn:
            return None
        pos = fqn.rfind(".")
        while pos != -1:
            candidate = fqn[:pos]
            if candidate in first_index:
                return first_index[candidate]
            pos = fqn.rfind(".", 0, pos)
        return first_index.get("")

    children: dict[int | None, list[int]] = {}
    for idx, fqn in enumerate(fqn_list):
        parent_idx = _nearest_ancestor_index(fqn)
        children.setdefault(parent_idx, []).append(idx)

    result: list[str] = []

    def _visit(parent_idx: int | None) -> None:
        for child_idx in children.get(parent_idx, ()):
            _visit(child_idx)
            result.append(fqn_list[child_idx])

    _visit(None)
    return result
