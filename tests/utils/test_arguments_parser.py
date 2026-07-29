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

"""CLI parsing rules of ``veomni.arguments.parse_args``."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from veomni.arguments import parse_args


@dataclass
class _SubConfig:
    names: List[str] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    coefficients: List[float] = field(default_factory=list)
    flags: List[bool] = field(default_factory=list)
    optional_names: Optional[List[str]] = None
    model_path: str = ""


@dataclass
class _RootConfig:
    train: "_SubConfig" = field(default_factory=_SubConfig)


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str) -> _RootConfig:
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return parse_args(_RootConfig)


def _parse_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, *argv: str) -> str:
    with pytest.raises(SystemExit) as excinfo:
        _parse(monkeypatch, *argv)
    assert excinfo.value.code == 2  # argparse usage error, not a help exit
    return capsys.readouterr().err


class TestListArguments:
    """``nargs="+"`` takes space-separated values; YAML list syntax must not slip through."""

    def test_space_separated_values(self, monkeypatch):
        args = _parse(monkeypatch, "--train.names", "q_proj", "k_proj")
        assert args.train.names == ["q_proj", "k_proj"]

    def test_numeric_and_bool_elements_are_converted(self, monkeypatch):
        args = _parse(
            monkeypatch, "--train.sizes", "1", "2", "--train.coefficients", "3.4445", "--train.flags", "true", "false"
        )
        assert (args.train.sizes, args.train.coefficients, args.train.flags) == ([1, 2], [3.4445], [True, False])

    def test_negative_numbers_are_accepted(self, monkeypatch):
        """``muon_ns_coefficients`` ships a negative default, so this is a live shape."""
        args = _parse(monkeypatch, "--train.coefficients", "-4.7750")
        assert args.train.coefficients == [-4.7750]

    def test_optional_list_is_unwrapped(self, monkeypatch):
        args = _parse(monkeypatch, "--train.optional_names", "q_proj")
        assert args.train.optional_names == ["q_proj"]

    def test_element_conversion_failure_is_reported(self, monkeypatch, capsys):
        assert "invalid value 'two'" in _parse_error(monkeypatch, capsys, "--train.sizes", "two")

    def test_single_bracketed_value_is_rejected(self, monkeypatch, capsys):
        """``[q_b_proj]`` used to parse as the literal one-element list ``['[q_b_proj]']``."""
        assert "is YAML list syntax" in _parse_error(monkeypatch, capsys, "--train.names", "[q_b_proj]")

    def test_bracketed_value_is_rejected_in_equals_form(self, monkeypatch, capsys):
        assert "is YAML list syntax" in _parse_error(monkeypatch, capsys, "--train.names=[q_b_proj]")

    def test_bracketed_value_split_across_tokens_is_rejected(self, monkeypatch, capsys):
        assert "is YAML list syntax" in _parse_error(monkeypatch, capsys, "--train.names", "[q_proj,", "k_proj]")

    def test_non_string_lists_are_rejected_the_same_way(self, monkeypatch, capsys):
        """The element type must not decide whether the diagnosis is understandable."""
        assert "is YAML list syntax" in _parse_error(monkeypatch, capsys, "--train.sizes", "[2]")

    @pytest.mark.parametrize("argv", [("q_proj,", "k_proj"), ("q_proj,k_proj",)])
    def test_comma_separated_values_are_rejected(self, monkeypatch, capsys, argv):
        """Stripping the brackets but keeping the commas is the next version of the mistake."""
        assert "separated by spaces, not commas" in _parse_error(monkeypatch, capsys, "--train.names", *argv)

    @pytest.mark.parametrize(
        "argv",
        [
            ("data/part-[0-9]", "[qk]_proj"),  # brackets present, but not wrapping a sequence
            ("[qk]v_proj", "gate[0]"),  # same values reordered: wrapping alone must not decide
        ],
    )
    def test_unwrapped_brackets_inside_a_value_are_kept(self, monkeypatch, argv):
        """A glob or character class is not a pasted sequence, whatever its position."""
        args = _parse(monkeypatch, "--train.names", *argv)
        assert args.train.names == list(argv)

    def test_non_list_fields_still_accept_brackets(self, monkeypatch):
        """The guard is scoped to lists; a path may legitimately contain brackets."""
        args = _parse(monkeypatch, "--train.model_path", "/models/ckpt[0]")
        assert args.train.model_path == "/models/ckpt[0]"

    def test_yaml_flow_sequence_stays_valid(self, monkeypatch, tmp_path):
        """Brackets are correct YAML, so the config-file path must be untouched."""
        config = tmp_path / "config.yaml"
        config.write_text("train:\n  names: [q_proj, k_proj]\n  sizes: [1, 2]\n")

        args = _parse(monkeypatch, str(config))
        assert args.train.names == ["q_proj", "k_proj"]
        assert args.train.sizes == [1, 2]

    def test_cli_overrides_yaml(self, monkeypatch, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("train:\n  names: [q_proj]\n")

        args = _parse(monkeypatch, str(config), "--train.names", "q_b_proj")
        assert args.train.names == ["q_b_proj"]
