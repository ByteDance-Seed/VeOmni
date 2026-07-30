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
    """``nargs="+"`` takes space-separated values, and also accepts the YAML list spelling."""

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

    def test_single_bracketed_value_is_one_element(self, monkeypatch):
        """``[q_b_proj]`` used to parse as the literal one-element list ``['[q_b_proj]']``."""
        args = _parse(monkeypatch, "--train.names", "[q_b_proj]")
        assert args.train.names == ["q_b_proj"]

    @pytest.mark.parametrize(
        "argv",
        [
            ("[q_proj, k_proj]",),  # shell-quoted, so argparse sees one token
            ("[q_proj,", "k_proj]"),  # unquoted, so the shell splits on the space
            ("[q_proj,k_proj]",),  # no space after the separator
            ("[", "q_proj", ",", "k_proj", "]"),  # spaces around every separator
        ],
    )
    def test_flow_sequence_spellings_agree(self, monkeypatch, argv):
        """Whitespace inside a flow sequence is YAML's business, not the shell's."""
        args = _parse(monkeypatch, "--train.names", *argv)
        assert args.train.names == ["q_proj", "k_proj"]

    def test_flow_sequence_in_equals_form(self, monkeypatch):
        args = _parse(monkeypatch, "--train.names=[q_proj, k_proj]")
        assert args.train.names == ["q_proj", "k_proj"]

    def test_empty_flow_sequence_clears_the_list(self, monkeypatch, tmp_path):
        """The only way to empty a list set by a config file, since ``nargs="+"`` needs a value."""
        config = tmp_path / "config.yaml"
        config.write_text("train:\n  names: [q_proj]\n")

        args = _parse(monkeypatch, str(config), "--train.names", "[]")
        assert args.train.names == []

    def test_non_string_flow_sequence_is_converted(self, monkeypatch):
        args = _parse(monkeypatch, "--train.sizes", "[1, 2]")
        assert args.train.sizes == [1, 2]

    @pytest.mark.parametrize("value", ["['LayerNorm', 'bias']", '["LayerNorm", "bias"]'])
    def test_quoted_elements_lose_their_quotes(self, monkeypatch, value):
        """Quoting scalars is ordinary YAML, so a config copy-paste must survive it."""
        args = _parse(monkeypatch, "--train.names", value)
        assert args.train.names == ["LayerNorm", "bias"]

    def test_bool_elements_are_not_resolved_by_yaml(self, monkeypatch):
        """The field's element type owns conversion; YAML must not decide it first."""
        args = _parse(monkeypatch, "--train.flags", "[true, no]")
        assert args.train.flags == [True, False]

    @pytest.mark.parametrize(
        "argv, element",
        [
            (("[q_proj k_proj]",), "q_proj k_proj"),  # every comma forgotten
            (("[q_proj", "k_proj]"), "q_proj k_proj"),  # same, unquoted
            (("[q_proj, k_proj v_proj]",), "k_proj v_proj"),  # one comma forgotten
        ],
    )
    def test_missing_comma_inside_brackets_is_rejected(self, monkeypatch, capsys, argv, element):
        """YAML reads this as one longer element; the message must name that element."""
        error = _parse_error(monkeypatch, capsys, "--train.names", *argv)
        assert f"element {element!r}" in error

    def test_whitespace_is_only_refused_inside_brackets(self, monkeypatch):
        """Space separation cannot express a space, so an unbracketed value keeps it."""
        args = _parse(monkeypatch, "--train.names", "a b")
        assert args.train.names == ["a b"]

    @pytest.mark.parametrize("value", ["[a,,b]", "[,]"])
    def test_malformed_sequence_is_rejected(self, monkeypatch, capsys, value):
        """YAML rejects these, so they must not silently drop the blank element."""
        assert "contains a comma" in _parse_error(monkeypatch, capsys, "--train.names", value)

    def test_nested_sequence_is_not_treated_as_a_list(self, monkeypatch, capsys):
        """A nested list is not something a ``List[str]`` field can hold."""
        assert "contains a comma" in _parse_error(monkeypatch, capsys, "--train.names", "[a, [b, c]]")

    def test_mapping_stays_a_literal_element(self, monkeypatch):
        """Not a list of scalars, and no comma to object to, so it is just a value."""
        args = _parse(monkeypatch, "--train.names", "[a: b]")
        assert args.train.names == ["[a: b]"]

    def test_flow_sequence_element_failure_is_reported(self, monkeypatch, capsys):
        """The element, not the whole group, is what the user has to fix."""
        assert "invalid value 'two'" in _parse_error(monkeypatch, capsys, "--train.sizes", "[1, two]")

    @pytest.mark.parametrize("argv", [("q_proj,", "k_proj"), ("q_proj,k_proj",)])
    def test_comma_without_brackets_is_rejected(self, monkeypatch, capsys, argv):
        """Under space separation ``a,b`` is one element, so a bare comma is always a mistake."""
        assert "separated by spaces" in _parse_error(monkeypatch, capsys, "--train.names", *argv)

    @pytest.mark.parametrize(
        "argv",
        [
            ("data/part-[0-9]", "[qk]_proj"),  # brackets present, but not wrapping a sequence
            ("[qk]v_proj", "gate[0]"),  # same values reordered: wrapping alone must not decide
        ],
    )
    def test_unwrapped_brackets_inside_a_value_are_kept(self, monkeypatch, argv):
        """An inner bracket says these are bracketed values, not one bracketed sequence."""
        args = _parse(monkeypatch, "--train.names", *argv)
        assert args.train.names == list(argv)

    def test_a_value_made_only_of_brackets_is_read_as_a_sequence(self, monkeypatch):
        """Known cost of accepting YAML syntax: a glob that *is* a bracket group loses them.

        Reachable only when the whole element is bracketed (``[0-9]*.parquet`` keeps
        its brackets, as the case above shows).
        """
        args = _parse(monkeypatch, "--train.names", "[0-9]")
        assert args.train.names == ["0-9"]

    def test_quoting_inside_the_sequence_keeps_the_brackets(self, monkeypatch):
        """The escape hatch for the case above, and the same one YAML itself offers."""
        args = _parse(monkeypatch, "--train.names", '["[0-9]"]')
        assert args.train.names == ["[0-9]"]

    def test_non_list_fields_still_accept_brackets(self, monkeypatch):
        """Flow-sequence handling is scoped to lists; a path may legitimately be bracketed."""
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
