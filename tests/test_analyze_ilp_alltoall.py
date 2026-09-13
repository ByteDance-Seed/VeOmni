import sqlite3
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_SPEC = spec_from_file_location(
    "analyze_ilp_alltoall", Path(__file__).parents[1] / "scripts/profile/analyze_ilp_alltoall.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ANALYZER)
_semantic_steps = _ANALYZER._semantic_steps
_step_mean = _ANALYZER._step_mean


def _write_profile(path: str, *, num_layers: int) -> None:
    with sqlite3.connect(path) as db:
        db.execute("create table STEP_TIME (id integer, startNs integer, endNs integer)")
        db.execute("create table STRING_IDS (id integer, value text)")
        db.execute("create table COMMUNICATION_OP (id integer, opName integer, startNs integer, endNs integer)")
        db.execute("insert into STRING_IDS values (1, 'hcom_alltoall_test')")
        for step, duration in enumerate((100_000_000, 20_000_000, 30_000_000)):
            start = step * 1_000_000_000
            db.execute("insert into STEP_TIME values (?, ?, ?)", (step, start, start + duration))
            for call in range(6 * num_layers):
                call_start = start + call * 10_000
                call_duration = (call + 1) * 1_000
                db.execute(
                    "insert into COMMUNICATION_OP values (?, 1, ?, ?)",
                    (step * 6 * num_layers + call, call_start, call_start + call_duration),
                )


@pytest.mark.parametrize("num_layers", [8, 24, 48])
def test_analyzer_derives_alltoall_layout_from_model_depth(tmp_path, num_layers: int) -> None:
    profile = tmp_path / f"profile-{num_layers}.db"
    _write_profile(str(profile), num_layers=num_layers)

    steps = _semantic_steps(
        str(profile), ilp=True, num_layers=num_layers, current_layer=-1, window_size=0, warmup_steps=2
    )

    assert len(steps) == 1
    assert len(steps[0]) == 3 + 4 * (num_layers - 1)
    assert f"B{num_layers - 1}/replay_dispatch" in steps[0]
    assert "B1/replay_combine" in steps[0]


def test_analyzer_maps_an_interior_ilp_window(tmp_path) -> None:
    profile = tmp_path / "profile.db"
    _write_profile(str(profile), num_layers=8)

    step = _semantic_steps(str(profile), ilp=True, num_layers=8, current_layer=5, window_size=2, warmup_steps=2)[0]

    assert step["B6/replay_dispatch"] == pytest.approx(0.025)
    assert step["B5/replay_dispatch"] == pytest.approx(0.028)
    assert step["B4/replay_dispatch"] == pytest.approx(0.032)
    assert step["B3/replay_dispatch"] == pytest.approx(0.037)


def test_analyzer_excludes_warmup_steps_consistently(tmp_path) -> None:
    profile = tmp_path / "profile.db"
    _write_profile(str(profile), num_layers=8)

    assert _step_mean(str(profile), warmup_steps=1) == pytest.approx(25.0)
    assert _step_mean(str(profile), warmup_steps=2) == pytest.approx(30.0)
