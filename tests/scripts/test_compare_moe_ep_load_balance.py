import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/profile/compare_moe_ep_load_balance.py"


def _load_reporter():
    assert SCRIPT.is_file(), f"comparison reporter is missing: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("compare_moe_ep_load_balance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(metrics=None, metadata=None):
    return {
        "schema_version": 1,
        "metadata": {} if metadata is None else metadata,
        "metrics": {
            "loss": [1.0, 2.0],
            "grad_norm": [4.0, 2.0],
            **({} if metrics is None else metrics),
        },
    }


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _compare(baseline, candidate, **kwargs):
    reporter = _load_reporter()
    options = {
        "warmup_steps": 0,
        "rtol": 0.1,
        "atol": 0.1,
        "relative_error_threshold": 0.1,
        "epsilon": 1e-12,
    }
    options.update(kwargs)
    return reporter.compare_runs(baseline, candidate, **options)


def test_matched_curves_report_precision_metrics():
    baseline = _canonical()
    candidate = _canonical(metrics={"loss": [1.05, 1.8], "grad_norm": [4.1, 1.9]})

    report = _compare(baseline, candidate)

    assert report["precision"]["status"] == "pass"
    assert report["precision"]["passed"] is True
    assert report["precision"]["loss"]["max_absolute_error"] == pytest.approx(0.2)
    assert report["precision"]["loss"]["max_relative_error"] == pytest.approx(0.1)
    assert report["precision"]["loss"]["repository_close_hit_rate"] == 1.0
    assert report["precision"]["loss"]["threshold_relative_hit_rate"] == 1.0
    assert report["precision"]["loss"]["pearson_correlation"] == pytest.approx(1.0)
    assert report["precision"]["loss"]["correlation_unavailable_reason"] is None
    assert report["precision"]["grad_norm"]["repository_close_hit_rate"] == 1.0


def test_mismatched_curve_fails_precision_if_one_point_is_outside_envelope():
    baseline = _canonical()
    candidate = _canonical(metrics={"loss": [1.0, 2.31]})

    report = _compare(baseline, candidate)

    assert report["precision"]["status"] == "fail"
    assert report["precision"]["passed"] is False
    assert report["precision"]["loss"]["repository_close_hit_rate"] == 0.5
    assert report["precision"]["grad_norm"]["repository_close_hit_rate"] == 1.0


@pytest.mark.parametrize(
    ("baseline_values", "candidate_values", "expected_reason"),
    [
        ([1.0], [1.0], "fewer than two points"),
        ([1.0, 1.0], [1.0, 1.0], "baseline has zero variance"),
        ([1.0, 2.0], [3.0, 3.0], "candidate has zero variance"),
    ],
)
def test_correlation_is_unavailable_instead_of_invented(baseline_values, candidate_values, expected_reason):
    baseline = _canonical(metrics={"loss": baseline_values, "grad_norm": baseline_values})
    candidate = _canonical(metrics={"loss": candidate_values, "grad_norm": candidate_values})

    report = _compare(baseline, candidate, rtol=10.0, atol=10.0)

    loss = report["precision"]["loss"]
    assert loss["pearson_correlation"] is None
    assert loss["correlation_unavailable_reason"] == expected_reason


def test_near_zero_relative_error_uses_epsilon_denominator():
    baseline = _canonical(metrics={"loss": [0.0], "grad_norm": [0.0]})
    candidate = _canonical(metrics={"loss": [1e-10], "grad_norm": [5e-11]})

    report = _compare(
        baseline,
        candidate,
        rtol=0.0,
        atol=1e-9,
        relative_error_threshold=0.2,
        epsilon=1e-9,
    )

    assert report["precision"]["loss"]["max_relative_error"] == pytest.approx(0.1)
    assert report["precision"]["loss"]["threshold_relative_hit_rate"] == 1.0
    assert report["precision"]["passed"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload["metrics"].pop("loss"), "required metric 'loss'"),
        (lambda payload: payload["metrics"].pop("grad_norm"), "required metric 'grad_norm'"),
        (lambda payload: payload["metrics"].update(loss=[]), "must not be empty"),
        (lambda payload: payload["metrics"].update(loss=[True, 1.0]), "bool"),
        (lambda payload: payload["metrics"].update(loss=["1.0", 2.0]), "numeric"),
        (lambda payload: payload["metrics"].update(loss=[math.nan, 2.0]), "finite"),
        (lambda payload: payload["metrics"].update(loss=[math.inf, 2.0]), "finite"),
        (lambda payload: payload["metrics"].update(grad_norm=[1.0]), "equal lengths"),
    ],
)
def test_invalid_required_curves_are_rejected(mutate, match):
    reporter = _load_reporter()
    payload = _canonical()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        reporter.normalize_run(payload, source="invalid.json")


def test_cross_run_curve_length_mismatch_is_rejected():
    baseline = _canonical()
    candidate = _canonical(metrics={"loss": [1.0], "grad_norm": [4.0]})

    with pytest.raises(ValueError, match="baseline and candidate.*equal lengths"):
        _compare(baseline, candidate)


@pytest.mark.parametrize(
    "metric",
    ["step_time_s", "step_tokens", "tokens_per_second", "p2p_bytes", "p2p_wait_time_s"],
)
def test_each_step_performance_curve_must_align_with_required_curves_in_the_same_run(metric):
    reporter = _load_reporter()
    payload = _canonical(metrics={metric: [1.0]})

    with pytest.raises(ValueError, match=rf"{metric}.*same number of steps"):
        reporter.normalize_run(payload, source="misaligned-performance.json")


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "metadata": {}, "metrics": {"loss": [1.0], "grad_norm": [1.0]}},
        {"schema_version": 1, "metadata": [], "metrics": {"loss": [1.0], "grad_norm": [1.0]}},
        {"schema_version": 1, "metadata": {}, "metrics": []},
    ],
)
def test_invalid_canonical_envelopes_are_rejected(payload):
    reporter = _load_reporter()

    with pytest.raises(ValueError):
        reporter.normalize_run(payload, source="invalid.json")


def test_canonical_schema_version_requires_an_integer_not_a_float():
    reporter = _load_reporter()
    payload = _canonical()
    payload["schema_version"] = 1.0

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        reporter.normalize_run(payload, source="invalid-version.json")


def test_explicit_critical_metadata_mismatch_is_rejected():
    baseline = _canonical(metadata={"seed": 0, "model": "qwen", "run_label": "baseline"})
    candidate = _canonical(metadata={"seed": 1, "model": "qwen", "run_label": "candidate"})

    with pytest.raises(ValueError, match="critical metadata mismatch.*seed"):
        _compare(baseline, candidate)


@pytest.mark.parametrize(("baseline_value", "candidate_value"), [(True, 1), (1, 1.0)])
def test_critical_metadata_comparison_is_type_sensitive(baseline_value, candidate_value):
    baseline = _canonical(metadata={"world_size": baseline_value})
    candidate = _canonical(metadata={"world_size": candidate_value})

    with pytest.raises(ValueError, match="critical metadata mismatch.*world_size"):
        _compare(baseline, candidate)


def test_noncritical_and_one_sided_metadata_do_not_block_comparison():
    baseline = _canonical(metadata={"seed": 0, "run_label": "baseline", "output_dir": "a"})
    candidate = _canonical(metadata={"seed": 0, "run_label": "candidate", "feature_enabled": True})

    report = _compare(baseline, candidate)

    assert report["metadata_validation"]["status"] == "pass"
    assert report["metadata_validation"]["compared_fields"] == ["seed"]


def test_all_explicit_performance_fields_report_matched_aggregates_and_directions():
    baseline = _canonical(
        metrics={
            "loss": [1.0, 1.0, 1.0, 1.0],
            "grad_norm": [2.0, 2.0, 2.0, 2.0],
            "step_time_s": [10.0, 4.0, 2.0, 2.0],
            "step_tokens": [100.0, 100.0, 100.0, 100.0],
            "tokens_per_second": [10.0, 25.0, 50.0, 50.0],
            "peak_accelerator_memory_bytes": [1000.0, 1200.0],
            "peak_host_memory_bytes": 3000.0,
        }
    )
    candidate = _canonical(
        metrics={
            "loss": [1.0, 1.0, 1.0, 1.0],
            "grad_norm": [2.0, 2.0, 2.0, 2.0],
            "step_time_s": [8.0, 2.0, 1.0, 1.0],
            "step_tokens": [100.0, 100.0, 100.0, 100.0],
            "tokens_per_second": [12.5, 50.0, 100.0, 100.0],
            "peak_accelerator_memory_bytes": 1000.0,
            "peak_host_memory_bytes": [3100.0, 3300.0],
        }
    )

    performance = _compare(baseline, candidate, warmup_steps=1)["performance"]

    timing = performance["timing"]
    assert timing["status"] == "available"
    assert timing["baseline"]["warmup_mean_s"] == 10.0
    assert timing["baseline"]["steady_mean_s"] == pytest.approx(8.0 / 3.0)
    assert timing["candidate"]["steady_mean_s"] == pytest.approx(4.0 / 3.0)
    assert timing["steady_delta_s"] == pytest.approx(-4.0 / 3.0)
    assert timing["steady_speedup"] == pytest.approx(2.0)

    explicit = performance["explicit_tokens_per_second"]
    assert explicit["status"] == "available"
    assert explicit["baseline_steady_tokens_per_second"] == pytest.approx(125.0 / 3.0)
    assert explicit["candidate_steady_tokens_per_second"] == pytest.approx(250.0 / 3.0)
    assert explicit["steady_delta_tokens_per_second"] == pytest.approx(125.0 / 3.0)
    assert explicit["steady_speedup"] == pytest.approx(2.0)

    derived = performance["derived_tokens_per_second"]
    assert derived["status"] == "available"
    assert derived["baseline_steady_tokens_per_second"] == pytest.approx(37.5)
    assert derived["candidate_steady_tokens_per_second"] == pytest.approx(75.0)
    assert derived["steady_speedup"] == pytest.approx(2.0)

    accelerator_memory = performance["peak_accelerator_memory_bytes"]
    assert accelerator_memory == {
        "status": "available",
        "reason": None,
        "baseline_peak_bytes": 1200.0,
        "candidate_peak_bytes": 1000.0,
        "delta_bytes": -200.0,
        "baseline_over_candidate_ratio": 1.2,
    }
    host_memory = performance["peak_host_memory_bytes"]
    assert host_memory["baseline_peak_bytes"] == 3000.0
    assert host_memory["candidate_peak_bytes"] == 3300.0
    assert host_memory["delta_bytes"] == 300.0
    assert host_memory["baseline_over_candidate_ratio"] == pytest.approx(3000.0 / 3300.0)


def test_p2p_curves_report_warmup_and_steady_aggregates():
    baseline = _canonical(
        metrics={
            "loss": [1.0, 1.0, 1.0, 1.0],
            "grad_norm": [2.0, 2.0, 2.0, 2.0],
            "p2p_bytes": [100.0, 200.0, 300.0, 500.0],
            "p2p_wait_time_s": [4.0, 2.0, 1.0, 1.0],
        }
    )
    candidate = _canonical(
        metrics={
            "loss": [1.0, 1.0, 1.0, 1.0],
            "grad_norm": [2.0, 2.0, 2.0, 2.0],
            "p2p_bytes": [80.0, 100.0, 120.0, 280.0],
            "p2p_wait_time_s": [3.0, 1.0, 0.5, 0.5],
        }
    )

    performance = _compare(baseline, candidate, warmup_steps=1)["performance"]

    p2p_bytes = performance["p2p_bytes"]
    assert p2p_bytes["status"] == "available"
    assert p2p_bytes["baseline"] == {
        "warmup_total_bytes": 100.0,
        "warmup_mean_bytes": 100.0,
        "steady_total_bytes": 1000.0,
        "steady_mean_bytes": pytest.approx(1000.0 / 3.0),
    }
    assert p2p_bytes["candidate"] == {
        "warmup_total_bytes": 80.0,
        "warmup_mean_bytes": 80.0,
        "steady_total_bytes": 500.0,
        "steady_mean_bytes": pytest.approx(500.0 / 3.0),
    }
    assert p2p_bytes["steady_total_delta_bytes"] == -500.0
    assert p2p_bytes["steady_total_ratio"] == 0.5
    assert p2p_bytes["steady_mean_delta_bytes"] == pytest.approx(-500.0 / 3.0)
    assert p2p_bytes["steady_mean_ratio"] == 0.5

    p2p_wait = performance["p2p_wait_time_s"]
    assert p2p_wait["status"] == "available"
    assert p2p_wait["baseline"]["warmup_mean_s"] == 4.0
    assert p2p_wait["baseline"]["steady_mean_s"] == pytest.approx(4.0 / 3.0)
    assert p2p_wait["candidate"]["warmup_mean_s"] == 3.0
    assert p2p_wait["candidate"]["steady_mean_s"] == pytest.approx(2.0 / 3.0)
    assert p2p_wait["steady_delta_s"] == pytest.approx(-2.0 / 3.0)
    assert p2p_wait["steady_speedup"] == 2.0


@pytest.mark.parametrize(
    ("metric", "section"),
    [
        ("step_time_s", "timing"),
        ("tokens_per_second", "explicit_tokens_per_second"),
        ("p2p_bytes", "p2p_bytes"),
        ("p2p_wait_time_s", "p2p_wait_time_s"),
        ("peak_accelerator_memory_bytes", "peak_accelerator_memory_bytes"),
        ("peak_host_memory_bytes", "peak_host_memory_bytes"),
    ],
)
@pytest.mark.parametrize(
    ("present_on", "reason_fragment"),
    [
        ("neither", "missing on both baseline and candidate"),
        ("baseline", "missing on candidate"),
        ("candidate", "missing on baseline"),
    ],
)
def test_performance_field_missing_combinations_are_unavailable(metric, section, present_on, reason_fragment):
    value = 1.0 if metric.startswith("peak_") else [1.0, 1.0]
    baseline_metrics = {metric: value} if present_on == "baseline" else {}
    candidate_metrics = {metric: value} if present_on == "candidate" else {}

    result = _compare(_canonical(metrics=baseline_metrics), _canonical(metrics=candidate_metrics))["performance"][
        section
    ]

    assert result["status"] == "unavailable"
    assert reason_fragment in result["reason"]
    for key, number in result.items():
        if key not in {"status", "reason"}:
            assert number is None


@pytest.mark.parametrize(
    ("baseline_metrics", "candidate_metrics", "reason_fragment"),
    [
        ({}, {}, "step_time_s and step_tokens missing on both baseline and candidate"),
        (
            {"step_time_s": [1.0, 1.0], "step_tokens": [8.0, 8.0]},
            {"step_time_s": [1.0, 1.0]},
            "step_tokens missing on candidate",
        ),
        (
            {"step_time_s": [1.0, 1.0]},
            {"step_time_s": [1.0, 1.0], "step_tokens": [8.0, 8.0]},
            "step_tokens missing on baseline",
        ),
    ],
)
def test_derived_throughput_missing_combinations_are_unavailable(baseline_metrics, candidate_metrics, reason_fragment):
    derived = _compare(_canonical(metrics=baseline_metrics), _canonical(metrics=candidate_metrics))["performance"][
        "derived_tokens_per_second"
    ]

    assert derived["status"] == "unavailable"
    assert reason_fragment in derived["reason"]
    assert derived["baseline_steady_tokens_per_second"] is None
    assert derived["candidate_steady_tokens_per_second"] is None
    assert derived["steady_delta_tokens_per_second"] is None
    assert derived["steady_speedup"] is None


@pytest.mark.parametrize("metric", ["step_time_s", "p2p_bytes", "p2p_wait_time_s"])
@pytest.mark.parametrize("warmup_steps", [-1, 2, 3])
def test_warmup_bounds_are_rejected_when_performance_curves_are_present(metric, warmup_steps):
    baseline = _canonical(metrics={metric: [2.0, 1.0]})
    candidate = _canonical(metrics={metric: [2.0, 1.0]})

    with pytest.raises(ValueError, match="warmup_steps"):
        _compare(baseline, candidate, warmup_steps=warmup_steps)


@pytest.mark.parametrize("metric", ["p2p_bytes", "p2p_wait_time_s"])
def test_p2p_curves_reject_negative_values(metric):
    reporter = _load_reporter()
    payload = _canonical(metrics={metric: [1.0, -0.1]})

    with pytest.raises(ValueError, match=rf"{metric}.*non-negative"):
        reporter.normalize_run(payload, source="invalid-p2p.json")


def test_optional_performance_curves_reject_nonfinite_bool_and_same_run_length_mismatch():
    reporter = _load_reporter()
    for invalid in ([1.0, math.nan], [1.0, True]):
        payload = _canonical(metrics={"step_time_s": invalid})
        with pytest.raises(ValueError):
            reporter.normalize_run(payload, source="invalid-performance.json")

    baseline = _canonical(metrics={"tokens_per_second": [1.0, 2.0]})
    candidate = _canonical(metrics={"tokens_per_second": [1.0]})
    with pytest.raises(ValueError, match="tokens_per_second.*same number of steps"):
        _compare(baseline, candidate)


def test_canonical_flat_jsonl_and_plain_log_inputs_are_normalized(tmp_path):
    reporter = _load_reporter()
    canonical_path = _write_json(tmp_path / "canonical.json", _canonical(metadata={"seed": 0}))
    flat_path = _write_json(tmp_path / "log_dict.json", {"loss": [1.0, 2.0], "grad_norm": [4.0, 2.0]})
    jsonl_path = tmp_path / "steps.jsonl"
    jsonl_path.write_text(
        '{"metadata":{"seed":0},"loss":1.0,"grad_norm":4.0,"step_time_s":2.0}\n'
        '{"metadata":{"seed":0},"loss":2.0,"grad_norm":2.0,"step_time_s":1.0}\n',
        encoding="utf-8",
    )
    plain_path = tmp_path / "train.log"
    plain_path.write_text(
        "rank=0 step: 1, loss: 1.0, grad_norm: 4.0\nrank=0 step: 2 loss=2.0 gradient_norm=2.0\n",
        encoding="utf-8",
    )

    canonical = reporter.load_run(canonical_path)
    flat = reporter.load_run(flat_path)
    jsonl = reporter.load_run(jsonl_path)
    plain = reporter.load_run(plain_path)

    for run in (canonical, flat, jsonl, plain):
        assert run["metrics"]["loss"] == [1.0, 2.0]
        assert run["metrics"]["grad_norm"] == [4.0, 2.0]
    assert canonical["metadata"] == {"seed": 0}
    assert flat["metadata"] == {}
    assert jsonl["metadata"] == {"seed": 0}
    assert jsonl["metrics"]["step_time_s"] == [2.0, 1.0]
    assert plain["metadata"] == {}


def test_jsonl_rejects_inconsistent_metadata_and_partial_metric_rows(tmp_path):
    reporter = _load_reporter()
    inconsistent = tmp_path / "inconsistent.jsonl"
    inconsistent.write_text(
        '{"metadata":{"seed":0},"loss":1.0,"grad_norm":2.0}\n{"metadata":{"seed":1},"loss":1.0,"grad_norm":2.0}\n',
        encoding="utf-8",
    )
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"loss":1.0,"grad_norm":2.0,"step_time_s":1.0}\n{"loss":1.0,"grad_norm":2.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata changes"):
        reporter.load_run(inconsistent)
    with pytest.raises(ValueError, match="step_time_s.*missing from JSONL step"):
        reporter.load_run(partial)


def test_jsonl_rejects_metadata_change_when_first_metadata_object_is_empty(tmp_path):
    reporter = _load_reporter()
    path = tmp_path / "empty-first-metadata.jsonl"
    path.write_text(
        '{"metadata":{},"loss":1.0,"grad_norm":2.0}\n{"metadata":{"seed":0},"loss":1.0,"grad_norm":2.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata changes"):
        reporter.load_run(path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_jsonl_schema_version_requires_an_integer_not_a_bool_or_float(tmp_path, schema_version):
    reporter = _load_reporter()
    path = tmp_path / "invalid-version.jsonl"
    path.write_text(
        json.dumps({"schema_version": schema_version, "loss": 1.0, "grad_norm": 2.0})
        + "\n"
        + json.dumps({"schema_version": 1, "loss": 1.0, "grad_norm": 2.0})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"invalid-version\.jsonl:1: schema_version must be integer 1"):
        reporter.load_run(path)


def test_plain_log_without_both_required_values_is_rejected(tmp_path):
    reporter = _load_reporter()
    path = tmp_path / "train.log"
    path.write_text("step=1 loss=1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required metric 'grad_norm'"):
        reporter.load_run(path)


def test_plain_log_accepts_repository_total_loss_name(tmp_path):
    reporter = _load_reporter()
    path = tmp_path / "train.log"
    path.write_text(
        "Epoch 1/1: 1/2 total_loss: 1.25, grad_norm: 2.5\nEpoch 1/1: 2/2 total_loss=1.0, grad_norm=2.0\n",
        encoding="utf-8",
    )

    run = reporter.load_run(path)

    assert run["metrics"]["loss"] == [1.25, 1.0]
    assert run["metrics"]["grad_norm"] == [2.5, 2.0]


def _run_cli(tmp_path: Path, baseline_payload, candidate_payload, stem: str):
    baseline = _write_json(tmp_path / f"{stem}-baseline.json", baseline_payload)
    candidate = _write_json(tmp_path / f"{stem}-candidate.json", candidate_payload)
    json_out = tmp_path / f"{stem}.report.json"
    markdown_out = tmp_path / f"{stem}.report.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--warmup-steps",
            "0",
            "--rtol",
            "0.1",
            "--atol",
            "0.1",
            "--relative-error-threshold",
            "0.1",
            "--epsilon",
            "1e-12",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json_out, markdown_out


def test_cli_pass_is_zero_and_outputs_are_byte_deterministic(tmp_path):
    first, first_json, first_markdown = _run_cli(tmp_path, _canonical(), _canonical(), "first")
    second, second_json, second_markdown = _run_cli(tmp_path, _canonical(), _canonical(), "second")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_markdown.read_bytes() == second_markdown.read_bytes()
    parsed = json.loads(first_json.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    assert parsed["precision"]["status"] == "pass"
    markdown = first_markdown.read_text(encoding="utf-8")
    assert "# MoE EP Load-Balance Comparison" in markdown
    assert "Precision: PASS" in markdown


def test_json_and_markdown_reports_include_warmup_timing():
    reporter = _load_reporter()
    baseline = _canonical(
        metrics={
            "loss": [1.0, 1.0, 1.0],
            "grad_norm": [2.0, 2.0, 2.0],
            "step_time_s": [9.0, 3.0, 1.0],
        }
    )
    candidate = _canonical(
        metrics={
            "loss": [1.0, 1.0, 1.0],
            "grad_norm": [2.0, 2.0, 2.0],
            "step_time_s": [6.0, 2.0, 1.0],
        }
    )

    report = _compare(baseline, candidate, warmup_steps=1)
    json_report = json.loads(reporter._json_text(report))
    markdown_report = reporter.render_markdown(report)

    assert json_report["performance"]["warmup_steps"] == 1
    assert json_report["performance"]["timing"]["baseline"]["warmup_mean_s"] == 9.0
    assert json_report["performance"]["timing"]["candidate"]["warmup_mean_s"] == 6.0
    assert "Baseline warmup mean (s): 9" in markdown_report
    assert "Candidate warmup mean (s): 6" in markdown_report


def test_json_and_markdown_reports_include_p2p_aggregates():
    reporter = _load_reporter()
    baseline = _canonical(
        metrics={
            "loss": [1.0, 1.0],
            "grad_norm": [2.0, 2.0],
            "p2p_bytes": [10.0, 30.0],
            "p2p_wait_time_s": [2.0, 1.0],
        }
    )
    candidate = _canonical(
        metrics={
            "loss": [1.0, 1.0],
            "grad_norm": [2.0, 2.0],
            "p2p_bytes": [8.0, 15.0],
            "p2p_wait_time_s": [1.5, 0.5],
        }
    )

    report = _compare(baseline, candidate, warmup_steps=1)
    json_report = json.loads(reporter._json_text(report))
    markdown_report = reporter.render_markdown(report)

    assert json_report["performance"]["p2p_bytes"]["baseline"]["steady_total_bytes"] == 30.0
    assert json_report["performance"]["p2p_bytes"]["candidate"]["steady_total_bytes"] == 15.0
    assert json_report["performance"]["p2p_wait_time_s"]["baseline"]["warmup_mean_s"] == 2.0
    assert json_report["performance"]["p2p_wait_time_s"]["candidate"]["steady_mean_s"] == 0.5
    assert "P2P bytes: available" in markdown_report
    assert "Baseline steady total bytes: 30" in markdown_report
    assert "P2P wait time: available" in markdown_report
    assert "Candidate steady mean (s): 0.5" in markdown_report


def test_cli_precision_failure_is_nonzero_after_writing_outputs(tmp_path):
    candidate = _canonical(metrics={"loss": [3.0, 4.0]})

    completed, json_out, markdown_out = _run_cli(tmp_path, _canonical(), candidate, "fail")

    assert completed.returncode == 1
    assert json_out.is_file()
    assert markdown_out.is_file()
    assert json.loads(json_out.read_text(encoding="utf-8"))["precision"]["status"] == "fail"
    assert "Precision: FAIL" in markdown_out.read_text(encoding="utf-8")


def test_cli_invalid_input_returns_two_with_actionable_error(tmp_path):
    invalid = _canonical()
    invalid["metrics"]["loss"] = [True, 2.0]

    completed, json_out, markdown_out = _run_cli(tmp_path, invalid, _canonical(), "invalid")

    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "bool" in completed.stderr
    assert not json_out.exists()
    assert not markdown_out.exists()
