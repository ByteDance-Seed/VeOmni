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

import sys
from types import SimpleNamespace

import veomni.trainer.callbacks.channel_loss_dashboard as dashboard_module
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks.channel_loss_dashboard import (
    ChannelLossDashboardCallback,
    ChannelLossDashboardData,
)
from veomni.trainer.text_dpo_trainer import TextDPOTrainer
from veomni.trainer.text_trainer import TextTrainer
from veomni.trainer.vlm_trainer import VLMTrainer


def _trainer(*, enabled: bool = True):
    config = SimpleNamespace(
        enable=enabled,
        loss_metric_prefix="channel_loss",
        weighted_loss_metric_prefix="channel_loss_weighted",
        token_count_metric_prefix="channel_tokens",
    )
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(
                global_rank=0,
                wandb=SimpleNamespace(enable=True),
                channel_loss=config,
            )
        ),
        channel_loss_callback=SimpleNamespace(config=config, _source_registry={0: "repo__qa", 1: "swecompass"}),
        step_env_metrics={},
        _channel_loss_trace_step_id=None,
    )

    def set_trace_step_id(trace_step_id):
        trainer._channel_loss_trace_step_id = trace_step_id

    def pop_trace_step_id():
        trace_step_id = trainer._channel_loss_trace_step_id
        trainer._channel_loss_trace_step_id = None
        return trace_step_id

    trainer.set_channel_loss_trace_step_id = set_trace_step_id
    trainer.pop_channel_loss_trace_step_id = pop_trace_step_id
    return trainer


def _metrics(step: int = 1):
    return {
        "training/foundation_loss": 0.7 + step / 100,
        "channel_loss/source-i-0__repo_qa": 2.0,
        "channel_loss/source-i-1__swecompass": 0.5,
        "channel_loss_weighted/source-i-0__repo_qa": 0.2,
        "channel_loss_weighted/source-i-1__swecompass": 0.5,
        "channel_tokens/source-i-0__repo_qa": 20,
        "channel_tokens/source-i-1__swecompass": 80,
        "samples/repo_qa": 2,
        "samples/swecompass": 8,
        "input_tokens/repo_qa": 100,
        "input_tokens/swecompass": 400,
        "label_tokens/repo_qa": 20,
        "label_tokens/swecompass": 80,
        "label_tokens_per_sample/repo_qa": 10,
        "label_tokens_per_sample/swecompass": 10,
    }


def test_dashboard_records_scalar_contract_without_splitting_source_names():
    trainer = _trainer()
    data = ChannelLossDashboardData()

    assert data.record(trainer, 1, _metrics())

    assert data.labels == {
        "source-i-0": "repo__qa",
        "source-i-1": "swecompass",
    }
    assert data.points[0]["values"]["source-i-0"] == {
        "raw": 2.0,
        "weighted": 0.2,
        "tokens": 20.0,
        "samples": 2.0,
        "input_tokens": 100.0,
        "label_tokens": 20.0,
        "label_tokens_per_sample": 10.0,
    }


def test_dashboard_resolves_adversarial_data_metric_suffixes_one_to_one():
    trainer = _trainer()
    trainer.channel_loss_callback._source_registry = {0: "foo", 1: "source-i-0__foo"}
    metrics = {
        "channel_loss/source-i-0__foo": 2.0,
        "channel_loss/source-i-1__source-i-0__foo": 0.5,
        "samples/foo": 2,
        "samples/source-i-0__foo": 7,
        "input_tokens/foo": 100,
        "input_tokens/source-i-0__foo": 350,
        "label_tokens/foo": 20,
        "label_tokens/source-i-0__foo": 70,
        "label_tokens_per_sample/foo": 10,
        "label_tokens_per_sample/source-i-0__foo": 10,
    }
    data = ChannelLossDashboardData()

    assert data.record(trainer, 1, metrics)

    values = data.points[0]["values"]
    assert values["source-i-0"]["samples"] == 2
    assert values["source-i-0"]["input_tokens"] == 100
    assert values["source-i-1"]["samples"] == 7
    assert values["source-i-1"]["input_tokens"] == 350


def test_dashboard_resolves_three_way_short_and_qualified_name_collision():
    trainer = _trainer()
    trainer.channel_loss_callback._source_registry = {0: "foo", 1: "foo", 2: "source-i-0__foo"}
    metrics = {
        "channel_loss/source-i-0__foo": 2.0,
        "channel_loss/source-i-1__foo": 1.0,
        "channel_loss/source-i-2__source-i-0__foo": 0.5,
        "samples/source-i-0__foo": 2,
        "samples/source-i-1__foo": 3,
        "samples/source-i-2__source-i-0__foo": 7,
        "input_tokens/source-i-0__foo": 100,
        "input_tokens/source-i-1__foo": 150,
        "input_tokens/source-i-2__source-i-0__foo": 350,
        "label_tokens/source-i-0__foo": 20,
        "label_tokens/source-i-1__foo": 30,
        "label_tokens/source-i-2__source-i-0__foo": 70,
        "label_tokens_per_sample/source-i-0__foo": 10,
        "label_tokens_per_sample/source-i-1__foo": 10,
        "label_tokens_per_sample/source-i-2__source-i-0__foo": 10,
    }
    data = ChannelLossDashboardData()

    assert data.record(trainer, 1, metrics)

    values = data.points[0]["values"]
    assert values["source-i-0"]["samples"] == 2
    assert values["source-i-1"]["samples"] == 3
    assert values["source-i-2"]["samples"] == 7


def test_dashboard_ignores_steps_without_channel_loss_and_replaces_duplicate_step():
    trainer = _trainer()
    data = ChannelLossDashboardData()

    assert not data.record(trainer, 1, {"training/foundation_loss": 0.5})
    assert data.record(trainer, 2, _metrics(2))
    replacement = _metrics(2)
    replacement["channel_loss/source-i-0__repo_qa"] = 1.5
    assert data.record(trainer, 2, replacement)

    assert len(data.points) == 1
    assert data.points[0]["values"]["source-i-0"]["raw"] == 1.5


def test_dashboard_derives_overall_text_loss_from_weighted_channels_not_total_objective():
    trainer = _trainer()
    metrics = _metrics()
    metrics["training/foundation_loss"] = 7.0
    metrics["training/total_loss"] = 99.0
    data = ChannelLossDashboardData()

    data.record(trainer, 1, metrics)

    assert data.points[0]["overall"] == 0.7


def test_dashboard_falls_back_only_to_explicit_foundation_loss_without_complete_weighted_channels():
    trainer = _trainer()
    metrics = _metrics()
    del metrics["channel_loss_weighted/source-i-1__swecompass"]
    metrics["training/foundation_loss"] = 0.73
    metrics["training/total_loss"] = 99.0
    data = ChannelLossDashboardData()

    data.record(trainer, 1, metrics)

    assert data.points[0]["overall"] == 0.73


def test_dashboard_keeps_one_series_when_source_display_name_changes():
    trainer = _trainer()
    trainer.channel_loss_callback._source_registry[0] = None
    initial = {
        name.replace("source-i-0__repo_qa", "source-i-0__source_0"): value for name, value in _metrics().items()
    }
    data = ChannelLossDashboardData()
    data.record(trainer, 1, initial)
    trainer.channel_loss_callback._source_registry[0] = "repoqa"
    data.record(trainer, 2, _metrics(2))

    assert data.labels["source-i-0"] == "repoqa"
    assert all("source-i-0" in point["values"] for point in data.points)
    assert all("source-i-0__" not in key for point in data.points for key in point["values"])


def test_dashboard_html_is_self_contained_interactive_and_script_safe():
    trainer = _trainer()
    trainer.channel_loss_callback._source_registry[0] = "repo</script><script>alert(1)</script>"
    data = ChannelLossDashboardData()
    data.record(trainer, 1, _metrics())

    html = data.render_html()

    assert "Raw CE" in html
    assert "Weighted contribution" in html
    assert "Actual data mix" in html
    assert "Step sample summary" in html
    assert "Sample-level details remain in optional dataloader-owned trace artifacts" in html
    assert "dashboard renders aggregate metrics only" in html
    assert "repo</script>" not in html
    assert "repo\\u003c/script\\u003e" in html
    assert "https://" not in html
    assert "sampledMarkers(observations)" in html
    assert "observations.length<=200" not in html
    assert ".dashboard{width:100%;min-width:0" in html
    assert "@media(max-width:520px)" in html
    assert "min-width:min(150px,calc(100% - 16px))" in html
    assert "Math.max(8,Math.min(left" in html
    assert "min-width:620px" not in html
    assert "overflow-x:hidden" not in html


def test_dashboard_payload_decimates_but_keeps_endpoints():
    trainer = _trainer()
    data = ChannelLossDashboardData()
    for step in range(1, 11):
        data.record(trainer, step, _metrics(step))

    payload = data.payload(max_points=4)

    assert payload["sampled_points"] == 10
    assert payload["rendered_points"] == 4
    assert payload["points"][0]["step"] == 1
    assert payload["points"][-1]["step"] == 10
    assert payload["has_weighted"]
    assert payload["has_tokens"]
    assert payload["has_data_metrics"]


def test_dashboard_decimation_preserves_a_rare_dynamic_source():
    trainer = _trainer()
    data = ChannelLossDashboardData()
    for step in range(1, 11):
        metrics = _metrics(step)
        if step == 2:
            trainer.channel_loss_callback._source_registry[2] = "rare"
            metrics.update(
                {
                    "channel_loss/source-i-2__rare": 1.0,
                    "channel_loss_weighted/source-i-2__rare": 0.01,
                    "channel_tokens/source-i-2__rare": 1,
                }
            )
        data.record(trainer, step, metrics)

    payload = data.payload(max_points=3)

    assert any(source["key"] == "source-i-2" for source in payload["sources"])
    assert any("source-i-2" in point["values"] for point in payload["points"])


def test_dashboard_marks_optional_weighted_and_token_views_unavailable():
    trainer = _trainer()
    metrics = {
        name: value
        for name, value in _metrics().items()
        if not name.startswith(
            (
                "channel_loss_weighted/",
                "channel_tokens/",
                "samples/",
                "input_tokens/",
                "label_tokens/",
                "label_tokens_per_sample/",
            )
        )
    }
    data = ChannelLossDashboardData()
    data.record(trainer, 1, metrics)

    payload = data.payload(max_points=10)

    assert not payload["has_weighted"]
    assert not payload["has_tokens"]
    assert not payload["has_data_metrics"]
    assert '"has_weighted":false' in data.render_html()
    assert '"has_tokens":false' in data.render_html()


def test_dashboard_compacts_retained_history_without_losing_latest_point(monkeypatch):
    monkeypatch.setattr(dashboard_module, "_MAX_RETAINED_POINTS", 5)
    trainer = _trainer()
    data = ChannelLossDashboardData()
    for step in range(1, 8):
        metrics = _metrics(step)
        if step == 2:
            trainer.channel_loss_callback._source_registry[2] = "transient"
            metrics["channel_loss/source-i-2__transient"] = 1.0
        data.record(trainer, step, metrics)

    assert data.sampled_points == 7
    assert len(data.points) <= 5
    assert data.points[-1]["step"] == 7
    assert "source-i-2" in data.labels
    assert any("source-i-2" in point["values"] for point in data.points)


def test_dashboard_callback_is_disabled_without_channel_loss():
    callback = ChannelLossDashboardCallback(_trainer(enabled=False))

    assert not callback.enabled


def test_dashboard_callback_publishes_one_complete_html_at_train_end(monkeypatch):
    trainer = _trainer()
    trainer.step_env_metrics = _metrics()
    logged = []
    fake_wandb = SimpleNamespace(
        run=object(),
        Html=lambda content, inject: {"content": content, "inject": inject},
        log=lambda values, step: logged.append((values, step)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    callback = ChannelLossDashboardCallback(trainer)

    callback.on_step_end(SimpleNamespace(global_step=1))
    callback.on_step_end(SimpleNamespace(global_step=2))

    assert logged == []

    callback.on_train_end(SimpleNamespace(global_step=2))

    assert len(logged) == 1
    values, step = logged[0]
    assert step == 2
    assert values["channel_overview"]["inject"] is False
    assert "Channel loss · source overview" in values["channel_overview"]["content"]
    assert '"step":1' in values["channel_overview"]["content"]
    assert '"step":2' in values["channel_overview"]["content"]


def test_dashboard_callback_does_not_publish_incomplete_media_snapshots(monkeypatch):
    trainer = _trainer()
    logged_steps = []
    fake_wandb = SimpleNamespace(
        run=object(),
        Html=lambda content, inject: content,
        log=lambda values, step: logged_steps.append(step),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    callback = ChannelLossDashboardCallback(trainer)
    for step in range(1, 11):
        trainer.step_env_metrics = _metrics(step)
        callback.on_step_end(SimpleNamespace(global_step=step))

    assert logged_steps == []

    callback.on_train_end(SimpleNamespace(global_step=10))

    assert logged_steps == [10]


def test_dashboard_callback_flushes_latest_snapshot_once_at_interpreter_exit(monkeypatch):
    trainer = _trainer()
    trainer.step_env_metrics = _metrics()
    logged_steps = []
    registered = []
    monkeypatch.setattr(dashboard_module.atexit, "register", registered.append)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            run=object(),
            Html=lambda content, inject: content,
            log=lambda values, step: logged_steps.append(step),
        ),
    )
    callback = ChannelLossDashboardCallback(trainer)
    assert registered == []
    callback.on_step_end(SimpleNamespace(global_step=3))

    assert registered == [callback._publish_latest_at_exit]

    callback._publish_latest_at_exit()
    callback._publish_latest_at_exit()

    assert logged_steps == [3]


def test_dashboard_exit_flush_uses_latest_completed_step_after_last_sample(monkeypatch):
    trainer = _trainer()
    trainer.step_env_metrics = _metrics()
    logged_steps = []
    monkeypatch.setattr(dashboard_module.atexit, "register", lambda hook: None)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            run=object(),
            Html=lambda content, inject: content,
            log=lambda values, step: logged_steps.append(step),
        ),
    )
    callback = ChannelLossDashboardCallback(trainer)
    callback.on_step_end(SimpleNamespace(global_step=10))
    trainer.step_env_metrics = {"training/loss": 0.5}
    callback.on_step_end(SimpleNamespace(global_step=14))

    callback._publish_latest_at_exit()
    callback._publish_latest_at_exit()

    assert logged_steps == [14]


def test_dashboard_includes_falsey_optional_adapter_trace_step_without_sample_ids():
    trainer = _trainer()
    data = ChannelLossDashboardData()

    data.record(trainer, 7, _metrics(7), trace_step_id=0)

    point = data.payload(max_points=10)["points"][0]
    assert point["trace_step_id"] == "0"
    assert "global_sample_id" not in point


def test_dashboard_callback_consumes_trace_id_on_every_step_without_stale_reuse(monkeypatch):
    trainer = _trainer()
    monkeypatch.setattr(dashboard_module.atexit, "register", lambda hook: None)
    callback = ChannelLossDashboardCallback(trainer)
    trainer.set_channel_loss_trace_step_id("snapshot-abc:step-1")
    trainer.step_env_metrics = _metrics(1)
    callback.on_step_end(SimpleNamespace(global_step=1))

    trainer.set_channel_loss_trace_step_id("must-be-consumed")
    trainer.step_env_metrics = {"training/loss": 0.5}
    callback.on_step_end(SimpleNamespace(global_step=2))
    trainer.step_env_metrics = _metrics(3)
    callback.on_step_end(SimpleNamespace(global_step=3))

    assert callback.data.points[0]["trace_step_id"] == "snapshot-abc:step-1"
    assert callback.data.points[1]["trace_step_id"] is None
    assert trainer._channel_loss_trace_step_id is None


def test_base_trainer_trace_step_contract_is_one_shot_and_preserves_falsey_ids():
    trainer = object.__new__(BaseTrainer)
    trainer._channel_loss_trace_step_id = None

    trainer.set_channel_loss_trace_step_id(0)

    assert trainer.pop_channel_loss_trace_step_id() == 0
    assert trainer.pop_channel_loss_trace_step_id() is None


def test_composed_trainers_forward_the_public_trace_step_contract():
    for trainer_type in (TextTrainer, VLMTrainer, TextDPOTrainer):
        captured = []
        trainer = object.__new__(trainer_type)
        trainer.base = SimpleNamespace(set_channel_loss_trace_step_id=captured.append)

        trainer.set_channel_loss_trace_step_id("trace-step")

        assert captured == ["trace-step"]
