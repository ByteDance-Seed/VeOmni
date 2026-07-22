import veomni.trainer.callbacks.trace_callback as trace_callback


def test_reduce_training_metrics_coalesces_numeric_values(monkeypatch):
    calls = []

    def fake_all_reduce(value, group=None):
        calls.append((value, group))
        return [item + 1 for item in value]

    monkeypatch.setattr(trace_callback, "all_reduce", fake_all_reduce)

    reduced = trace_callback._reduce_training_metrics({"loss": 2.0, "grad_norm": 3.0}, group="fsdp")

    assert reduced == {"loss": 3.0, "grad_norm": 4.0}
    assert calls == [([2.0, 3.0], "fsdp")]


def test_reduce_training_metrics_falls_back_for_non_numeric_values(monkeypatch):
    calls = []

    def fake_all_reduce(value, group=None):
        calls.append((value, group))
        return value

    monkeypatch.setattr(trace_callback, "all_reduce", fake_all_reduce)

    metrics = {"loss": 2.0, "custom": object()}
    reduced = trace_callback._reduce_training_metrics(metrics, group="fsdp")

    assert reduced == metrics
    assert calls == [(metrics["loss"], "fsdp"), (metrics["custom"], "fsdp")]
