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

import json
import os
from typing import Callable, Optional

import torch
import torch.testing

from ....utils.device import get_torch_device, synchronize
from . import envvars
from . import logger as blog


_BENCHMARK_RESULT_FILE = "benchmark_results.txt"


def _benchmark_fn(f, repeats):
    warmup_repeats = 100

    if envvars.testing_is_ci_env():
        repeats = min(10, repeats)

    if envvars.benchmarking_minimal_run():
        # Mostly used together w/ Nsight Compute. Nishgt compute itself will run kernel multiple
        # times, so we don't bother repeat launching kernel here.
        repeats = 1
        warmup_repeats = 0

    start_event = [get_torch_device().Event(enable_timing=True) for _ in range(repeats)]
    end_event = [get_torch_device().Event(enable_timing=True) for _ in range(repeats)]
    for _ in range(warmup_repeats):
        f()

    if not envvars.benchmarking_minimal_run():
        # Tens of milliseconds, should be sufficient for CPU to catch up.
        torch.cuda._sleep(50_000_000)

    for i in range(repeats):
        start_event[i].record()
        f()
        end_event[i].record()
    synchronize()

    durations = sorted([start_event[i].elapsed_time(end_event[i]) for i in range(repeats)])
    if repeats >= 10:  # We only preserve 25% to 75% timings.
        durations = durations[int(len(durations) * 0.25) : int(len(durations) * 0.75)]

    elapsed = sum(durations) * 1e-3  # ms -> s
    return elapsed, len(durations) / elapsed


def _append_result_to_on_disk_file(result):
    current = []

    if os.path.exists(_BENCHMARK_RESULT_FILE):
        with open(_BENCHMARK_RESULT_FILE) as f:
            current = json.loads(f.read())

    current.append(result)

    with open(_BENCHMARK_RESULT_FILE, "w") as f:
        f.write(json.dumps(current, indent=4))


def _report_benchmark_result(
    name,
    iters_per_sec,
    elapsed_secs,
    measurement,
    measurement_unit,
    is_baseline,
    key_metric,
):
    if is_baseline:
        name = name + " [baseline]"  # ...

    msec_per_iter = 1000 / iters_per_sec
    blog.logging.info(
        f"{name}: used {elapsed_secs:.2f} seconds ({msec_per_iter:.2f} ms per iter), "
        f"{measurement:.2f} {measurement_unit}/s"
    )

    if envvars.benchmarking_write_report():
        _append_result_to_on_disk_file(
            {
                "name": name,
                "elapsed_secs": elapsed_secs,
                "measurement": measurement,
                "measurement_unit": measurement_unit,
                "msec_per_iter": msec_per_iter,
                "is_baseline": is_baseline,
                "key_metric": key_metric,
            }
        )


def benchmark_tflops(name, flops, run_func=None, baseline=None, key_metric=False, repeats=1000):
    assert run_func is not None or baseline is not None

    if baseline is not None and not envvars.benchmarking_no_baseline():
        elapsed, iters_per_sec = _benchmark_fn(baseline, repeats)
        _report_benchmark_result(
            name,
            iters_per_sec,
            elapsed,
            flops * iters_per_sec / 1e12,
            "TFlops",
            True,
            key_metric,
        )
    if run_func is not None:
        elapsed, iters_per_sec = _benchmark_fn(run_func, repeats)
        _report_benchmark_result(
            name,
            iters_per_sec,
            elapsed,
            flops * iters_per_sec / 1e12,
            "TFlops",
            False,
            key_metric,
        )


def benchmark_gibps(
    name: str,
    bytes: int,
    run_func: Optional[Callable] = None,
    baseline: Optional[Callable] = None,
    key_metric: bool = False,
    repeats: int = 100,
):
    assert run_func is not None or baseline is not None

    if baseline is not None and not envvars.benchmarking_no_baseline():
        elapsed, iters_per_sec = _benchmark_fn(baseline, repeats)
        _report_benchmark_result(
            name,
            iters_per_sec,
            elapsed,
            bytes * iters_per_sec / 2**30,
            "GiB",
            True,
            key_metric,
        )
    if run_func is not None:
        elapsed, iters_per_sec = _benchmark_fn(run_func, repeats)
        _report_benchmark_result(
            name,
            iters_per_sec,
            elapsed,
            bytes * iters_per_sec / 2**30,
            "GiB",
            False,
            key_metric,
        )
