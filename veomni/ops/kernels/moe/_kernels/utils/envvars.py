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

import os
from functools import lru_cache


@lru_cache
def is_env_option_enabled(opt: str) -> bool:
    return int(os.getenv(opt, "0"))


def is_assertion_enabled():
    return is_env_option_enabled("BPEX_DEBUG")


def is_untuned_warning_suppressed():
    return is_env_option_enabled("BPEX_NO_WARN_ON_UNTUNED_CASE") or testing_is_ci_env()


def debugging_fake_benchmark_result():
    return is_env_option_enabled("BPEX_DEBUGGING_FAKE_BENCHMARK_RESULT")


def debugging_is_verbose():
    return is_env_option_enabled("BPEX_DEBUGGING_VERBOSE")


def testing_is_ci_env():
    return is_env_option_enabled("BPEX_TESTING_IS_CI_ENV")


def testing_no_noncontiguous_tensors():
    return is_env_option_enabled("BPEX_TESTING_NO_NONCONTIGUOUS_TENSORS")


def benchmarking_minimal_run():
    return is_env_option_enabled("BPEX_BENCHMARKING_MINIMAL_RUN") or benchmarking_using_ncu()


def benchmarking_no_baseline():
    return is_env_option_enabled("BPEX_BENCHMARKING_NO_BASELINE") or benchmarking_using_ncu()


def benchmarking_using_ncu():
    return is_env_option_enabled("BPEX_BENCHMARKING_USE_NCU")


def benchmarking_write_report():
    return is_env_option_enabled("BPEX_BENCHMARKING_WRITE_REPORT")


def tuning_correctness_check_only():
    return is_env_option_enabled("BPEX_TUNING_CORRECTNESS_CHECK_ONLY")
