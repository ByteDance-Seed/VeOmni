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

import triton

from ....utils import logging
from ....utils.device import get_device_name
from . import envvars
from .config import load_all_configs_for
from .kernel import innermost_fn, qualified_name


logger = logging.get_logger(__name__)

CATCH_ALL_ALGO_KEY = "__CATCH_ALL__"


def algo_key_scaled(names, scales, rest_key=None):
    def key_maker(**kwargs):
        lower_names = [name.lower() for name in names]
        temp = []
        for i, name in enumerate(lower_names):
            t = name + str(kwargs[names[i]] // scales[i])
            if scales[i] != 1:
                t += f"x{scales[i]}"
            temp.append(t)
        res = "_".join(temp)

        if rest_key is not None:
            for k in rest_key:
                res += f"_{kwargs[k]}"
        return res

    return key_maker


class Pretuned(triton.KernelInterface):
    def __init__(self, fn, algo_key_maker, configs):
        self.fn = fn  # In case the outer decorator cares.
        self.kernel_name = qualified_name(fn)
        self.algo_key_maker = algo_key_maker
        self.configs = configs

        assert CATCH_ALL_ALGO_KEY in self.configs

    def run(self, *args, **kwargs):
        algo_key = self.algo_key_maker(**kwargs)
        if algo_key not in self.configs:
            if not envvars.is_untuned_warning_suppressed():
                logger.debug(
                    f"Untuned case (using algo-key [{algo_key}]) is seen when invoking "
                    f"kernel [{qualified_name(self)}], performance may suffer."
                )
            extra_kwargs = self.configs[CATCH_ALL_ALGO_KEY]
        else:
            extra_kwargs = self.configs[algo_key]
        return self.fn.run(*args, **kwargs, **extra_kwargs)


# TODO: Support using `triton.autotune` as an fallback.
def pretuned(*, algo_key=None, fallback=None):
    """Decorator to annotate a Triton kernel as pre-tuned. Hyperparameters are loaded from `PRETUNED`
    in the same folder as the kernel being defined.

    By default we look up pre-tuned hyperparameters via `kernel_name, device_name`. However, users
    are allowed to provide `algo_key` option by providing a lambda that converts arguments passed
    to kernel to a string that's used as a third level key in looking up pre-tuned hyperparameters.

    Note that ONLY named arguments (but not positional arguments) are passed to `algo_key` callback.
    """

    if algo_key is None:

        def catch_all(**kwargs):
            return CATCH_ALL_ALGO_KEY

        algo_key = catch_all

    def decorator(fn: triton.KernelInterface):
        nonlocal algo_key
        nonlocal fallback

        name = qualified_name(fn)
        configs = load_all_configs_for(innermost_fn(fn))

        if CATCH_ALL_ALGO_KEY not in configs:
            # We'd like to find a fallback hyperparameter for each `device`. This is not the same one
            # as `fallback` provided to `pretuned`. The latter is used when we're running on an untuned
            # device, while the former is just a catch-all for a specific device.
            if not envvars.is_untuned_warning_suppressed():
                logger.debug(
                    f"No pre-tuned hyperparameter for kernel [{name}], using fallback config, "
                    "performance may suffer. You may have triton version or device name mismatch. "
                    f"You have triton=={triton.__version__} and device name [{get_device_name()}]",
                )
            configs.update({CATCH_ALL_ALGO_KEY: fallback})

        assert configs[CATCH_ALL_ALGO_KEY] is not None, "No usable fallback hyperparameter for kernel {name}"
        return Pretuned(
            fn,
            algo_key_maker=algo_key,
            configs=configs,
        )

    return decorator
