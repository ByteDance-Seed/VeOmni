# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Make ``python -m patchgen`` an alias for the ``patchgen`` console script.

Standard Python convention. The submodules ``patchgen.run_codegen`` and
``patchgen.check_patchgen`` are library code only; they intentionally do
**not** support ``python -m patchgen.<sub>`` because the CLI in
:mod:`patchgen.cli` is the single supported entry point — see the
``__main__`` guards on those modules.
"""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
