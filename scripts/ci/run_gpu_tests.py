"""GPU CI entry point backed by the common accelerator case scheduler."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.run_accelerator_tests import main


if __name__ == "__main__":
    if "--backend" not in sys.argv and not any(arg.startswith("--backend=") for arg in sys.argv[1:]):
        sys.argv.extend(["--backend", "cuda"])
    raise SystemExit(main())
