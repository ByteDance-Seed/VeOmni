"""Run GPU/NPU unit or e2e suites through the common scheduler."""

from scripts.ci.accelerator.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
