import os

from . import logging


logger = logging.get_logger(__name__)


ENV_DEFAULTS = {
    "MODELING_BACKEND": "veomni",
    "USE_LIGER_KERNEL": "1",
    "USE_GROUP_GEMM": "1",
}


def get_env(name: str):
    try:
        default = ENV_DEFAULTS[name]
    except KeyError:
        raise KeyError(f"Env var `{name}` not defined in ENV_DEFAULTS")

    return os.environ.get(name, default)


def format_envs() -> str:
    lines = []
    lines.append("\n========== Environment Variables ==========")

    for name in sorted(ENV_DEFAULTS):
        raw = os.environ.get(name)
        value = raw if raw is not None else ENV_DEFAULTS[name]
        source = "env" if raw is not None else "default"
        lines.append(f"{name}={value} (source={source})")

    lines.append("===========================================")
    return "\n".join(lines)
