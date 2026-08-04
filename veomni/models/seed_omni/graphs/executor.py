"""Backward-compatible re-exports — VeOmni executors live in ``accelerator/``."""

from ..accelerator.executor import execute_generation_node, execute_train_node


__all__ = ["execute_train_node", "execute_generation_node"]
