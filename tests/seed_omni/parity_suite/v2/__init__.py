"""V2-side observation and runner helpers for the parity suite."""

from .model import ModuleNode, load_omni_config_from_dir, load_omni_module_from_pretrained, run_module_nodes
from .observation import (
    ModuleObservationSink,
    ObserverSink,
    arm_generation_observer,
    capture_forward_outputs,
    get_training_loss,
    get_training_node_output,
    record_module_output,
)


__all__ = [
    "ObserverSink",
    "ModuleObservationSink",
    "ModuleNode",
    "arm_generation_observer",
    "capture_forward_outputs",
    "get_training_loss",
    "get_training_node_output",
    "load_omni_config_from_dir",
    "load_omni_module_from_pretrained",
    "record_module_output",
    "run_module_nodes",
]
