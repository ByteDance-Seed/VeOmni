"""V2-side observation and runner helpers for the parity suite."""

from .model import (
    graph_active_module_names,
    load_graph_active_omni_config,
    load_graph_active_omni_model,
    load_graph_active_omni_modules,
    load_omni_config_from_dir,
    load_omni_module_from_pretrained,
)
from .observation import (
    LOSS_FIELD,
    ModuleObservationSink,
    ObserverSink,
    arm_generation_observer,
    capture_forward_outputs,
    get_training_loss,
    get_training_node_output,
    record_conversation_output,
    record_module_output,
)


__all__ = [
    "LOSS_FIELD",
    "ObserverSink",
    "ModuleObservationSink",
    "arm_generation_observer",
    "capture_forward_outputs",
    "get_training_loss",
    "get_training_node_output",
    "record_conversation_output",
    "graph_active_module_names",
    "load_graph_active_omni_config",
    "load_graph_active_omni_model",
    "load_graph_active_omni_modules",
    "load_omni_config_from_dir",
    "load_omni_module_from_pretrained",
    "record_module_output",
]
