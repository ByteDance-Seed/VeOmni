"""V2-side observation and runner helpers for the parity suite."""

from .graph import run_v2_infer_graph, run_v2_train_graph, run_v2_train_graph_batch
from .model import load_omni_config_from_dir, load_omni_module_from_pretrained
from .module import (
    InferModulePolicy,
    ModuleNode,
    run_infer_module_fsm,
    run_module_nodes,
    run_v2_infer_module,
    run_v2_train_module,
    run_v2_train_module_batch,
    run_v2_train_nodes,
)
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
    "InferModulePolicy",
    "ModuleObservationSink",
    "ModuleNode",
    "arm_generation_observer",
    "capture_forward_outputs",
    "get_training_loss",
    "get_training_node_output",
    "load_omni_config_from_dir",
    "load_omni_module_from_pretrained",
    "record_module_output",
    "run_infer_module_fsm",
    "run_module_nodes",
    "run_v2_infer_graph",
    "run_v2_infer_module",
    "run_v2_train_graph",
    "run_v2_train_graph_batch",
    "run_v2_train_module",
    "run_v2_train_module_batch",
    "run_v2_train_nodes",
]
