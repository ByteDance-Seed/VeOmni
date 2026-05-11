from .comparison_utils import (
    TensorComparator,
    assert_close,
    assert_exact,
    compare_metrics,
    print_comparison_table,
)
from .data_generators import DummyDataset
from .hf_paths import hf_local_or_remote
from .launch_utils import find_free_port, torchrun
from .training_utils import (
    ParallelConfig,
    build_torchrun_cmd,
    make_eager_ops_config,
    make_npu_ops_config,
    materialize_weights,
    release_device_memory,
    resolve_ops_overrides,
    run_training_config,
)
