"""SeedOmni V2 inference via the VeOmni runtime (split checkpoint + generation FSM).

This is the **framework** inference path:

* ``model`` / per-module entries resolve to
  :class:`~veomni.arguments.omni_arguments_types.OmniModelRuntimeArguments`
  and :class:`~veomni.arguments.omni_arguments_types.OmniModuleRuntimeArguments`.
* When every module is ``fsdp_mode: eager``, :class:`~veomni.trainer.omni.omni_inferencer.OmniInferencer`
  loads a composed :class:`~veomni.models.seed_omni.modeling_omni.OmniModel` from the split checkpoint.
* When any module opts into FSDP2 / DDP / ExtraParallel, the handle becomes
  :class:`~veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime.OmniModelRuntime`.

For **native eager** inference on a split checkpoint (simple process + generate, no
VeOmni runtime / YAML launcher), use ``tasks/omni/infer_omni_native.py`` instead.

Examples
--------
Single-process eager (``resolve_model(for_inference=True)`` forces eager unless the
``modules:`` overlay pins another ``fsdp_mode``):

    python tasks/omni/infer_omni.py configs/seed_omni/fake_model/train/base.yaml \\
        --model.model_config.modules configs/seed_omni/fake_model/infer/modules_infer_eager.yaml \\
        --infer.prompt "hi"

Distributed inference (modules keep their DDP / FSDP2 wraps):

    bash train.sh tasks/omni/infer_omni.py configs/seed_omni/fake_model/train/base.yaml \\
        --model.model_config.modules configs/seed_omni/fake_model/infer/modules_infer_fsdp.yaml \\
        --infer.prompt "hi"
"""

from veomni.arguments.omni_arguments_types import OmniArguments
from veomni.arguments.omni_parser import parse_omni_args
from veomni.trainer.omni import OmniInferencer


def main() -> None:
    args = parse_omni_args(
        OmniArguments,
        preload_path_fields=("model.model_config.modules",),
    )
    inferencer = OmniInferencer(args)
    inferencer.generate()


if __name__ == "__main__":
    main()
