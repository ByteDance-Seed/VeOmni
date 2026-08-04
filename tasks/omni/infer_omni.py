"""SeedOmni V2 inference via the VeOmni runtime (split checkpoint + generation FSM).

This is the **framework** inference path:

* ``model`` / per-module entries resolve to :class:`~veomni.omni_arguments.model_runtime.OmniModelRuntimeArguments`
  and :class:`~veomni.omni_arguments.arguments_types.OmniModuleRuntimeArguments`.
* When every module is ``fsdp_mode: eager``, :class:`~veomni.trainer.omni.omni_inferencer.OmniInferencer`
  loads a composed :class:`~veomni.models.seed_omni.modeling_omni.OmniModel` from the split checkpoint.
* When any module opts into FSDP2 / DDP / ExtraParallel, the handle becomes
  :class:`~veomni.models.seed_omni.accelerator.omni_model_runtime.OmniModelRuntime`.

For **native eager** inference on a split checkpoint (simple process + generate, no
VeOmni runtime / YAML launcher), use ``tasks/omni/infer_omni_native.py`` instead.

Examples
--------
Single-process eager (default — ``resolve_model(for_inference=True)`` forces eager
unless ``modules_infer_*.yaml`` overrides):

    python tasks/omni/infer_omni.py configs/seed_omni/Qwen/qwen3vl_2b/base.yaml \\
        --model.model_config.infer_type vision_understanding \\
        --infer.prompt "What is in this image?" \\
        --infer.image /path/to/image.jpg \\
        --infer.output_dir qwen3vl_out

Distributed inference (override modules to FSDP2 / DDP in ``modules_infer_fsdp.yaml``):

    bash train.sh tasks/omni/infer_omni.py \\
        configs/seed_omni/Janus/janus_1.3b/base.yaml \\
        --model.model_config.modules configs/seed_omni/Janus/janus_1.3b/modules_infer_fsdp.yaml \\
        --model.model_config.infer_type infer_gen \\
        --infer.prompt "A cat on a windowsill"
"""

from veomni.arguments import OmniArguments, parse_omni_args
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
