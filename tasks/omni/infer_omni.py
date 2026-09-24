"""SeedOmni inference via the VeOmni runtime (split checkpoint + generation FSM).

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

Media is decoded by the same fetchers the training transform uses, so a request
carries the same metadata a training sample does (a clip's sampling rate, a
video's frame timeline)::

    --infer.images /path/to/image.jpg           # an image
    --infer.audios /path/to/speech.wav          # standalone sound
    --infer.videos /path/to/clip.mp4            # a clip

``infer.mm_configs`` holds the decode knobs — the same free-form bag training
spells ``data.mm_configs``, reaching the same fetchers, so a key means the same
thing on either side. It is not inherited from the training config: state the
decode budget an inference run wants. Like ``infer.generation_kwargs``, set a
group of them in the launcher YAML and override one on the command line with a
dotted flag. A clip with sound is one entry in ``videos`` plus the flag, never
also an entry in ``audios``, since the two streams share a timeline the backbone
interleaves::

    infer:
      videos: [/path/to/clip.mp4]
      mm_configs:
        use_audio_in_video: true
        fps: 2.0

    --infer.mm_configs.fps 4.0                # overrides just that key

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
