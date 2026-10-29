# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VeOmni's MiniMax M3 VL processor.

Registered in ``MODEL_PROCESSOR_REGISTRY`` under both the upstream class name
and the name the checkpoint's bundled ``processing_minimax.py`` uses, so a
MiniMax M3 run always gets this class regardless of which one ``AutoProcessor``
resolved from the checkpoint.
"""

from transformers import AutoTokenizer, PretrainedConfig
from transformers.models.minimax_m3_vl.processing_minimax_m3_vl import (
    MiniMaxM3VLProcessor as HfMiniMaxM3VLProcessor,
)

from ....utils import logging


logger = logging.get_logger(__name__)


def _require_minimax_m3_checkpoint(pretrained_model_name_or_path, kwargs) -> None:
    """Refuse to stand in for a processor whose checkpoint is not MiniMax M3 VL.

    ``MiniMaxVLProcessor`` is the generic class name in MiniMax's bundled remote
    code, so another MiniMax VL release could resolve to the same registry key.
    A directory without ``config.json`` (processor files only) is not checked.
    """
    hub_kwargs = {
        key: kwargs[key]
        for key in ("token", "revision", "cache_dir", "subfolder", "local_files_only")
        if key in kwargs
    }
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **hub_kwargs)
    except OSError:
        return
    model_type = config_dict.get("model_type")
    if model_type != "minimax_m3_vl":
        raise ValueError(
            f"VeOmni's MiniMax M3 VL processor only supports model_type 'minimax_m3_vl', but "
            f"{pretrained_model_name_or_path!r} is {model_type!r}. Register a processor for that model instead."
        )


class MiniMaxM3VLProcessor(HfMiniMaxM3VLProcessor):
    """MiniMax M3 VL processor built purely from the transformers-native classes.

    VeOmni pins MiniMax M3 to the transformers>=5.16 implementation -- that is
    what the generated modeling is written against -- and
    the checkpoint's bundled ``processing_minimax.py`` is unusable regardless: its
    ``MiniMaxVLProcessor.__init__`` drops ``**kwargs`` and so loses the
    ``chat_template.jinja`` that ``from_pretrained`` passes through, and its
    ``preprocessor_config.json`` declares the image/video processors *only* via a
    ``trust_remote_code`` ``auto_map``.

    So this class never runs the bundled code and never goes through
    ``AutoProcessor`` / ``ProcessorMixin.from_pretrained``. It builds the tokenizer
    and the native image/video processors by their concrete class -- each of which
    reads all of its parameters from the checkpoint's own config files -- and takes
    the chat template straight from the tokenizer, which is the one component that
    parses both ``chat_template.jinja`` and a ``tokenizer_config.json`` template.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # ``trust_remote_code`` only ever gated the Auto* sub-processor resolution
        # we deliberately bypass; the concrete native classes take no such flag.
        kwargs.pop("trust_remote_code", None)
        _require_minimax_m3_checkpoint(pretrained_model_name_or_path, kwargs)

        from transformers import MiniMaxM3VLImageProcessor, MiniMaxM3VLVideoProcessor

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False, **kwargs)
        image_processor = MiniMaxM3VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False, **kwargs
        )
        video_processor = MiniMaxM3VLVideoProcessor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False, **kwargs
        )

        # The tokenizer already parses the checkpoint's chat template from
        # whichever file carries it (``chat_template.jinja`` on the public
        # checkpoint, or a ``tokenizer_config.json`` entry), so it is the single
        # authoritative source. The M3 data transform renders conversations
        # through ``processor.apply_chat_template``, which needs it on the
        # processor.
        chat_template = tokenizer.chat_template
        if chat_template is None:
            logger.warning_rank0(
                "[PROCESSOR] MiniMax M3 VL checkpoint ships no chat template on its tokenizer; "
                "processor.apply_chat_template will refuse to run until one is provided."
            )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )
