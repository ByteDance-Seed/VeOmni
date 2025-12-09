from ..qwen2_vl_vision_model.processing_qwen2_vl_vision_model import Qwen2VLVisionModelProcessor


class Qwen25VLVisionModelProcessor(Qwen2VLVisionModelProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
