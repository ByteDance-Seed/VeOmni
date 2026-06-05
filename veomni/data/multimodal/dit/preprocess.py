# dit preprocess should not be used for llm or mllms
from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("Tom-and-Jerry-VideoGeneration-Dataset")
def tom_and_jerry_preprocess(conversations, **kwargs):
    prompt = conversations["prompt"]
    outputs = {}
    images = {}
    videos = [conversations["video_bytes"]]
    return prompt, outputs, images, videos


@PREPROCESSOR_REGISTRY.register("Qwen-Image")
@PREPROCESSOR_REGISTRY.register("QwenImage")
def qwen_image_preprocess(conversations, **kwargs):
    prompt = conversations.get("prompt") or conversations.get("text") or conversations.get("caption")
    image = (
        conversations.get("image")
        or conversations.get("image_bytes")
        or conversations.get("image_path")
        or conversations.get("target_image")
    )
    if prompt is None:
        raise ValueError("Qwen-Image data requires one of: prompt, text, caption.")
    if image is None:
        raise ValueError("Qwen-Image data requires one of: image, image_bytes, image_path, target_image.")
    return prompt, {}, [image], []
