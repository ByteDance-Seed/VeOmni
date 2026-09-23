# dit preprocess should not be used for llm or mllms
import os

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
    prompt = next(
        (conversations[key] for key in ("prompt", "text", "caption") if conversations.get(key) is not None), None
    )
    image = next(
        (
            conversations[key]
            for key in ("image", "image_bytes", "image_path", "target_image")
            if conversations.get(key) is not None
        ),
        None,
    )
    if prompt is None:
        raise ValueError("Qwen-Image data requires one of: prompt, text, caption.")
    if image is None:
        raise ValueError("Qwen-Image data requires one of: image, image_bytes, image_path, target_image.")
    if not isinstance(prompt, str) or not isinstance(image, (str, bytes)):
        raise ValueError("Qwen-Image requires a text prompt and an image path, URL, or bytes.")
    data_dir = kwargs.get("data_dir", "")
    if (
        data_dir
        and isinstance(image, str)
        and not os.path.isabs(image)
        and not image.startswith(("http://", "https://"))
    ):
        image = os.path.join(data_dir, image)
    return prompt, {}, [image], []


@PREPROCESSOR_REGISTRY.register("minimax_h3")
def minimax_h3_preprocess(conversations, **kwargs):
    data_dir = kwargs.get("data_dir", "")
    prompt = conversations["prompt"]

    # Video path
    video_path = conversations.get("video", "")
    if video_path and data_dir:
        video_path = os.path.join(data_dir, video_path)

    # Audio path (optional)
    audio_path = conversations.get("input_audio", "")
    if audio_path and data_dir:
        audio_path = os.path.join(data_dir, audio_path)

    audios = {"audio": audio_path} if audio_path else {}
    videos = [video_path] if video_path else []

    # FL2VA keyframes extracted from video frames in condition_model.get_condition()
    return prompt, audios, [], videos
