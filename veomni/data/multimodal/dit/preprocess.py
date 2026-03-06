# dit preprocess should not be used for llm or mllms
from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("Tom-and-Jerry-VideoGeneration-Dataset")
def webvid_10M_preprocess(conversations, **kwargs):
    prompt = conversations["prompt"]
    outputs = {}
    images = {}
    videos = [conversations["video_bytes"]]
    return prompt, outputs, images, videos
