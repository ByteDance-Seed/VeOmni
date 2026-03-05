# dit preprocess should not be used for llm or mllms
from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("video_generation_demo")
def video_generation_demo_preprocess(conversations, **kwargs):
    prompts = conversations["inputs"]
    total_prompts = {}
    for prompt in prompts:
        total_prompts[prompt["type"]] = prompt[prompt["type"]]
    languages = ["en"]
    return_prompts = {}
    for lang in languages:
        prompt_image = total_prompts[f"image_caption_{lang}"]
        prompt_video = total_prompts[f"video_caption_{lang}"]
        return_prompts[f"prompt_{lang}"] = prompt_image + prompt_video
    inputs = return_prompts["prompt_en"]
    outputs = {}
    images = {}
    video_info = conversations["outputs"][0]
    videos = [video_info["video_bytes"].encode("latin-1")]
    return inputs, outputs, images, videos
