# dit preprocess should not be used for llm or mllms
from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("video_generation_demo")
def video_generation_demo_preprocess(example, **kwargs):
    prompts = example["inputs"]
    inputs = prompts[0]  # Dict{str: List[str]}
    outputs = {}
    images = {}
    video_info = example["outputs"][0]
    videos = [video_info["video_bytes"].encode("latin-1")]
    return inputs, outputs, images, videos
