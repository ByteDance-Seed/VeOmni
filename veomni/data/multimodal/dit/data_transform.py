import PIL

from ...data_transform import DATA_TRANSFORM_REGISTRY
from ..image_utils import fetch_images
from ..preprocess import conv_preprocess
from ..video_utils import fetch_videos


@DATA_TRANSFORM_REGISTRY.register("dit_online")
def process_dit_online_example(example, source_name, **kwargs):
    inputs, outputs, images, videos = conv_preprocess(source=source_name, conversations=example, **kwargs)
    if kwargs.get("use_audio_in_video", False):
        raise NotImplementedError("Audio in video is not supported yet for dit training.")
    videos, _ = fetch_videos(videos, use_audio_in_video=False, **kwargs)
    images = fetch_images(images, **kwargs)
    processed_example = {
        "inputs": inputs,
        "outputs": outputs,
        "images": images,
        "videos": videos,
    }
    return [processed_example]


@DATA_TRANSFORM_REGISTRY.register("minimax_h3_online")
def process_minimax_h3_online_example(example, source_name, **kwargs):
    """Raw data loading for MiniMax-H3 FL2VA.

    LoadVideo + ImageCropAndResize + LoadAudioWithTorchaudio steps:
      - video: imageio reader, fix_frame_rate=True (24fps), num_frames = min_frames
        (17n+5), bilinear resize + center crop to (height, width), frames → float32
        [0,1] tensors [3,H,W] (preprocess_video torch_dtype=float32, min_value=0)
      - audio: torchaudio.load, trim/pad to int(num_frames/24 * original_sr) at
        ORIGINAL sample rate, return (waveform[C,T], sample_rate)
    """
    import math

    import imageio
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF

    prompt, audios, _, videos = conv_preprocess(source=source_name, conversations=example, **kwargs)

    num_frames = int(kwargs.get("min_frames", 124))
    height = int(kwargs.get("height", 480))
    width = int(kwargs.get("width", 832))
    frame_rate = float(kwargs.get("fps", 24))

    frames = []
    n_frames = num_frames
    if videos and videos[0]:
        reader = imageio.get_reader(videos[0])
        meta = reader.get_meta_data()
        raw_fps = meta["fps"]
        total_raw_frames = int(reader.count_frames())
        duration = meta["duration"] if "duration" in meta else total_raw_frames / raw_fps
        available = math.floor(duration * frame_rate)
        if int(available) < num_frames:
            n_frames = int(available)
            while n_frames > 1 and n_frames % 17 != 5:
                n_frames -= 1
        for i in range(n_frames):
            raw_idx = min(int(round(i / frame_rate * raw_fps)), total_raw_frames - 1)
            img = PIL.Image.fromarray(reader.get_data(raw_idx))
            # ImageCropAndResize(height, width)
            w, h = img.size
            scale = max(width / w, height / h)
            img = TF.resize(
                img,
                (round(h * scale), round(w * scale)),
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            img = TF.center_crop(img, (height, width))
            frames.append(img)
        reader.close()
    # preprocess_video(torch_dtype=float32, min_value=0): [0,1] float32
    frames_t = [torch.tensor(np.array(f, dtype=np.float32)).permute(2, 0, 1) * (1.0 / 255.0) for f in frames]

    audio_out = None
    if audios and "audio" in audios and audios["audio"]:
        import torchaudio

        waveform, sample_rate = torchaudio.load(audios["audio"])
        target_samples = int((n_frames / frame_rate) * sample_rate)
        current_samples = waveform.shape[-1]
        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - current_samples))
        audio_out = (waveform, sample_rate)

    processed_example = {
        "inputs": prompt,
        "audios": audio_out,
        # FL2VA keyframes = first + last cropped frame, kept as native uint8 PIL
        # (rebuilding PIL from float tensors later would round-trip through
        # *255/uint8 and lose exactness)
        "images": [frames[0], frames[-1]] if frames else [],
        "videos": frames_t,
    }
    return [processed_example]


@DATA_TRANSFORM_REGISTRY.register("dit_offline")
def process_dit_offline_example(example, **kwargs):
    import pickle as pk

    processed_example = {key: pk.loads(value) for key, value in example.items()}
    return [processed_example]
