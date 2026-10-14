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
        try:
            meta = reader.get_meta_data()
            raw_fps = meta["fps"]
            if "duration" in meta:
                available = math.floor(meta["duration"] * frame_rate)
            else:
                # No duration in meta: fall back to a full frame-count scan.
                total_raw_frames = int(reader.count_frames())
                available = math.floor((total_raw_frames / raw_fps) * frame_rate)
            if int(available) < num_frames:
                n_frames = int(available)
                if n_frames < 5:
                    raise ValueError(
                        f"Video clip is too short: {n_frames} frames available at "
                        f"{frame_rate:g}fps, but MiniMax-H3 needs at least 5 frames "
                        "(frame count must satisfy (N-5) % 17 == 0)."
                    )
                while n_frames > 1 and n_frames % 17 != 5:
                    n_frames -= 1
            # Single sequential pass over the stream. raw_idx only moves
            # forward, so selected indices are consumed in order; repeated
            # indices (round collisions) re-append the current frame, and
            # indices past the end of the stream clamp to the last frame —
            # exactly the frames the previous get_data(raw_idx) loop selected.
            desired = [int(round(i / frame_rate * raw_fps)) for i in range(n_frames)]
            ptr = 0
            for j, raw_frame in enumerate(reader.iter_data()):
                while ptr < n_frames and desired[ptr] <= j:
                    img = PIL.Image.fromarray(raw_frame)
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
                    ptr += 1
                last_raw_frame = raw_frame
            # Indices beyond the stream end clamp to the last decoded frame.
            while frames and ptr < n_frames:
                img = PIL.Image.fromarray(last_raw_frame)
                w, h = img.size
                scale = max(width / w, height / h)
                img = TF.resize(
                    img,
                    (round(h * scale), round(w * scale)),
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                img = TF.center_crop(img, (height, width))
                frames.append(img)
                ptr += 1
        finally:
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

    # FL2VA keyframes selected by keyframe_indices (config, default first +
    # last), kept as native uint8 PIL (rebuilding PIL from float tensors later
    # would round-trip through *255/uint8 and lose exactness). Source indices
    # travel with the sample so the condition model can validate them.
    keyframe_indices = list(kwargs.get("keyframe_indices", [0, -1]))
    keyframe_images = []
    for idx in keyframe_indices:
        if not -len(frames) <= idx < len(frames):
            raise ValueError(f"keyframe index {idx} out of range for {len(frames)} frames")
        keyframe_images.append(frames[idx])

    processed_example = {
        "inputs": prompt,
        "audios": audio_out,
        "images": keyframe_images,
        "keyframe_indices": keyframe_indices,
        "videos": frames_t,
    }
    return [processed_example]


@DATA_TRANSFORM_REGISTRY.register("dit_offline")
def process_dit_offline_example(example, **kwargs):
    import pickle as pk

    processed_example = {key: pk.loads(value) for key, value in example.items()}
    return [processed_example]
