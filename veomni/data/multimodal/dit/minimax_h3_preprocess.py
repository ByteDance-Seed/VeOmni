"""MiniMax H3 FL2VA data preprocessor.

CSV metadata format: prompt, video columns, optional input_audio column.
Keyframes (first/last frame) are extracted from the training video itself in
condition_model.get_condition() — not provided by the preprocessor.
"""

import os

from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("minimax_h3")
def minimax_h3_preprocess(conversations, **kwargs):
    """Parse CSV row → (prompt, audios_dict, images_list, videos_list).

    Args:
        conversations: dict with keys prompt, video, input_audio (optional)
        kwargs: must contain data_dir for relative path resolution

    Returns:
        (prompt, audios, images, videos)
    """
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
