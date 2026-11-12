"""Utility functions for image preprocessing."""

import random

from PIL import Image


def center_crop(pil_image, crop_size):
    cw, ch = crop_size
    w, h = pil_image.size
    # Resize-to-cover first so we don't crop away content from larger inputs
    # (the previous version cropped the center region before resizing, which
    # zoomed/cut off images larger than the target crop size).
    scale = max(cw / w, ch / h)
    rw, rh = round(w * scale), round(h * scale)
    pil_image = pil_image.resize((rw, rh), Image.LANCZOS)
    left = max(0, (rw - cw) // 2)
    top = max(0, (rh - ch) // 2)
    return pil_image.crop((left, top, left + cw, top + ch))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list
