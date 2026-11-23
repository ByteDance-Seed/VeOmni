"""Accelerated (training-graph) variant of the Qwen3-Omni audio tower."""

from .accelerated import Qwen3OmniAudioEncoderAccelerated, TrainingMixin, VeOmniMixin


__all__ = ["Qwen3OmniAudioEncoderAccelerated", "TrainingMixin", "VeOmniMixin"]
