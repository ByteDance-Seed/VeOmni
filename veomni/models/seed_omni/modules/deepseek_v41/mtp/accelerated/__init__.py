"""VeOmni accelerated ``deepseek_v41_mtp`` (training hooks + EP parallel plan + FLOPs meter)."""

from .accelerated import DeepseekV41MtpAccelerated, MeterMixin, TrainingMixin, VeOmniMixin


__all__ = ["DeepseekV41MtpAccelerated", "MeterMixin", "TrainingMixin", "VeOmniMixin"]
