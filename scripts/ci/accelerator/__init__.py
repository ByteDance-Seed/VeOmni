"""Shared accelerator CI tooling; importing the package does not initialize devices."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
