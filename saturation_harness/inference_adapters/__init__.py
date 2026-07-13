"""Adapter implementations for each inference platform.

Each adapter conforms to ``base.InferenceAdapter`` so ``rate_sweep.py`` can
drive any platform with the same control flow.
"""
from .base import InferenceAdapter, RunResult, AdapterConfig  # noqa: F401
