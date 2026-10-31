"""Compatibility imports for the shared official-Ultralytics OUR adapter."""
from app.infrastructure.inference.our import OurSegmentationPredictor, adapt_our_model

__all__ = ["OurSegmentationPredictor", "adapt_our_model"]
