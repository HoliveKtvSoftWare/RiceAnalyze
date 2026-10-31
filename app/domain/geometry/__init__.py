"""Geometry primitives used by inference and metrics."""

from .mask_geometry import (
    SIDE_BUNDLE_SWAP,
    extract_topology_contours,
    filter_small_components,
    reassign_side_bundle_labels,
    resize_binary_mask,
    topology_geometry,
)

__all__ = [
    "SIDE_BUNDLE_SWAP",
    "extract_topology_contours",
    "filter_small_components",
    "reassign_side_bundle_labels",
    "resize_binary_mask",
    "topology_geometry",
]
