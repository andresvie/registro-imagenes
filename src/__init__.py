"""
Paquete src para registro de imágenes.
"""

__version__ = "0.1.0"

from .feature_detection import FeatureDetector, detect_features
from .matching import FeatureMatcher, match_features, visualize_matches
from .registration import ImageRegistrator, estimate_homography, warp_image
from .utils import (
    load_image, save_image, create_synthetic_image,
    calculate_registration_error, visualize_comparison
)

__all__ = [
    'FeatureDetector',
    'detect_features',
    'FeatureMatcher',
    'match_features',
    'visualize_matches',
    'ImageRegistrator',
    'estimate_homography',
    'warp_image',
    'load_image',
    'save_image',
    'create_synthetic_image',
    'calculate_registration_error',
    'visualize_comparison'
]

