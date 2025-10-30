"""
Módulo para detección de características en imágenes.
Implementa diferentes detectores de características (SIFT, ORB, AKAZE, etc.)
"""

import cv2
import numpy as np
from typing import Tuple, List


class FeatureDetector:
    """Clase para detectar características en imágenes."""
    
    def __init__(self, method: str = 'SIFT', n_features: int = 500):
        """
        Inicializa el detector de características.
        
        Args:
            method: Método de detección ('SIFT', 'ORB', 'AKAZE', 'SURF')
            n_features: Número máximo de características a detectar
        """
        self.method = method
        self.n_features = n_features
        self.detector = self._create_detector()
    
    def _create_detector(self):
        """Crea el detector según el método especificado."""
        if self.method == 'SIFT':
            return cv2.SIFT_create(nfeatures=self.n_features)
        elif self.method == 'ORB':
            return cv2.ORB_create(nfeatures=self.n_features)
        elif self.method == 'AKAZE':
            return cv2.AKAZE_create()        
        else:
            raise ValueError(f"Método {self.method} no soportado. Use 'SIFT', 'ORB' o 'AKAZE'")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detecta características (keypoints y descriptors) en una imagen.
        
        Args:
            image: Imagen en escala de grises o color
        
        Returns:
            keypoints: Lista de keypoints detectados
            descriptors: Array de descriptores (N x descriptor_size)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        return keypoints, descriptors
    
    def visualize_keypoints(self, image: np.ndarray, keypoints: List) -> np.ndarray:
        """
        Visualiza los keypoints detectados en la imagen.
        
        Args:
            image: Imagen original
            keypoints: Lista de keypoints a visualizar
    
        Returns:
            image_with_keypoints: Imagen con keypoints dibujados
        """
        if len(image.shape) == 3:
            display_image = image.copy()
        else:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Dibujar keypoints
        image_with_keypoints = cv2.drawKeypoints(
            display_image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return image_with_keypoints


def detect_features(image: np.ndarray, method: str = 'SIFT', n_features: int = 500) -> Tuple[List, np.ndarray]:
    """
    Función conveniente para detectar características.
    
    Args:
        image: Imagen en escala de grises o color
        method: Método de detección ('SIFT', 'ORB', 'AKAZE')
        n_features: Número máximo de características
    
    Returns:
        keypoints: Lista de keypoints detectados
        descriptors: Array de descriptores
    """
    detector = FeatureDetector(method=method, n_features=n_features)
    return detector.detect_and_compute(image)

