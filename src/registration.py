"""
Módulo para registro de imágenes.
Implementa estimación de homografías y fusión de imágenes.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

# Imports locales
try:
    from .matching import FeatureMatcher
    from .feature_detection import FeatureDetector
except ImportError:
    # Si falla el import relativo, usar import absoluto
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.matching import FeatureMatcher
    from src.feature_detection import FeatureDetector


class ImageRegistrator:
    """Clase para registrar y fusionar imágenes."""
    
    def __init__(self, detector_method: str = 'SIFT', 
                 matcher_method: str = 'FLANN',
                 ratio_threshold: float = 0.75,
                 ransac_threshold: float = 5.0):
        """
        Inicializa el registrador de imágenes.
        
        Args:
            detector_method: Método de detección ('SIFT', 'ORB', 'AKAZE')
            matcher_method: Método de matching ('FLANN', 'BF')
            ratio_threshold: Umbral para ratio test de Lowe
            ransac_threshold: Umbral para RANSAC
        """
        self.detector = FeatureDetector(method=detector_method)
        self.matcher = FeatureMatcher(method=matcher_method, ratio_threshold=ratio_threshold)
        self.ransac_threshold = ransac_threshold
    
    def estimate_homography(self, src_points: np.ndarray, 
                           dst_points: np.ndarray,
                           ransac: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estima la homografía entre dos conjuntos de puntos.
        
        Args:
            src_points: Puntos fuente (N, 2)
            dst_points: Puntos destino (N, 2)
            ransac: Si usar RANSAC para filtrar outliers
    
        Returns:
            homography: Matriz de homografía (3, 3)
            mask: Máscara de inliers
        """
        if len(src_points) < 4:
            raise ValueError("Se necesitan al menos 4 puntos para estimar una homografía")
        
        if ransac:
            homography, mask = cv2.findHomography(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=0.99,
                maxIters=2000
            )
        else:
            homography, mask = cv2.findHomography(
                src_points,
                dst_points,
                method=0
            )
            if mask is None:
                mask = np.ones((len(src_points),), dtype=np.uint8)
        
        return homography, mask
    
    def register_pair(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Registra un par de imágenes.
        
        Args:
            image1: Primera imagen (referencia)
            image2: Segunda imagen (a registrar)
    
        Returns:
            homography: Matriz de homografía desde image2 a image1
            num_matches: Número de matches usados
        """
        # Detectar características
        kp1, desc1 = self.detector.detect_and_compute(image1)
        kp2, desc2 = self.detector.detect_and_compute(image2)
        
        if desc1 is None or len(desc1) == 0:
            raise ValueError("No se encontraron características en la primera imagen")
        if desc2 is None or len(desc2) == 0:
            raise ValueError("No se encontraron características en la segunda imagen")
        
        # Emparejar características
        all_matches = self.matcher.match(desc1, desc2)
        good_matches = self.matcher.filter_matches(all_matches)
        
        if len(good_matches) < 4:
            raise ValueError(f"Solo se encontraron {len(good_matches)} matches, se necesitan al menos 4")
        
        # Extraer puntos correspondientes
        src_pts, dst_pts = self.matcher.get_matched_points(kp1, kp2, good_matches)
        
        # Estimar homografía
        homography, mask = self.estimate_homography(src_pts, dst_pts, ransac=True)
        
        num_inliers = np.sum(mask) if mask is not None else len(good_matches)
        
        return homography, num_inliers


def estimate_homography(src_points: np.ndarray, dst_points: np.ndarray, 
                        ransac: bool = True, 
                        ransac_threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima la homografía entre dos conjuntos de puntos.
    
    Args:
        src_points: Puntos fuente (N, 2)
        dst_points: Puntos destino (N, 2)
        ransac: Si usar RANSAC para filtrar outliers
        ransac_threshold: Umbral para RANSAC
    
    Returns:
        homography: Matriz de homografía (3, 3)
        mask: Máscara de inliers
    """
    if len(src_points) < 4:
        raise ValueError("Se necesitan al menos 4 puntos para estimar una homografía")
    
    if ransac:
        homography, mask = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=0.99,
            maxIters=2000
        )
    else:
        homography, mask = cv2.findHomography(
            src_points,
            dst_points,
            method=0
        )
        if mask is None:
            mask = np.ones((len(src_points),), dtype=np.uint8)
    
    return homography, mask


def warp_image(image: np.ndarray, homography: np.ndarray, 
              output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Aplica transformación de homografía a una imagen.
    
    Args:
        image: Imagen a transformar
        homography: Matriz de homografía
        output_shape: Tamaño de la imagen de salida (height, width). 
                     Si es None, se calcula automáticamente
    
    Returns:
        warped_image: Imagen transformada
    """
    h, w = image.shape[:2]
    
    if output_shape is None:
        # Calcular el tamaño necesario
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
        
        transformed_corners = (homography @ corners_homogeneous.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
        
        min_x = int(np.floor(transformed_corners[:, 0].min()))
        max_x = int(np.ceil(transformed_corners[:, 0].max()))
        min_y = int(np.floor(transformed_corners[:, 1].min()))
        max_y = int(np.ceil(transformed_corners[:, 1].max()))
        
        output_w = max_x - min_x
        output_h = max_y - min_y
    else:
        output_h, output_w = output_shape
        min_x, min_y = 0, 0
    
    warped_image = cv2.warpPerspective(
        image,
        homography,
        (output_w, output_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    
    return warped_image

