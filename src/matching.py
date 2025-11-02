"""
Módulo para emparejamiento de características entre imágenes.
Implementa estrategias de matching robustas y filtrado de outliers.
"""

import cv2
import numpy as np
from typing import List, Tuple


class FeatureMatcher:
    """Clase para emparejar características entre imágenes."""
    
    def __init__(self, detector_type: str = 'ORB', ratio_threshold: float = 0.75):
        """
        Inicializa el matcher de características.
        
        Args:
            detector_type: Tipo de detector ('SIFT', 'ORB', 'AKAZE')
            ratio_threshold: Umbral para el ratio test de Lowe
        """
        self.detector_type = detector_type
        self.ratio_threshold = ratio_threshold
        
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Empareja características usando Lowe's ratio test
        
        Args:
            desc1, desc2: Descriptores de las imágenes
        """
        if self.detector_type == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches

    def estimate_transform(self, kp1: List, kp2: List, matches: List,
                          method: str = 'RANSAC') -> Tuple[np.ndarray, np.ndarray]:
        """
        Estima la transformación entre dos conjuntos de puntos
        
        Args:
            kp1, kp2: Keypoints de las imágenes
            matches: Lista de matches
            method: 'RANSAC' o 'LMEDS'
            
        Returns:
            matriz_transformacion, mascara_inliers
        """
        if len(matches) < 4:
            return None, None
        
        # Extraer coordenadas de los matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimar transformación
        if method == 'RANSAC':
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
        
        return M, mask

