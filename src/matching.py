"""
Módulo para emparejamiento de características entre imágenes.
Implementa estrategias de matching robustas y filtrado de outliers.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FeatureMatcher:
    """Clase para emparejar características entre imágenes."""
    
    def __init__(self, method: str = 'FLANN', ratio_threshold: float = 0.75):
        """
        Inicializa el matcher de características.
        
        Args:
            method: Método de matching ('FLANN', 'BF')
            ratio_threshold: Umbral para el ratio test de Lowe
        """
        self.method = method
        self.ratio_threshold = ratio_threshold
        self.matcher = self._create_matcher()
    
    def _create_matcher(self):
        """Crea el matcher según el método especificado."""
        if self.method == 'FLANN':
            # FLANN parameters para SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        elif self.method == 'BF':
            # Brute Force para descriptors binarios (ORB, AKAZE)
            return cv2.BFMatcher()
        else:
            raise ValueError(f"Método {self.method} no soportado. Use 'FLANN' o 'BF'")
    
    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List:
        """
        Empareja características entre dos imágenes.
        
        Args:
            descriptors1: Descriptores de la primera imagen
            descriptors2: Descriptores de la segunda imagen
    
        Returns:
            matches: Lista de matches encontrados (k=2 para ratio test)
        """
        if descriptors1 is None or len(descriptors1) == 0:
            return []
        if descriptors2 is None or len(descriptors2) == 0:
            return []
        
        try:
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error:
            # Si hay error, usar BFMatcher sin ratio test
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        return matches
    
    def filter_matches(self, matches: List) -> List:
        """
        Filtra matches usando el ratio test de Lowe.
        
        Args:
            matches: Lista de matches (lista de listas con k=2 matches por cada descriptor)
    
        Returns:
            good_matches: Matches filtrados
        """
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Ratio test de Lowe
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def get_matched_points(self, keypoints1: List, keypoints2: List, 
                          matches: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae puntos correspondientes de los matches.
        
        Args:
            keypoints1: Keypoints de la primera imagen
            keypoints2: Keypoints de la segunda imagen
            matches: Lista de matches filtrados
    
        Returns:
            src_points: Puntos fuente (N, 2)
            dst_points: Puntos destino (N, 2)
        """
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2)


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, 
                  method: str = 'FLANN', ratio_threshold: float = 0.75) -> List:
    """
    Función conveniente para emparejar características.
    
    Args:
        descriptors1: Descriptores de la primera imagen
        descriptors2: Descriptores de la segunda imagen
        method: Método de matching ('FLANN', 'BF')
        ratio_threshold: Umbral para el ratio test
    
    Returns:
        matches: Lista de matches filtrados
    """
    matcher = FeatureMatcher(method=method, ratio_threshold=ratio_threshold)
    all_matches = matcher.match(descriptors1, descriptors2)
    return matcher.filter_matches(all_matches)


def visualize_matches(image1: np.ndarray, image2: np.ndarray,
                     keypoints1: List, keypoints2: List,
                     matches: List, max_matches: int = 50) -> np.ndarray:
    """
    Visualiza los matches entre dos imágenes.
    
    Args:
        image1: Primera imagen
        image2: Segunda imagen
        keypoints1: Keypoints de la primera imagen
        keypoints2: Keypoints de la segunda imagen
        matches: Lista de matches
    
    Returns:
        matched_image: Imagen con matches visualizados
    """
    # Limitar número de matches para visualización
    matches_to_show = matches[:max_matches]
    
    matched_image = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches_to_show,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return matched_image

