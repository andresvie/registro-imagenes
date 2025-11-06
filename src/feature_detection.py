import numpy as np
import cv2
from typing import Tuple

class FeatureDetector:
    """Clase para registro de imágenes basado en características"""
    
    def __init__(self, detector_type: str = 'SIFT'):
        """
        Inicializa el detector de características
        
        Args:
            detector_type: 'SIFT', 'ORB', o 'AKAZE'
        """
        self.detector_type = detector_type
        
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=2000)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError("Detector no soportado")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple:
        """Detecta keypoints y calcula descriptores"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors