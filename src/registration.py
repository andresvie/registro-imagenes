import numpy as np
import cv2
from typing import Tuple, Dict
from src.feature_detection import FeatureDetector
from src.matching import FeatureMatcher

class ImageRegistrator:
    def __init__(self, detector: FeatureDetector, matcher: FeatureMatcher):
        self.detector = detector
        self.matcher = matcher

    def register_images(self, img_ref: np.ndarray, img_mov: np.ndarray) -> Dict:
        """
        Registro completo de dos imágenes
        
        Returns:
            Diccionario con resultados del registro
        """
        # Detectar características
        kp1, desc1 = self.detector.detect_and_compute(img_ref)
        kp2, desc2 = self.detector.detect_and_compute(img_mov)
        
        # Emparejar características
        matches = self.matcher.match_features(desc1, desc2)
        
        # Estimar transformación
        M, mask = self.matcher.estimate_transform(kp1, kp2, matches)
        
        # Aplicar transformación
        h, w = img_ref.shape[:2]
        img_registered = None
        if M is not None:
            img_registered = cv2.warpPerspective(img_mov, M, (w, h))
        
        return {
            'keypoints_ref': kp1,
            'keypoints_mov': kp2,
            'matches': matches,
            'inliers_mask': mask,
            'transform_matrix': M,
            'registered_image': img_registered,
            'num_keypoints_ref': len(kp1),
            'num_keypoints_mov': len(kp2),
            'num_matches': len(matches),
            'num_inliers': np.sum(mask) if mask is not None else 0
        }


class RegistrationEvaluator:
    """Clase para evaluar la calidad del registro"""
    
    @staticmethod
    def decompose_affine_matrix(M: np.ndarray) -> Dict:
        """
        Descompone una matriz afín en sus componentes
        
        Returns:
            Dict con rotación, traslación y escala
        """
        if M.shape == (3, 3):
            M = M[:2, :]
        
        # Extraer componentes
        tx, ty = M[0, 2], M[1, 2]
        
        # Calcular escala y rotación
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        
        sx = np.sqrt(a**2 + c**2)
        sy = np.sqrt(b**2 + d**2)
        
        rotation = np.arctan2(c, a)
        rotation_deg = np.degrees(rotation)
        
        return {
            'translation_x': tx,
            'translation_y': ty,
            'scale_x': sx,
            'scale_y': sy,
            'rotation_rad': rotation,
            'rotation_deg': rotation_deg
        }
    
    @staticmethod
    def calculate_rmse(M_true: np.ndarray, M_est: np.ndarray, 
                      image_shape: Tuple) -> float:
        """
        Calcula RMSE entre transformaciones usando puntos de la imagen
        
        Args:
            M_true: Matriz de transformación verdadera
            M_est: Matriz de transformación estimada
            image_shape: (height, width) de la imagen
        """
        h, w = image_shape[:2]
        
        # Puntos de prueba en las esquinas y centro
        pts = np.float32([
            [0, 0], [w, 0], [w, h], [0, h], [w/2, h/2]
        ]).reshape(-1, 1, 2)
        
        # Transformar puntos
        if M_true.shape == (3, 3):
            pts_true = cv2.perspectiveTransform(pts, M_true)
            pts_est = cv2.perspectiveTransform(pts, M_est)
        else:
            M_true_full = np.vstack([M_true, [0, 0, 1]])
            M_est_full = np.vstack([M_est, [0, 0, 1]])
            pts_true = cv2.perspectiveTransform(pts, M_true_full)
            pts_est = cv2.perspectiveTransform(pts, M_est_full)
        
        # Calcular RMSE
        diff = pts_true - pts_est
        rmse = np.sqrt(np.mean(diff**2))
        
        return rmse
    
    @staticmethod
    def calculate_angular_error(angle_true: float, angle_est: float) -> float:
        """Calcula error angular en grados"""
        error = np.abs(angle_true - angle_est)
        # Normalizar al rango [-180, 180]
        error = (error + 180) % 360 - 180
        return np.abs(error)
    
    @staticmethod
    def calculate_translation_error(M_true: np.ndarray, M_est: np.ndarray) -> float:
        """Calcula error euclidiano en traslación"""
        true_params = RegistrationEvaluator.decompose_affine_matrix(M_true)
        est_params = RegistrationEvaluator.decompose_affine_matrix(M_est)
        
        dx = true_params['translation_x'] - est_params['translation_x']
        dy = true_params['translation_y'] - est_params['translation_y']
        
        return np.sqrt(dx**2 + dy**2)
    
    @staticmethod
    def calculate_scale_error(M_true: np.ndarray, M_est: np.ndarray) -> float:
        """Calcula error relativo en escala"""
        true_params = RegistrationEvaluator.decompose_affine_matrix(M_true)
        est_params = RegistrationEvaluator.decompose_affine_matrix(M_est)
        
        scale_true = (true_params['scale_x'] + true_params['scale_y']) / 2
        scale_est = (est_params['scale_x'] + est_params['scale_y']) / 2
        
        return np.abs(scale_true - scale_est) / scale_true * 100
    
    @staticmethod
    def evaluate_registration(M_true: np.ndarray, M_est: np.ndarray, 
                            image_shape: Tuple) -> Dict:
        """
        Evaluación completa del registro
        
        Returns:
            Dict con todas las métricas de error
        """
        true_params = RegistrationEvaluator.decompose_affine_matrix(M_true)
        est_params = RegistrationEvaluator.decompose_affine_matrix(M_est)
        
        rmse = RegistrationEvaluator.calculate_rmse(M_true, M_est, image_shape)
        angular_error = RegistrationEvaluator.calculate_angular_error(
            true_params['rotation_deg'], est_params['rotation_deg']
        )
        translation_error = RegistrationEvaluator.calculate_translation_error(M_true, M_est)
        scale_error = RegistrationEvaluator.calculate_scale_error(M_true, M_est)
        
        return {
            'rmse': rmse,
            'angular_error_deg': angular_error,
            'translation_error_px': translation_error,
            'scale_error_percent': scale_error,
            'true_rotation_deg': true_params['rotation_deg'],
            'estimated_rotation_deg': est_params['rotation_deg'],
            'true_translation_x': true_params['translation_x'],
            'estimated_translation_x': est_params['translation_x'],
            'true_translation_y': true_params['translation_y'],
            'estimated_translation_y': est_params['translation_y'],
            'true_scale': (true_params['scale_x'] + true_params['scale_y']) / 2,
            'estimated_scale': (est_params['scale_x'] + est_params['scale_y']) / 2
        }