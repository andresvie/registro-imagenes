"""
Tests unitarios para registration.py
"""
import pytest
import numpy as np
import cv2
from src.registration import ImageRegistrator, RegistrationEvaluator
from src.feature_detection import FeatureDetector
from src.matching import FeatureMatcher


class TestImageRegistrator:
    """Tests para la clase ImageRegistrator"""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba"""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    @pytest.fixture
    def transformed_image(self, sample_image):
        """Crea una imagen transformada"""
        M = np.float32([[1, 0, 10], [0, 1, 5]])
        return cv2.warpAffine(sample_image, M, (100, 100))
    
    @pytest.fixture
    def registrator(self):
        """Crea un registrador de prueba"""
        detector = FeatureDetector(detector_type='SIFT')
        matcher = FeatureMatcher(detector_type='SIFT', ratio_threshold=0.75)
        return ImageRegistrator(detector, matcher)
    
    def test_init(self):
        """Test inicialización del registrador"""
        detector = FeatureDetector(detector_type='SIFT')
        matcher = FeatureMatcher(detector_type='SIFT')
        registrator = ImageRegistrator(detector, matcher)
        
        assert registrator.detector is not None
        assert registrator.matcher is not None
    
    def test_register_images_same_image(self, sample_image, registrator):
        """Test registro de la misma imagen"""
        result = registrator.register_images(sample_image, sample_image)
        
        assert isinstance(result, dict)
        assert 'keypoints_ref' in result
        assert 'keypoints_mov' in result
        assert 'matches' in result
        assert 'transform_matrix' in result
        assert 'num_matches' in result
        assert 'num_inliers' in result
    
    def test_register_images_transformed(self, sample_image, transformed_image, registrator):
        """Test registro con imagen transformada"""
        result = registrator.register_images(sample_image, transformed_image)
        
        assert isinstance(result, dict)
        assert result['num_keypoints_ref'] >= 0
        assert result['num_keypoints_mov'] >= 0
        assert result['num_matches'] >= 0
        assert result['num_inliers'] >= 0
    
    def test_register_images_all_fields(self, sample_image, registrator):
        """Test que todos los campos están presentes en el resultado"""
        result = registrator.register_images(sample_image, sample_image)
        
        required_fields = [
            'keypoints_ref', 'keypoints_mov', 'matches',
            'inliers_mask', 'transform_matrix', 'registered_image',
            'num_keypoints_ref', 'num_keypoints_mov',
            'num_matches', 'num_inliers'
        ]
        
        for field in required_fields:
            assert field in result
    
    def test_register_images_empty_images(self, registrator):
        """Test registro con imágenes vacías"""
        empty1 = np.zeros((10, 10), dtype=np.uint8)
        empty2 = np.zeros((10, 10), dtype=np.uint8)
        
        result = registrator.register_images(empty1, empty2)
        
        assert isinstance(result, dict)
        assert result['num_keypoints_ref'] >= 0
        assert result['num_keypoints_mov'] >= 0


class TestRegistrationEvaluator:
    """Tests para la clase RegistrationEvaluator"""
    
    @pytest.fixture
    def identity_matrix(self):
        """Matriz identidad 3x3"""
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    
    @pytest.fixture
    def rotation_matrix(self):
        """Matriz de rotación de 45 grados"""
        angle = np.radians(45)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    
    @pytest.fixture
    def translation_matrix(self):
        """Matriz de traslación"""
        return np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float32)
    
    def test_decompose_affine_matrix_identity(self, identity_matrix):
        """Test descomposición de matriz identidad"""
        params = RegistrationEvaluator.decompose_affine_matrix(identity_matrix)
        
        assert isinstance(params, dict)
        assert 'translation_x' in params
        assert 'translation_y' in params
        assert 'scale_x' in params
        assert 'scale_y' in params
        assert 'rotation_rad' in params
        assert 'rotation_deg' in params
        
        assert abs(params['translation_x']) < 1e-6
        assert abs(params['translation_y']) < 1e-6
        assert abs(params['scale_x'] - 1.0) < 1e-6
        assert abs(params['scale_y'] - 1.0) < 1e-6
    
    def test_decompose_affine_matrix_rotation(self, rotation_matrix):
        """Test descomposición de matriz de rotación"""
        params = RegistrationEvaluator.decompose_affine_matrix(rotation_matrix)
        
        assert abs(params['rotation_deg'] - 45.0) < 1.0  # Permite pequeña tolerancia
        assert abs(params['translation_x']) < 1e-6
        assert abs(params['translation_y']) < 1e-6
    
    def test_decompose_affine_matrix_translation(self, translation_matrix):
        """Test descomposición de matriz de traslación"""
        params = RegistrationEvaluator.decompose_affine_matrix(translation_matrix)
        
        assert abs(params['translation_x'] - 10.0) < 1e-6
        assert abs(params['translation_y'] - 20.0) < 1e-6
    
    def test_decompose_affine_matrix_2x3(self):
        """Test descomposición de matriz 2x3 (afín)"""
        M_2x3 = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float32)
        params = RegistrationEvaluator.decompose_affine_matrix(M_2x3)
        
        assert isinstance(params, dict)
        assert abs(params['translation_x'] - 10.0) < 1e-6
        assert abs(params['translation_y'] - 20.0) < 1e-6
    
    def test_calculate_rmse_identical_matrices(self, identity_matrix):
        """Test RMSE con matrices idénticas"""
        image_shape = (100, 100)
        rmse = RegistrationEvaluator.calculate_rmse(
            identity_matrix, identity_matrix, image_shape
        )
        
        assert isinstance(rmse, (float, np.floating))
        assert rmse >= 0
        assert rmse < 1.0  # Debería ser muy pequeño
    
    def test_calculate_rmse_different_matrices(self, identity_matrix, translation_matrix):
        """Test RMSE con matrices diferentes"""
        image_shape = (100, 100)
        rmse = RegistrationEvaluator.calculate_rmse(
            identity_matrix, translation_matrix, image_shape
        )
        
        assert isinstance(rmse, (float, np.floating))
        assert rmse >= 0
    
    def test_calculate_angular_error_same_angle(self):
        """Test error angular con mismo ángulo"""
        error = RegistrationEvaluator.calculate_angular_error(45.0, 45.0)
        
        assert error == 0.0
    
    def test_calculate_angular_error_different_angle(self):
        """Test error angular con ángulos diferentes"""
        error = RegistrationEvaluator.calculate_angular_error(0.0, 90.0)
        
        assert error == 90.0
    
    def test_calculate_angular_error_wrap_around(self):
        """Test error angular que maneja el wrap-around correctamente"""
        error1 = RegistrationEvaluator.calculate_angular_error(350.0, 10.0)
        error2 = RegistrationEvaluator.calculate_angular_error(10.0, 350.0)
        
        # Ambos deberían dar aproximadamente 20 grados
        assert error1 < 25.0
        assert error2 < 25.0
    
    def test_calculate_translation_error_identical(self, translation_matrix):
        """Test error de traslación con matrices idénticas"""
        error = RegistrationEvaluator.calculate_translation_error(
            translation_matrix, translation_matrix
        )
        
        assert error == 0.0
    
    def test_calculate_translation_error_different(self, identity_matrix, translation_matrix):
        """Test error de traslación con matrices diferentes"""
        error = RegistrationEvaluator.calculate_translation_error(
            identity_matrix, translation_matrix
        )
        
        assert error > 0
        assert abs(error - np.sqrt(10**2 + 20**2)) < 1.0
    
    def test_calculate_scale_error_identical(self, identity_matrix):
        """Test error de escala con matrices idénticas"""
        error = RegistrationEvaluator.calculate_scale_error(
            identity_matrix, identity_matrix
        )
        
        assert error == 0.0
    
    def test_calculate_scale_error_different(self):
        """Test error de escala con escalas diferentes"""
        M1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        M2 = np.array([[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1]], dtype=np.float32)
        
        error = RegistrationEvaluator.calculate_scale_error(M1, M2)
        
        assert error > 0
        assert abs(error - 10.0) < 2.0  # Aproximadamente 10%
    
    def test_evaluate_registration_full(self, identity_matrix, translation_matrix):
        """Test evaluación completa del registro"""
        image_shape = (100, 100)
        errors = RegistrationEvaluator.evaluate_registration(
            identity_matrix, translation_matrix, image_shape
        )
        
        assert isinstance(errors, dict)
        assert 'rmse' in errors
        assert 'angular_error_deg' in errors
        assert 'translation_error_px' in errors
        assert 'scale_error_percent' in errors
        assert 'true_rotation_deg' in errors
        assert 'estimated_rotation_deg' in errors
        assert 'true_translation_x' in errors
        assert 'estimated_translation_x' in errors
        assert 'true_translation_y' in errors
        assert 'estimated_translation_y' in errors
        assert 'true_scale' in errors
        assert 'estimated_scale' in errors
        
        assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in errors.values())
