"""
Tests unitarios para feature_detection.py
"""
import pytest
import numpy as np
import cv2
from src.feature_detection import FeatureDetector


class TestFeatureDetector:
    """Tests para la clase FeatureDetector"""
    
    @pytest.fixture
    def sample_image_grayscale(self):
        """Crea una imagen de prueba en escala de grises"""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    @pytest.fixture
    def sample_image_color(self):
        """Crea una imagen de prueba a color"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_init_sift(self):
        """Test inicialización con detector SIFT"""
        detector = FeatureDetector(detector_type='SIFT')
        assert detector.detector_type == 'SIFT'
        assert detector.detector is not None
        assert isinstance(detector.detector, cv2.SIFT)
    
    def test_init_orb(self):
        """Test inicialización con detector ORB"""
        detector = FeatureDetector(detector_type='ORB')
        assert detector.detector_type == 'ORB'
        assert detector.detector is not None
        assert isinstance(detector.detector, cv2.ORB)
    
    def test_init_akaze(self):
        """Test inicialización con detector AKAZE"""
        detector = FeatureDetector(detector_type='AKAZE')
        assert detector.detector_type == 'AKAZE'
        assert detector.detector is not None
        assert isinstance(detector.detector, cv2.AKAZE)
    
    def test_init_invalid_detector(self):
        """Test que lanza error con detector inválido"""
        with pytest.raises(ValueError, match="Detector no soportado"):
            FeatureDetector(detector_type='INVALID')
    
    def test_detect_and_compute_grayscale(self, sample_image_grayscale):
        """Test detección en imagen escala de grises"""
        detector = FeatureDetector(detector_type='SIFT')
        keypoints, descriptors = detector.detect_and_compute(sample_image_grayscale)
        
        assert isinstance(keypoints, tuple) or isinstance(keypoints, list)
        assert descriptors is not None
        assert isinstance(descriptors, np.ndarray)
        assert len(keypoints) >= 0
    
    def test_detect_and_compute_color(self, sample_image_color):
        """Test detección en imagen a color"""
        detector = FeatureDetector(detector_type='SIFT')
        keypoints, descriptors = detector.detect_and_compute(sample_image_color)
        
        assert isinstance(keypoints, tuple) or isinstance(keypoints, list)
        assert descriptors is not None
        assert isinstance(descriptors, np.ndarray)
        assert len(keypoints) >= 0
    
    def test_detect_and_compute_different_detectors(self, sample_image_grayscale):
        """Test que diferentes detectores funcionan correctamente"""
        detectors = ['SIFT', 'ORB', 'AKAZE']
        
        for detector_type in detectors:
            detector = FeatureDetector(detector_type=detector_type)
            keypoints, descriptors = detector.detect_and_compute(sample_image_grayscale)
            
            assert keypoints is not None
            assert descriptors is not None
    
    def test_detect_and_compute_empty_image(self):
        """Test con imagen vacía"""
        detector = FeatureDetector(detector_type='SIFT')
        empty_image = np.zeros((10, 10), dtype=np.uint8)
        keypoints, descriptors = detector.detect_and_compute(empty_image)
        
        # Debería no fallar, aunque puede no detectar keypoints
        assert keypoints is not None
