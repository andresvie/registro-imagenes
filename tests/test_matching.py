"""
Tests unitarios para matching.py
"""
import pytest
import numpy as np
import cv2
from src.matching import FeatureMatcher
from src.feature_detection import FeatureDetector


class TestFeatureMatcher:
    """Tests para la clase FeatureMatcher"""
    
    @pytest.fixture
    def sample_descriptors_sift(self):
        """Crea descriptores SIFT de prueba"""
        # Crear imagen y detectar características
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        detector = cv2.SIFT_create()
        kp, desc = detector.detectAndCompute(img, None)
        return desc
    
    @pytest.fixture
    def sample_descriptors_orb(self):
        """Crea descriptores ORB de prueba"""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        detector = cv2.ORB_create()
        kp, desc = detector.detectAndCompute(img, None)
        return desc
    
    @pytest.fixture
    def sample_keypoints(self):
        """Crea keypoints de prueba"""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        detector = cv2.SIFT_create()
        kp, _ = detector.detectAndCompute(img, None)
        return kp
    
    def test_init_default(self):
        """Test inicialización con valores por defecto"""
        matcher = FeatureMatcher()
        assert matcher.detector_type == 'ORB'
        assert matcher.ratio_threshold == 0.75
    
    def test_init_custom(self):
        """Test inicialización con valores personalizados"""
        matcher = FeatureMatcher(detector_type='SIFT', ratio_threshold=0.6)
        assert matcher.detector_type == 'SIFT'
        assert matcher.ratio_threshold == 0.6
    
    def test_match_features_sift(self, sample_descriptors_sift):
        """Test matching con descriptores SIFT"""
        matcher = FeatureMatcher(detector_type='SIFT', ratio_threshold=0.75)
        
        # Crear dos conjuntos de descriptores (mismo tipo)
        desc1 = sample_descriptors_sift
        desc2 = sample_descriptors_sift
        
        matches = matcher.match_features(desc1, desc2)
        
        assert isinstance(matches, list)
        assert len(matches) >= 0
    
    def test_match_features_orb(self, sample_descriptors_orb):
        """Test matching con descriptores ORB"""
        matcher = FeatureMatcher(detector_type='ORB', ratio_threshold=0.75)
        
        desc1 = sample_descriptors_orb
        desc2 = sample_descriptors_orb
        
        matches = matcher.match_features(desc1, desc2)
        
        assert isinstance(matches, list)
        assert len(matches) >= 0
    
    def test_match_features_empty(self):
        """Test matching con descriptores vacíos"""
        matcher = FeatureMatcher(detector_type='SIFT')
        
        desc1 = np.array([]).reshape(0, 128)
        desc2 = np.array([]).reshape(0, 128)
        
        matches = matcher.match_features(desc1, desc2)
        
        assert isinstance(matches, list)
        assert len(matches) == 0
    
    def test_match_features_different_ratio_thresholds(self, sample_descriptors_sift):
        """Test que diferentes ratio_threshold afectan el número de matches"""
        desc1 = sample_descriptors_sift
        desc2 = sample_descriptors_sift
        
        matcher_loose = FeatureMatcher(detector_type='SIFT', ratio_threshold=0.9)
        matcher_strict = FeatureMatcher(detector_type='SIFT', ratio_threshold=0.5)
        
        matches_loose = matcher_loose.match_features(desc1, desc2)
        matches_strict = matcher_strict.match_features(desc1, desc2)
        
        # Un threshold más alto debería dar más matches
        assert len(matches_loose) >= len(matches_strict)
    
    def test_estimate_transform_sufficient_matches(self, sample_keypoints):
        """Test estimación de transformación con suficientes matches"""
        matcher = FeatureMatcher(detector_type='SIFT')
        
        kp1 = sample_keypoints
        kp2 = sample_keypoints
        
        # Crear matches ficticios
        matches = []
        min_len = min(len(kp1), len(kp2))
        for i in range(min(4, min_len)):  # Al menos 4 matches
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = i
            matches.append(match)
        
        M, mask = matcher.estimate_transform(kp1, kp2, matches)
        
        if len(matches) >= 4:
            assert M is not None
            assert M.shape == (3, 3)
            assert mask is not None
    
    def test_estimate_transform_insufficient_matches(self, sample_keypoints):
        """Test estimación de transformación con pocos matches"""
        matcher = FeatureMatcher(detector_type='SIFT')
        
        kp1 = sample_keypoints
        kp2 = sample_keypoints
        
        # Crear menos de 4 matches
        matches = []
        for i in range(2):
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = i
            matches.append(match)
        
        M, mask = matcher.estimate_transform(kp1, kp2, matches)
        
        assert M is None
        assert mask is None
    
    def test_estimate_transform_ransac(self, sample_keypoints):
        """Test estimación con método RANSAC"""
        matcher = FeatureMatcher(detector_type='SIFT')
        
        kp1 = sample_keypoints
        kp2 = sample_keypoints
        
        matches = []
        min_len = min(len(kp1), len(kp2))
        for i in range(min(10, min_len)):
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = i
            matches.append(match)
        
        M, mask = matcher.estimate_transform(kp1, kp2, matches, method='RANSAC')
        
        if len(matches) >= 4:
            assert M is not None
            assert mask is not None
    
    def test_estimate_transform_lmeds(self, sample_keypoints):
        """Test estimación con método LMEDS"""
        matcher = FeatureMatcher(detector_type='SIFT')
        
        kp1 = sample_keypoints
        kp2 = sample_keypoints
        
        matches = []
        min_len = min(len(kp1), len(kp2))
        for i in range(min(10, min_len)):
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = i
            matches.append(match)
        
        M, mask = matcher.estimate_transform(kp1, kp2, matches, method='LMEDS')
        
        if len(matches) >= 4:
            assert M is not None
            assert mask is not None
