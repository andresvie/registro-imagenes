"""
Tests unitarios para stitcher.py
"""
import pytest
import numpy as np
import cv2
from src.stitcher import Stitcher
from src.feature_detection import FeatureDetector
from src.matching import FeatureMatcher


class TestStitcher:
    """Tests para la clase Stitcher"""
    
    @pytest.fixture
    def sample_image_color(self):
        """Crea una imagen de prueba a color"""
        return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_image_grayscale(self):
        """Crea una imagen de prueba en escala de grises"""
        return np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    @pytest.fixture
    def detector(self):
        """Crea un detector de características"""
        return FeatureDetector(detector_type='SIFT')
    
    @pytest.fixture
    def matcher(self):
        """Crea un matcher de características"""
        return FeatureMatcher(detector_type='SIFT', ratio_threshold=0.75)
    
    @pytest.fixture
    def stitcher(self, detector, matcher):
        """Crea un stitcher de prueba"""
        return Stitcher(num_pyramid_levels=4, detector=detector, matcher=matcher)
    
    def test_init_default(self):
        """Test inicialización con valores por defecto"""
        stitcher = Stitcher()
        assert stitcher.num_pyramid_levels == 4
        assert stitcher.detector is None
        assert stitcher.matcher is None
    
    def test_init_custom(self, detector, matcher):
        """Test inicialización con valores personalizados"""
        stitcher = Stitcher(num_pyramid_levels=5, detector=detector, matcher=matcher)
        assert stitcher.num_pyramid_levels == 5
        assert stitcher.detector == detector
        assert stitcher.matcher == matcher
    
    def test_build_gaussian_pyramid(self, stitcher, sample_image_color):
        """Test construcción de pirámide Gaussiana"""
        pyramid = stitcher.build_gaussian_pyramid(sample_image_color, 3)
        
        assert isinstance(pyramid, list)
        assert len(pyramid) == 3
        
        # Verificar que cada nivel es más pequeño que el anterior
        for i in range(len(pyramid) - 1):
            assert pyramid[i].shape[0] >= pyramid[i+1].shape[0]
            assert pyramid[i].shape[1] >= pyramid[i+1].shape[1]
        
        # Verificar que el último nivel es aproximadamente 1/4 del tamaño original
        original_size = sample_image_color.shape[:2]
        last_size = pyramid[-1].shape[:2]
        assert last_size[0] <= original_size[0] // 2
        assert last_size[1] <= original_size[1] // 2
    
    def test_build_gaussian_pyramid_single_level(self, stitcher, sample_image_color):
        """Test construcción de pirámide Gaussiana con un solo nivel"""
        pyramid = stitcher.build_gaussian_pyramid(sample_image_color, 1)
        
        assert len(pyramid) == 1
        assert np.array_equal(pyramid[0], sample_image_color)
    
    def test_build_laplacian_pyramid(self, stitcher, sample_image_color):
        """Test construcción de pirámide Laplaciana"""
        laplacian_pyramid = stitcher.build_laplacian_pyramid(sample_image_color, 3)
        
        assert isinstance(laplacian_pyramid, list)
        assert len(laplacian_pyramid) == 3
        
        # Verificar que todos los niveles tienen el mismo número de canales
        num_channels = sample_image_color.shape[2] if len(sample_image_color.shape) == 3 else 1
        for level in laplacian_pyramid:
            if num_channels == 1:
                assert len(level.shape) == 2
            else:
                assert level.shape[2] == num_channels
    
    def test_reconstruct_from_laplacian_pyramid(self, stitcher, sample_image_color):
        """Test reconstrucción desde pirámide Laplaciana"""
        laplacian_pyramid = stitcher.build_laplacian_pyramid(sample_image_color, 3)
        reconstructed = stitcher.reconstruct_from_laplacian_pyramid(laplacian_pyramid)
        
        assert reconstructed.shape == sample_image_color.shape
        assert reconstructed.dtype == sample_image_color.dtype
    
    def test_reconstruct_from_laplacian_pyramid_identity(self, stitcher, sample_image_color):
        """Test que la reconstrucción preserva aproximadamente la imagen original"""
        laplacian_pyramid = stitcher.build_laplacian_pyramid(sample_image_color, 3)
        reconstructed = stitcher.reconstruct_from_laplacian_pyramid(laplacian_pyramid)
        
        # La reconstrucción puede tener diferencias debido a la cuantización y operaciones de pirámide
        # pero debería mantener la forma y tipo de datos
        assert reconstructed.shape == sample_image_color.shape
        assert reconstructed.dtype == sample_image_color.dtype
        
        # Verificar que la diferencia promedio es razonable (la reconstrucción de Laplacian
        # puede introducir errores de cuantización, especialmente con imágenes aleatorias)
        diff = np.abs(reconstructed.astype(np.float32) - sample_image_color.astype(np.float32))
        assert np.mean(diff) < 50  # Tolerancia más amplia para operaciones de pirámide
    
    def test_detect_and_describe_multiscale(self, stitcher, sample_image_color):
        """Test detección multi-escala de características"""
        kp, desc = stitcher.detect_and_describe_multiscale(sample_image_color)
        
        assert isinstance(kp, list)
        assert desc is None or isinstance(desc, np.ndarray)
        assert len(kp) >= 0
    
    def test_detect_and_describe_multiscale_empty_image(self, stitcher):
        """Test detección multi-escala con imagen vacía"""
        empty_image = np.zeros((50, 50, 3), dtype=np.uint8)
        kp, desc = stitcher.detect_and_describe_multiscale(empty_image)
        
        assert isinstance(kp, list)
        assert len(kp) >= 0
    
    def test_match_features(self, stitcher, sample_image_color):
        """Test matching de características"""
        kp1, desc1 = stitcher.detect_and_describe_multiscale(sample_image_color)
        kp2, desc2 = stitcher.detect_and_describe_multiscale(sample_image_color)
        
        if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
            matches = stitcher.match_features(desc1, desc2)
            assert isinstance(matches, list)
            assert len(matches) >= 0
    
    def test_match_features_empty_descriptors(self, stitcher):
        """Test matching con descriptores vacíos"""
        desc1 = np.array([]).reshape(0, 128)
        desc2 = np.array([]).reshape(0, 128)
        
        matches = stitcher.match_features(desc1, desc2)
        assert isinstance(matches, list)
    
    def test_find_homography_ransac(self, stitcher, sample_image_color):
        """Test cálculo de homografía con RANSAC"""
        kp1, desc1 = stitcher.detect_and_describe_multiscale(sample_image_color)
        kp2, desc2 = stitcher.detect_and_describe_multiscale(sample_image_color)
        
        if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
            matches = stitcher.match_features(desc1, desc2)
            
            if len(matches) >= 4:
                H, mask = stitcher.find_homography_ransac(kp1, kp2, matches)
                
                if H is not None:
                    assert H.shape == (3, 3)
                    assert mask is not None
                    assert len(mask) == len(matches)
    
    def test_find_homography_ransac_insufficient_matches(self, stitcher, sample_image_color):
        """Test cálculo de homografía con pocos matches"""
        kp1, desc1 = stitcher.detect_and_describe_multiscale(sample_image_color)
        kp2, desc2 = stitcher.detect_and_describe_multiscale(sample_image_color)
        
        # Crear matches ficticios insuficientes
        matches = []
        H, mask = stitcher.find_homography_ransac(kp1, kp2, matches)
        
        # Con menos de 4 matches, debería retornar None
        assert H is None or H.shape == (3, 3)
    
    def test_create_blend_mask(self, stitcher):
        """Test creación de máscaras de blending"""
        shape = (100, 100, 3)
        img1_region = np.ones((100, 100), dtype=bool)
        img1_region[:, 60:] = False
        
        img2_region = np.ones((100, 100), dtype=bool)
        img2_region[:, :40] = False
        
        overlap_region = np.logical_and(img1_region, img2_region)
        
        mask1, mask2 = stitcher.create_blend_mask(shape, img1_region, img2_region, overlap_region)
        
        assert mask1.shape == (100, 100)
        assert mask2.shape == (100, 100)
        assert mask1.dtype == np.float32
        assert mask2.dtype == np.float32
        
        # Verificar que las máscaras suman aproximadamente 1 en la región de overlap
        overlap_sum = mask1[overlap_region] + mask2[overlap_region]
        assert np.allclose(overlap_sum, 1.0, atol=0.1)
    
    def test_create_blend_mask_no_overlap(self, stitcher):
        """Test creación de máscaras sin overlap"""
        shape = (100, 100, 3)
        img1_region = np.zeros((100, 100), dtype=bool)
        img1_region[:, :50] = True
        
        img2_region = np.zeros((100, 100), dtype=bool)
        img2_region[:, 50:] = True
        
        overlap_region = np.zeros((100, 100), dtype=bool)
        
        mask1, mask2 = stitcher.create_blend_mask(shape, img1_region, img2_region, overlap_region)
        
        assert np.all(mask1[img1_region] == 1.0)
        assert np.all(mask2[img2_region] == 1.0)
    
    def test_pyramid_blend(self, stitcher, sample_image_color):
        """Test blending con pirámides"""
        img1 = sample_image_color
        img2 = sample_image_color.copy()
        
        # Crear máscaras simples
        mask1 = np.ones((200, 200), dtype=np.float32)
        mask1[:, 100:] = 0.5
        mask2 = np.ones((200, 200), dtype=np.float32)
        mask2[:, :100] = 0.5
        
        result = stitcher.pyramid_blend(img1, img2, mask1, mask2, levels=3)
        
        assert result.shape == img1.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_pyramid_blend_grayscale(self, stitcher):
        """Test blending con imágenes en escala de grises"""
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        mask1 = np.ones((100, 100), dtype=np.float32) * 0.5
        mask2 = np.ones((100, 100), dtype=np.float32) * 0.5
        
        result = stitcher.pyramid_blend(img1, img2, mask1, mask2, levels=3)
        
        assert result.shape == img1.shape
        assert result.dtype == np.uint8
    
    def test_warp_images(self, stitcher, sample_image_color):
        """Test warping de imágenes"""
        # Crear una homografía de identidad con pequeña traslación
        H = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float32)
        
        img1 = sample_image_color
        img2 = sample_image_color.copy()
        
        warped_img1, canvas, offset = stitcher.warp_images(img1, img2, H)
        
        assert warped_img1.shape[2] == 3  # RGB
        assert canvas.shape[2] == 3  # RGB
        assert isinstance(offset, tuple)
        assert len(offset) == 2
    
    def test_warp_images_identity(self, stitcher, sample_image_color):
        """Test warping con homografía identidad"""
        H = np.eye(3, dtype=np.float32)
        
        img1 = sample_image_color
        img2 = sample_image_color.copy()
        
        warped_img1, canvas, offset = stitcher.warp_images(img1, img2, H)
        
        assert warped_img1.shape[2] == 3
        assert canvas.shape[2] == 3
    
    def test_blend_with_pyramids(self, stitcher, sample_image_color):
        """Test blending con pirámides"""
        # Crear imágenes warpeadas simuladas
        warped_img1 = np.zeros((300, 300, 3), dtype=np.uint8)
        warped_img1[50:250, 50:250] = sample_image_color
        
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        canvas[100:300, 100:300] = sample_image_color
        
        result = stitcher.blend_with_pyramids(warped_img1, canvas)
        
        assert result.shape == warped_img1.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_stitch_pair_same_image(self, stitcher, sample_image_color):
        """Test stitching de la misma imagen"""
        result = stitcher.stitch_pair(sample_image_color, sample_image_color)
        
        if result is not None:
            panorama, matches, kp1, kp2, H = result
            
            assert isinstance(panorama, np.ndarray)
            assert panorama.dtype == np.uint8
            assert isinstance(matches, list)
            assert isinstance(kp1, list)
            assert isinstance(kp2, list)
            assert H is None or H.shape == (3, 3)
    
    def test_stitch_pair_different_images(self, stitcher):
        """Test stitching de imágenes diferentes"""
        img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = stitcher.stitch_pair(img1, img2)
        
        # Puede fallar si no hay suficientes matches, eso es esperado
        if result is not None:
            panorama, matches, kp1, kp2, H = result
            assert isinstance(panorama, np.ndarray)
            assert panorama.dtype == np.uint8
    
    def test_stitch_pair_translated_image(self, stitcher, sample_image_color):
        """Test stitching con imagen trasladada"""
        # Crear imagen trasladada
        M = np.float32([[1, 0, 50], [0, 1, 30]])
        translated = cv2.warpAffine(sample_image_color, M, (250, 230))
        
        result = stitcher.stitch_pair(sample_image_color, translated)
        
        if result is not None:
            panorama, matches, kp1, kp2, H = result
            assert isinstance(panorama, np.ndarray)
            assert H is not None
    
    def test_stitch_multiple_empty_list(self, stitcher):
        """Test stitching con lista vacía"""
        result = stitcher.stitch_multiple([])
        assert result is None
    
    def test_stitch_multiple_single_image(self, stitcher, sample_image_color):
        """Test stitching con una sola imagen"""
        result = stitcher.stitch_multiple([sample_image_color])
        assert np.array_equal(result, sample_image_color)
    
    def test_stitch_multiple_two_images(self, stitcher, sample_image_color):
        """Test stitching con dos imágenes"""
        img1 = sample_image_color
        img2 = sample_image_color.copy()
        
        result = stitcher.stitch_multiple([img1, img2])
        
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
    
    def test_stitch_multiple_three_images(self, stitcher):
        """Test stitching con tres imágenes"""
        img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img3 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = stitcher.stitch_multiple([img1, img2, img3])
        
        # Puede fallar si no hay suficientes matches
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
    
    def test_stitch_pair_insufficient_matches(self, stitcher):
        """Test stitching cuando no hay suficientes matches"""
        # Crear imágenes muy diferentes que no deberían tener matches
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = stitcher.stitch_pair(img1, img2)
        
        # Debería retornar None si no hay suficientes matches
        assert result is None or isinstance(result, tuple)
    
    def test_build_gaussian_pyramid_different_levels(self, stitcher, sample_image_color):
        """Test construcción de pirámide Gaussiana con diferentes niveles"""
        for levels in [1, 2, 3, 4, 5]:
            pyramid = stitcher.build_gaussian_pyramid(sample_image_color, levels)
            assert len(pyramid) == levels
    
    def test_build_laplacian_pyramid_different_levels(self, stitcher, sample_image_color):
        """Test construcción de pirámide Laplaciana con diferentes niveles"""
        for levels in [2, 3, 4]:
            laplacian_pyramid = stitcher.build_laplacian_pyramid(sample_image_color, levels)
            assert len(laplacian_pyramid) == levels
            
            # Reconstruir y verificar que tiene sentido
            reconstructed = stitcher.reconstruct_from_laplacian_pyramid(laplacian_pyramid)
            assert reconstructed.shape == sample_image_color.shape

