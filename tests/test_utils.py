"""
Tests unitarios para utils.py
"""
import pytest
import numpy as np
import cv2
from src.utils import ImageTransformer, create_synthetic_dataset


class TestImageTransformer:
    """Tests para la clase ImageTransformer"""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def transformer(self, sample_image):
        """Crea un transformador de prueba"""
        return ImageTransformer(sample_image)
    
    def test_init(self, sample_image):
        """Test inicialización del transformador"""
        transformer = ImageTransformer(sample_image)
        
        assert transformer.original is not None
        assert transformer.height == 100
        assert transformer.width == 100
    
    def test_apply_rotation(self, transformer):
        """Test aplicación de rotación"""
        angle = 45.0
        rotated, M = transformer.apply_rotation(angle)
        
        assert rotated is not None
        assert rotated.shape == transformer.original.shape[:2] or rotated.shape == transformer.original.shape
        assert M is not None
        assert M.shape == (3, 3)
    
    def test_apply_rotation_zero_angle(self, transformer):
        """Test rotación con ángulo cero"""
        rotated, M = transformer.apply_rotation(0.0)
        
        assert rotated is not None
        assert M is not None
    
    def test_apply_rotation_negative_angle(self, transformer):
        """Test rotación con ángulo negativo"""
        rotated, M = transformer.apply_rotation(-30.0)
        
        assert rotated is not None
        assert M is not None
    
    def test_apply_rotation_large_angle(self, transformer):
        """Test rotación con ángulo grande"""
        rotated, M = transformer.apply_rotation(360.0)
        
        assert rotated is not None
        assert M is not None
    
    def test_apply_translation(self, transformer):
        """Test aplicación de traslación"""
        tx, ty = 10.0, -5.0
        translated, M = transformer.apply_translation(tx, ty)
        
        assert translated is not None
        assert translated.shape == transformer.original.shape[:2] or translated.shape == transformer.original.shape
        assert M is not None
        assert M.shape == (3, 3)
    
    def test_apply_translation_zero(self, transformer):
        """Test traslación cero"""
        translated, M = transformer.apply_translation(0.0, 0.0)
        
        assert translated is not None
        assert M is not None
    
    def test_apply_scale(self, transformer):
        """Test aplicación de escalado"""
        sx, sy = 1.5, 0.8
        scaled, M = transformer.apply_scale(sx, sy)
        
        assert scaled is not None
        assert scaled.shape == transformer.original.shape[:2] or scaled.shape == transformer.original.shape
        assert M is not None
        assert M.shape == (3, 3)
    
    def test_apply_scale_uniform(self, transformer):
        """Test escalado uniforme"""
        scaled, M = transformer.apply_scale(1.2, 1.2)
        
        assert scaled is not None
        assert M is not None
    
    def test_apply_scale_small(self, transformer):
        """Test escalado pequeño"""
        scaled, M = transformer.apply_scale(0.5, 0.5)
        
        assert scaled is not None
        assert M is not None
    
    def test_apply_combined_transform(self, transformer):
        """Test transformación combinada"""
        angle, tx, ty, scale = 15.0, 10.0, -5.0, 1.1
        transformed, M = transformer.apply_combined_transform(angle, tx, ty, scale)
        
        assert transformed is not None
        assert transformed.shape == transformer.original.shape[:2] or transformed.shape == transformer.original.shape
        assert M is not None
        assert M.shape == (3, 3)
    
    def test_apply_combined_transform_identity(self, transformer):
        """Test transformación combinada con valores de identidad"""
        transformed, M = transformer.apply_combined_transform(0.0, 0.0, 0.0, 1.0)
        
        assert transformed is not None
        assert M is not None
    
    def test_apply_combined_transform_large_values(self, transformer):
        """Test transformación combinada con valores grandes"""
        transformed, M = transformer.apply_combined_transform(180.0, 50.0, 50.0, 2.0)
        
        assert transformed is not None
        assert M is not None
    
    def test_rotation_matrix_properties(self, transformer):
        """Test que la matriz de rotación tiene las propiedades correctas"""
        rotated, M = transformer.apply_rotation(45.0)
        
        # La última fila debería ser [0, 0, 1] para matriz de transformación
        assert abs(M[2, 0]) < 1e-6
        assert abs(M[2, 1]) < 1e-6
        assert abs(M[2, 2] - 1.0) < 1e-6
    
    def test_translation_matrix_properties(self, transformer):
        """Test que la matriz de traslación tiene las propiedades correctas"""
        translated, M = transformer.apply_translation(10.0, 20.0)
        
        # Verificar componente de traslación
        assert abs(M[0, 2] - 10.0) < 1e-6
        assert abs(M[1, 2] - 20.0) < 1e-6


class TestCreateSyntheticDataset:
    """Tests para la función create_synthetic_dataset"""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba"""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_create_synthetic_dataset_default(self, sample_image):
        """Test creación de dataset con valores por defecto"""
        dataset = create_synthetic_dataset(sample_image)
        
        assert isinstance(dataset, list)
        assert len(dataset) == 20  # Valor por defecto
    
    def test_create_synthetic_dataset_custom_n_samples(self, sample_image):
        """Test creación de dataset con n_samples personalizado"""
        n_samples = 10
        dataset = create_synthetic_dataset(sample_image, n_samples=n_samples)
        
        assert isinstance(dataset, list)
        assert len(dataset) == n_samples
    
    def test_create_synthetic_dataset_structure(self, sample_image):
        """Test estructura de los elementos del dataset"""
        dataset = create_synthetic_dataset(sample_image, n_samples=5)
        
        for item in dataset:
            assert isinstance(item, dict)
            assert 'id' in item
            assert 'image' in item
            assert 'transform_matrix' in item
            assert 'angle' in item
            assert 'translation_x' in item
            assert 'translation_y' in item
            assert 'scale' in item
            
            assert isinstance(item['id'], int)
            assert isinstance(item['image'], np.ndarray)
            assert isinstance(item['transform_matrix'], np.ndarray)
            assert isinstance(item['angle'], (int, float))
            assert isinstance(item['translation_x'], (int, float))
            assert isinstance(item['translation_y'], (int, float))
            assert isinstance(item['scale'], (int, float))
    
    def test_create_synthetic_dataset_image_shapes(self, sample_image):
        """Test que las imágenes del dataset tienen el mismo shape"""
        dataset = create_synthetic_dataset(sample_image, n_samples=5)
        
        original_shape = sample_image.shape
        for item in dataset:
            assert item['image'].shape == original_shape
    
    def test_create_synthetic_dataset_transform_matrices(self, sample_image):
        """Test que las matrices de transformación tienen el tamaño correcto"""
        dataset = create_synthetic_dataset(sample_image, n_samples=5)
        
        for item in dataset:
            assert item['transform_matrix'].shape == (3, 3)
    
    def test_create_synthetic_dataset_unique_ids(self, sample_image):
        """Test que los IDs son únicos"""
        dataset = create_synthetic_dataset(sample_image, n_samples=10)
        
        ids = [item['id'] for item in dataset]
        assert len(ids) == len(set(ids))  # Todos los IDs son únicos
    
    def test_create_synthetic_dataset_single_sample(self, sample_image):
        """Test creación de dataset con un solo sample"""
        dataset = create_synthetic_dataset(sample_image, n_samples=1)
        
        assert len(dataset) == 1
        assert dataset[0]['id'] == 0
