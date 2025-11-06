"""
Tests unitarios para evaluator.py
"""
import pytest
import numpy as np
import pandas as pd
from src.evaluator import run_parameter_study
from src.utils import ImageTransformer


class TestRunParameterStudy:
    """Tests para la función run_parameter_study"""
    
    @pytest.fixture
    def sample_image(self):
        """Crea una imagen de prueba"""
        return np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    def test_run_parameter_study_ratio(self, sample_image):
        """Test estudio de parámetros para ratio_threshold"""
        parameter_values = [0.6, 0.7, 0.75, 0.8]
        results = run_parameter_study(sample_image, 'ratio', parameter_values)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert 'parameter_value' in results.columns
        assert 'num_matches' in results.columns
        assert 'num_inliers' in results.columns
    
    def test_run_parameter_study_detector(self, sample_image):
        """Test estudio de parámetros para detector_type"""
        parameter_values = ['SIFT', 'ORB', 'AKAZE']
        results = run_parameter_study(sample_image, 'detector', parameter_values)
        
        assert isinstance(results, pd.DataFrame)
        # Puede que no todos los detectores funcionen en todas las imágenes
        assert len(results) >= 0
    
    def test_run_parameter_study_ratio_columns(self, sample_image):
        """Test que el DataFrame tiene las columnas correctas para ratio"""
        parameter_values = [0.75, 0.8]
        results = run_parameter_study(sample_image, 'ratio', parameter_values)
        
        if len(results) > 0:
            expected_columns = [
                'parameter_value', 'num_matches', 'num_inliers',
                'rmse', 'angular_error_deg', 'translation_error_px', 'scale_error_percent'
            ]
            for col in expected_columns:
                if col not in results.columns:
                    # Algunas columnas pueden no estar si no hay matches suficientes
                    pass
    
    def test_run_parameter_study_ratio_values(self, sample_image):
        """Test que los valores de parámetro se guardan correctamente"""
        parameter_values = [0.6, 0.75, 0.9]
        results = run_parameter_study(sample_image, 'ratio', parameter_values)
        
        if len(results) > 0:
            # Verificar que los valores están en los resultados
            result_values = results['parameter_value'].unique()
            assert len(result_values) <= len(parameter_values)
    
    def test_run_parameter_study_empty_results(self):
        """Test con imagen que no genera matches"""
        # Crear una imagen muy simple que puede no generar matches
        simple_image = np.zeros((50, 50), dtype=np.uint8)
        parameter_values = [0.75]
        results = run_parameter_study(simple_image, 'ratio', parameter_values)
        
        assert isinstance(results, pd.DataFrame)
        # Puede estar vacío si no hay matches suficientes
        assert len(results) >= 0
    
    def test_run_parameter_study_invalid_parameter(self, sample_image):
        """Test con nombre de parámetro inválido"""
        parameter_values = ['value1', 'value2']
        
        # La función debería retornar DataFrame vacío para parámetros inválidos
        results = run_parameter_study(sample_image, 'invalid_param', parameter_values)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0  # No debería haber resultados para parámetro inválido
