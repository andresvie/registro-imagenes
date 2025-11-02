import pandas as pd
from .utils import ImageTransformer
from .registration import ImageRegistrator, RegistrationEvaluator
from .feature_detection import FeatureDetector
from .matching import FeatureMatcher

def run_parameter_study(image, parameter_name, parameter_values):
    """
    Estudia el efecto de un parámetro en la calidad del registro
    
    Args:
        image: Imagen base
        parameter_name: 'ratio', 'detector', etc.
        parameter_values: Lista de valores a probar
    """
    results = []
    
    transformer = ImageTransformer(image)
    
    # Crear una transformación de prueba
    img_transformed, M_true = transformer.apply_combined_transform(
        angle=15, tx=20, ty=-15, scale=1.1
    )
    
    for value in parameter_values:
        result = None
        if parameter_name == 'ratio':
            registrator = ImageRegistrator(detector=FeatureDetector(detector_type='SIFT'), matcher=FeatureMatcher(detector_type='SIFT', ratio_threshold=value))
            result = registrator.register_images(image, img_transformed)
        elif parameter_name == 'detector':
            registrator = ImageRegistrator(detector=FeatureDetector(detector_type=value), matcher=FeatureMatcher(detector_type=value, ratio_threshold=0.75))
            result = registrator.register_images(image, img_transformed)
        
        if result is not None and result['transform_matrix'] is not None:
            evaluator = RegistrationEvaluator()
            errors = evaluator.evaluate_registration(
                M_true, result['transform_matrix'], image.shape
            )
            
            errors['parameter_value'] = value
            errors['num_matches'] = result['num_matches']
            errors['num_inliers'] = result['num_inliers']
            results.append(errors)
    
    return pd.DataFrame(results)