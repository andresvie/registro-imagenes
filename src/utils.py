import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

class ImageTransformer:
    """Clase para aplicar transformaciones conocidas a imágenes"""
    
    def __init__(self, image: np.ndarray):
        self.original = image
        self.height, self.width = image.shape[:2]
        
    def apply_rotation(self, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica rotación a la imagen
        
        Args:
            angle: Ángulo en grados (positivo = antihorario)
            
        Returns:
            imagen_transformada, matriz_transformacion
        """
        center = (self.width / 2, self.height / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.original, M, (self.width, self.height))
        
        # Convertir a matriz 3x3
        M_full = np.vstack([M, [0, 0, 1]])
        return rotated, M_full
    
    def apply_translation(self, tx: float, ty: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica traslación a la imagen
        
        Args:
            tx: Traslación en x (píxeles)
            ty: Traslación en y (píxeles)
            
        Returns:
            imagen_transformada, matriz_transformacion
        """
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(self.original, M, (self.width, self.height))
        
        M_full = np.vstack([M, [0, 0, 1]])
        return translated, M_full
    
    def apply_scale(self, sx: float, sy: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica escalado a la imagen
        
        Args:
            sx: Factor de escala en x
            sy: Factor de escala en y
            
        Returns:
            imagen_transformada, matriz_transformacion
        """
        M = cv2.getRotationMatrix2D((self.width/2, self.height/2), 0, sx)
        M[1, 1] = sy  # Ajustar escala en y
        scaled = cv2.warpAffine(self.original, M, (self.width, self.height))
        
        M_full = np.vstack([M, [0, 0, 1]])
        return scaled, M_full
    
    def apply_combined_transform(self, angle: float, tx: float, ty: float, 
                                 scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica transformación combinada: escala -> rotación -> traslación
        
        Returns:
            imagen_transformada, matriz_transformacion_combinada
        """
        center = (self.width / 2, self.height / 2)
        
        # Crear matriz de transformación combinada
        M_scale = cv2.getRotationMatrix2D(center, 0, scale)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Combinar transformaciones
        M_temp = np.vstack([M_scale, [0, 0, 1]])
        M_rot_full = np.vstack([M_rot, [0, 0, 1]])
        M_trans_full = np.vstack([M_trans, [0, 0, 1]])
        
        M_combined = M_trans_full @ M_rot_full @ M_temp
        M_final = M_combined[:2, :]
        
        transformed = cv2.warpAffine(self.original, M_final, (self.width, self.height))
        
        return transformed, M_combined


def create_synthetic_dataset(image: np.ndarray, n_samples: int = 20) -> List[Dict]:
    """
    Crea un conjunto de datos sintético con transformaciones variadas
    
    Args:
        image: Imagen base
        n_samples: Número de imágenes transformadas a generar
        
    Returns:
        Lista de diccionarios con imágenes y transformaciones
    """
    transformer = ImageTransformer(image)
    dataset = []
    
    # Rangos de transformaciones
    angles = np.linspace(-30, 30, n_samples)
    translations_x = np.linspace(-50, 50, n_samples)
    translations_y = np.linspace(-30, 30, n_samples)
    scales = np.linspace(0.8, 1.2, n_samples)
    
    for i in range(n_samples):
        angle = angles[i]
        tx = translations_x[i]
        ty = translations_y[i]
        scale = scales[i]
        
        transformed, M = transformer.apply_combined_transform(angle, tx, ty, scale)
        
        dataset.append({
            'id': i,
            'image': transformed,
            'transform_matrix': M,
            'angle': angle,
            'translation_x': tx,
            'translation_y': ty,
            'scale': scale
        })
    
    return dataset


def visualize_results(img_ref, img_mov, img_registered, matches_info, errors):
    """Visualiza los resultados del registro"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Imagen de referencia
    axes[0, 0].imshow(img_ref, cmap='gray')
    axes[0, 0].set_title('Imagen de Referencia')
    axes[0, 0].axis('off')
    
    # Imagen movida
    axes[0, 1].imshow(img_mov, cmap='gray')
    axes[0, 1].set_title('Imagen Transformada')
    axes[0, 1].axis('off')
    
    # Imagen registrada
    axes[0, 2].imshow(img_registered, cmap='gray')
    axes[0, 2].set_title('Imagen Registrada')
    axes[0, 2].axis('off')
    
    # Diferencia antes del registro
    diff_before = np.abs(img_ref.astype(float) - img_mov.astype(float))
    axes[1, 0].imshow(diff_before, cmap='hot')
    axes[1, 0].set_title('Diferencia (Antes)')
    axes[1, 0].axis('off')
    
    # Diferencia después del registro
    diff_after = np.abs(img_ref.astype(float) - img_registered.astype(float))
    axes[1, 1].imshow(diff_after, cmap='hot')
    axes[1, 1].set_title('Diferencia (Después)')
    axes[1, 1].axis('off')
    
    # Métricas de error
    axes[1, 2].axis('off')
    error_text = f"""
    Métricas de Error:
    
    RMSE: {errors['rmse']:.2f} px
    Error Angular: {errors['angular_error_deg']:.2f}°
    Error Traslación: {errors['translation_error_px']:.2f} px
    Error Escala: {errors['scale_error_percent']:.2f}%
    
    Matches: {matches_info['num_matches']}
    Inliers: {matches_info['num_inliers']}
    """
    axes[1, 2].text(0.1, 0.5, error_text, fontsize=12, family='monospace',
                     verticalalignment='center')
    
    plt.tight_layout()
    return fig