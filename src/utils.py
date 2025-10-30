"""
Módulo de utilidades generales.
Funciones auxiliares para carga de imágenes, creación de imágenes sintéticas,
y cálculo de métricas de error.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Carga una imagen desde un archivo.
    
    Args:
        image_path: Ruta al archivo de imagen
        grayscale: Si cargar en escala de grises
    
    Returns:
        image: Imagen cargada
    """
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image: np.ndarray, output_path: str):
    """
    Guarda una imagen en un archivo.
    
    Args:
        image: Imagen a guardar
        output_path: Ruta de salida
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertir RGB a BGR para OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
    else:
        cv2.imwrite(output_path, image)


def create_synthetic_image(base_image: np.ndarray, 
                          rotation: float = 0, 
                          translation: Tuple[float, float] = (0, 0),
                          scale: float = 1.0,
                          shear: Tuple[float, float] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea una imagen sintética aplicando transformaciones conocidas.
    
    Args:
        base_image: Imagen base
        rotation: Ángulo de rotación en grados
        translation: Traslación en píxeles (tx, ty)
        scale: Factor de escala
        shear: Factor de cizallamiento (shx, shy)
    
    Returns:
        transformed_image: Imagen transformada
        true_homography: Homografía verdadera aplicada (3x3)
    """
    h, w = base_image.shape[:2]
    center = (w / 2, h / 2)
    
    # Matriz de transformación afín compuesta
    # Orden: escala -> rotación -> cizallamiento -> traslación
    M_scale = np.array([[scale, 0, 0],
                       [0, scale, 0],
                       [0, 0, 1]], dtype=np.float32)
    
    cos_r = np.cos(np.radians(rotation))
    sin_r = np.sin(np.radians(rotation))
    M_rotation = np.array([[cos_r, -sin_r, 0],
                          [sin_r, cos_r, 0],
                          [0, 0, 1]], dtype=np.float32)
    
    M_shear = np.array([[1, shear[0], 0],
                       [shear[1], 1, 0],
                       [0, 0, 1]], dtype=np.float32)
    
    M_translation = np.array([[1, 0, translation[0]],
                             [0, 1, translation[1]],
                             [0, 0, 1]], dtype=np.float32)
    
    # Transformar respecto al centro
    T1 = np.array([[1, 0, -center[0]],
                   [0, 1, -center[1]],
                   [0, 0, 1]], dtype=np.float32)
    
    T2 = np.array([[1, 0, center[0]],
                   [0, 1, center[1]],
                   [0, 0, 1]], dtype=np.float32)
    
    # Componer transformaciones
    M_affine = T2 @ M_translation @ M_rotation @ M_scale @ M_shear @ T1
    
    # Convertir a homografía (afín es un caso especial de homografía)
    true_homography = M_affine
    
    # Calcular nuevo tamaño de la imagen
    corners = np.array([[0, 0, 1],
                       [w, 0, 1],
                       [w, h, 1],
                       [0, h, 1]], dtype=np.float32).T
    
    transformed_corners = M_affine @ corners
    transformed_corners = transformed_corners / transformed_corners[2, :]
    
    min_x = int(np.floor(transformed_corners[0, :].min()))
    max_x = int(np.ceil(transformed_corners[0, :].max()))
    min_y = int(np.floor(transformed_corners[1, :].min()))
    max_y = int(np.ceil(transformed_corners[1, :].max()))
    
    new_w = max_x - min_x
    new_h = max_y - min_y
    
    # Ajustar la homografía para incluir la traslación necesaria
    M_adjust = np.array([[1, 0, -min_x],
                        [0, 1, -min_y],
                        [0, 0, 1]], dtype=np.float32)
    
    final_homography = M_adjust @ M_affine
    
    # Aplicar transformación
    transformed_image = cv2.warpPerspective(
        base_image,
        final_homography,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(base_image.shape) == 3 else 255
    )
    
    # Ajustar la homografía verdadera también
    true_homography = final_homography
    
    return transformed_image, true_homography


def calculate_registration_error(estimated_homography: np.ndarray, 
                                true_homography: np.ndarray,
                                image_shape: Tuple[int, int]) -> Dict[str, float]:
    """
    Calcula métricas de error para el registro.
    
    Args:
        estimated_homography: Homografía estimada (3x3)
        true_homography: Homografía verdadera (3x3)
        image_shape: Forma de la imagen (height, width)
    
    Returns:
        error_metrics: Diccionario con métricas de error
    """
    h, w = image_shape
    
    # Normalizar homografías (dividir por elemento [2,2])
    H_est_norm = estimated_homography / estimated_homography[2, 2]
    H_true_norm = true_homography / true_homography[2, 2]
    
    # Errores en los elementos de la matriz
    matrix_rmse = np.sqrt(np.mean((H_est_norm - H_true_norm)**2))
    
    # Error en puntos de prueba
    test_points = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1],
        [w//2, h//2, 1],
        [w//4, h//4, 1],
        [3*w//4, 3*h//4, 1]
    ], dtype=np.float32).T
    
    # Transformar puntos con homografías
    points_est = H_est_norm @ test_points
    points_est = points_est / points_est[2, :]
    
    points_true = H_true_norm @ test_points
    points_true = points_true / points_true[2, :]
    
    # Error euclidiano en puntos
    point_errors = np.sqrt(np.sum((points_est[:2, :] - points_true[:2, :])**2, axis=0))
    point_rmse = np.sqrt(np.mean(point_errors**2))
    point_mae = np.mean(point_errors)
    point_max_error = np.max(point_errors)
    
    # Extraer parámetros de transformación
    # Para rotación: atan2(m[1,0], m[0,0])
    rot_true = np.degrees(np.arctan2(true_homography[1, 0], true_homography[0, 0]))
    rot_est = np.degrees(np.arctan2(estimated_homography[1, 0], estimated_homography[0, 0]))
    rot_error = abs(rot_true - rot_est)
    
    # Escala (promedio de la norma de los vectores de escala)
    scale_true = np.sqrt(true_homography[0, 0]**2 + true_homography[1, 0]**2)
    scale_est = np.sqrt(estimated_homography[0, 0]**2 + estimated_homography[1, 0]**2)
    scale_error = abs(scale_true - scale_est)
    
    return {
        'matrix_rmse': matrix_rmse,
        'point_rmse': point_rmse,
        'point_mae': point_mae,
        'point_max_error': point_max_error,
        'rotation_error': rot_error,
        'scale_error': scale_error
    }


def visualize_comparison(image1: np.ndarray, image2: np.ndarray, 
                        title1: str = "Imagen 1", 
                        title2: str = "Imagen 2",
                        figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Visualiza dos imágenes lado a lado.
    
    Args:
        image1: Primera imagen
        image2: Segunda imagen
        title1: Título de la primera imagen
        title2: Título de la segunda imagen
        figsize: Tamaño de la figura
    
    Returns:
        fig: Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if len(image1.shape) == 2:
        axes[0].imshow(image1, cmap='gray')
    else:
        axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    if len(image2.shape) == 2:
        axes[1].imshow(image2, cmap='gray')
    else:
        axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

