import cv2
import numpy as np
from typing import List
from src.matching import FeatureMatcher
from src.feature_detection import FeatureDetector

class Stitcher:
    """
    Fusión de imágenes panorámicas usando el algoritmo de pirámide de imagen
    
    Mejoras clave sobre la fusión básica:
    1. Pirámide Gaussiana para detección multi-escala de características
    2. Pirámide Laplaciana para mezcla multi-banda
    3. Manejo mejorado de diferencias de exposición
    4. Transiciones más suaves en regiones de superposición
    """
    
    def __init__(self, num_pyramid_levels=4, detector: FeatureDetector = None, matcher: FeatureMatcher = None):
        
        self.num_pyramid_levels = num_pyramid_levels
        self.detector = detector
        self.matcher = matcher

    def build_gaussian_pyramid(self, image, levels):
        """
        Construir pirámide Gaussiana para representación multi-escala
        """
        pyramid = [image]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    def build_laplacian_pyramid(self, image, levels):
        """
        Construir pirámide Laplaciana para mezcla multi-banda
        Pirámide Laplaciana almacena la diferencia entre niveles Gaussiano
        """
        gaussian_pyramid = self.build_gaussian_pyramid(image, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            # Expandir el nivel superior y restar del nivel actual
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
            # Manejar desajuste de tamaño
            h, w = gaussian_pyramid[i].shape[:2]
            expanded = expanded[:h, :w]
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)
        
        # Agregar el nivel Gaussiano más pequeño como el top de la pirámide Laplaciana
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def reconstruct_from_laplacian_pyramid(self, laplacian_pyramid):
        """
        Reconstruir imagen desde pirámide Laplaciana
        """
        image = laplacian_pyramid[-1]
        
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            expanded = cv2.pyrUp(image)
            # Manejar desajuste de tamaño
            h, w = laplacian_pyramid[i].shape[:2]
            expanded = expanded[:h, :w]
            image = cv2.add(expanded, laplacian_pyramid[i])
        
        return image
    
    def detect_and_describe_multiscale(self, image):
        """
        Detectar keypoints y computar descriptores usando enfoque multi-escala
        Uses Gaussian pyramid for better feature detection at different scales
        """
        # Construir pirámide Gaussiana
        pyramid = self.build_gaussian_pyramid(image, 3)
        
        all_keypoints = []
        all_descriptors = []
        
        for level, pyr_img in enumerate(pyramid):
            gray = cv2.cvtColor(pyr_img, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detect_and_compute(gray)
            
            if desc is not None and len(kp) > 0:
                # Escalar keypoints de nuevo a tamaño original de la imagen
                scale_factor = 2 ** level
                for keypoint in kp:
                    keypoint.pt = (keypoint.pt[0] * scale_factor, 
                                  keypoint.pt[1] * scale_factor)
                    keypoint.size *= scale_factor
                
                all_keypoints.extend(kp)
                all_descriptors.append(desc)
        
        # Combinar todos los descriptores
        if len(all_descriptors) > 0:
            combined_descriptors = np.vstack(all_descriptors)
        else:
            combined_descriptors = None
            
        return all_keypoints, combined_descriptors
    
    def match_features(self, desc1, desc2):
        
        matches = self.matcher.match_features(desc1, desc2)        
        return matches
    
    def find_homography_ransac(self, kp1, kp2, matches):
        """
        Encontrar matriz de homografía usando RANSAC
        """
        
        
        H, mask = self.matcher.estimate_transform(kp1, kp2, matches, method='RANSAC')
        
        return H, mask
    
    def create_blend_mask(self, shape, img1_region, img2_region, overlap_region, feather_amount=50):
        """
        Crear máscaras de mezcla suaves usando la transformada de distancia para una mezcla suave
        """
        h, w = shape[:2]
        mask1 = np.zeros((h, w), dtype=np.float32)
        mask2 = np.zeros((h, w), dtype=np.float32)
        
        # Establecer máscaras base
        mask1[img1_region] = 1.0
        mask2[img2_region] = 1.0
        
        if overlap_region.sum() > 0:
            # Crear transición suave en la región de superposición usando la transformada de distancia
            overlap_mask = overlap_region.astype(np.uint8)
            
            # Distancia desde el borde de img1
            img1_only = np.logical_and(img1_region, np.logical_not(img2_region)).astype(np.uint8)
            dist1 = cv2.distanceTransform(overlap_mask, cv2.DIST_L2, 5)
            
            # Distancia desde el borde de img2
            img2_only = np.logical_and(img2_region, np.logical_not(img1_region)).astype(np.uint8)
            dist2 = cv2.distanceTransform(overlap_mask, cv2.DIST_L2, 5)
            
            # Crear transición suave
            if dist1.max() > 0 and dist2.max() > 0:
                # Normalizar distancias
                weight1 = dist1 / (dist1 + dist2 + 1e-6)
                weight2 = 1.0 - weight1
                
                # Aplicar suavizado Gaussiano para una transición más suave
                weight1 = cv2.GaussianBlur(weight1, (feather_amount*2+1, feather_amount*2+1), feather_amount/3)
                weight2 = cv2.GaussianBlur(weight2, (feather_amount*2+1, feather_amount*2+1), feather_amount/3)
                
                # Normalizar
                total = weight1 + weight2
                weight1 = weight1 / (total + 1e-6)
                weight2 = weight2 / (total + 1e-6)
                
                mask1[overlap_region] = weight1[overlap_region]
                mask2[overlap_region] = weight2[overlap_region]
        
        return mask1, mask2
    
    def pyramid_blend(self, img1, img2, mask1, mask2, levels=None):
        """
        Mezcla multi-banda usando pirámides Laplacianas
        Esto crea mezclas suaves fusionando diferentes bandas de frecuencia por separado
        """
        if levels is None:
            levels = self.num_pyramid_levels
        
        # Asegurar que las imágenes sean float
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Build Laplacian pyramids for both images
        lap_pyr1 = self.build_laplacian_pyramid(img1, levels)
        lap_pyr2 = self.build_laplacian_pyramid(img2, levels)
        
        # Construir pirámide Gaussiana para máscaras
        mask1_pyramid = self.build_gaussian_pyramid(mask1, levels)
        mask2_pyramid = self.build_gaussian_pyramid(mask2, levels)
        
        # Mezclar cada nivel
        blended_pyramid = []
        for i in range(levels):
            # Expandir máscara a 3 canales si es necesario
            if len(lap_pyr1[i].shape) == 3:
                m1 = np.dstack([mask1_pyramid[i]] * 3)
                m2 = np.dstack([mask2_pyramid[i]] * 3)
            else:
                m1 = mask1_pyramid[i]
                m2 = mask2_pyramid[i]
            
            # Blend at this level
            blended = lap_pyr1[i] * m1 + lap_pyr2[i] * m2
            blended_pyramid.append(blended)
        
        # Reconstruir desde la pirámide mezclada
        result = self.reconstruct_from_laplacian_pyramid(blended_pyramid)
        
        # Recortar valores a rango válido
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def warp_images(self, img1, img2, H):
        """
        Transformar img1 al sistema de coordenadas de img2 usando homografía H
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Obtener esquinas
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Transformar esquinas
        warped_corners1 = cv2.perspectiveTransform(corners1, H)
        
        # Obtener límites
        all_corners = np.concatenate((warped_corners1, corners2), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Traslación
        translation = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]])
        
        output_size = (x_max - x_min, y_max - y_min)
        
        # Transformar img1
        warped_img1 = cv2.warpPerspective(img1, translation.dot(H), output_size)
        
        # Crear lienzo para img2
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2
        
        return warped_img1, canvas, (-x_min, -y_min)
    
    def blend_with_pyramids(self, warped_img1, canvas):
        """
        Mezcla de dos imágenes usando mezcla de pirámide para resultados suaves
        """
        # Crear máscaras
        mask1_binary = (warped_img1.sum(axis=2) > 0).astype(np.uint8)
        mask2_binary = (canvas.sum(axis=2) > 0).astype(np.uint8)
        
        # Encontrar región de superposición
        overlap = np.logical_and(mask1_binary, mask2_binary)
        
        # Crear máscaras de mezcla suaves
        mask1, mask2 = self.create_blend_mask(
            warped_img1.shape, 
            mask1_binary.astype(bool), 
            mask2_binary.astype(bool),
            overlap,
            feather_amount=10
        )
        
        # Usar mezcla de pirámide
        result = self.pyramid_blend(warped_img1, canvas, mask1, mask2)
        
        return result
    
    def stitch_pair(self, img1, img2):
        """
        Fusión de dos imágenes usando enfoque de pirámide
        """
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        
        # 1. Detectar características usando pirámide
        print("Detecting features with pyramid...")
        kp1, desc1 = self.detect_and_describe_multiscale(img1)
        kp2, desc2 = self.detect_and_describe_multiscale(img2)
        print(f"Found {len(kp1)} keypoints in image 1 (multi-scale)")
        print(f"Found {len(kp2)} keypoints in image 2 (multi-scale)")
        
        # 2. Emparejar características
        print("Matching features...")
        matches = self.match_features(desc1, desc2)
        print(f"Found {len(matches)} good matches")
        
        if len(matches) < 4:
            print("Not enough matches found!")
            return None
        
        # 3. Encontrar la homografía
        print("Computing homography...")
        H, mask = self.find_homography_ransac(kp1, kp2, matches)
        
        if H is None:
            print("Could not compute homography!")
            return None
        
        inliers = mask.ravel().sum()
        print(f"Homography computed with {inliers} inliers")
        
        # 4. Transformar las imágenes
        print("Warping images...")
        warped_img1, canvas, offset = self.warp_images(img1, img2, H)
        
        # 5. Mezclar las imágenes usando el algoritmo de pirámide
        print("Blending with pyramid algorithm...")
        result = self.blend_with_pyramids(warped_img1, canvas)
        
        return result, matches, kp1, kp2, H
    
    def stitch_multiple(self, images: List[np.ndarray]):
        """
        Fusión de múltiples imágenes usando enfoque de pirámide
        """
        if len(images) < 2:
            return images[0] if len(images) == 1 else None
        
        # Usar la imagen del medio como referencia
        middle_idx = len(images) // 2
        print(f"\n=== Usando imagen {middle_idx + 1} como referencia ===")
        result = images[middle_idx]
        
        # Pegar a la derecha
        for i in range(middle_idx + 1, len(images)):
            print(f"\n=== Fusión de imagen {i + 1} (derecha) ===")
            stitch_output = self.stitch_pair(result, images[i])
            if stitch_output is None:
                print(f"No se pudo fusionar la imagen {i + 1}")
                return None
            result = stitch_output[0]
        
        # Pegar a la izquierda
        for i in range(middle_idx - 1, -1, -1):
            print(f"\n=== Fusión de imagen {i + 1} (izquierda) ===")
            stitch_output = self.stitch_pair(images[i], result)
            if stitch_output is None:
                print(f"No se pudo fusionar la imagen {i + 1}")
                return None
            result = stitch_output[0]
        
        return result


