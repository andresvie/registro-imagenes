import cv2
import numpy as np
from typing import List
from src.matching import FeatureMatcher
from src.feature_detection import FeatureDetector

class Stitcher:
    """
    Panorama image stitching using Image Pyramid Algorithm
    
    Key improvements over basic stitching:
    1. Gaussian Pyramid for multi-scale feature detection
    2. Laplacian Pyramid for multi-band blending
    3. Better handling of exposure differences
    4. Smoother transitions in overlapping regions
    """
    
    def __init__(self, num_pyramid_levels=4, detector: FeatureDetector = None, matcher: FeatureMatcher = None):
        
        self.num_pyramid_levels = num_pyramid_levels
        self.detector = detector
        self.matcher = matcher

    def build_gaussian_pyramid(self, image, levels):
        """
        Build Gaussian pyramid for multi-scale representation
        """
        pyramid = [image]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    def build_laplacian_pyramid(self, image, levels):
        """
        Build Laplacian pyramid for multi-band blending
        Laplacian pyramid stores the difference between Gaussian levels
        """
        gaussian_pyramid = self.build_gaussian_pyramid(image, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            # Expand the higher level and subtract from current level
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
            # Handle size mismatch
            h, w = gaussian_pyramid[i].shape[:2]
            expanded = expanded[:h, :w]
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)
        
        # Add the smallest Gaussian level as the top of Laplacian pyramid
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def reconstruct_from_laplacian_pyramid(self, laplacian_pyramid):
        """
        Reconstruct image from Laplacian pyramid
        """
        image = laplacian_pyramid[-1]
        
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            expanded = cv2.pyrUp(image)
            # Handle size mismatch
            h, w = laplacian_pyramid[i].shape[:2]
            expanded = expanded[:h, :w]
            image = cv2.add(expanded, laplacian_pyramid[i])
        
        return image
    
    def detect_and_describe_multiscale(self, image):
        """
        Detect keypoints and compute descriptors using multi-scale approach
        Uses Gaussian pyramid for better feature detection at different scales
        """
        # Build Gaussian pyramid
        pyramid = self.build_gaussian_pyramid(image, 3)
        
        all_keypoints = []
        all_descriptors = []
        
        for level, pyr_img in enumerate(pyramid):
            gray = cv2.cvtColor(pyr_img, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detect_and_compute(gray)
            
            if desc is not None and len(kp) > 0:
                # Scale keypoints back to original image size
                scale_factor = 2 ** level
                for keypoint in kp:
                    keypoint.pt = (keypoint.pt[0] * scale_factor, 
                                  keypoint.pt[1] * scale_factor)
                    keypoint.size *= scale_factor
                
                all_keypoints.extend(kp)
                all_descriptors.append(desc)
        
        # Combine all descriptors
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
        Find homography matrix using RANSAC
        """
        
        
        H, mask = self.matcher.estimate_transform(kp1, kp2, matches, method='RANSAC')
        
        return H, mask
    
    def create_blend_mask(self, shape, img1_region, img2_region, overlap_region, feather_amount=50):
        """
        Create smooth blending masks using distance transform for seamless blending
        """
        h, w = shape[:2]
        mask1 = np.zeros((h, w), dtype=np.float32)
        mask2 = np.zeros((h, w), dtype=np.float32)
        
        # Set base masks
        mask1[img1_region] = 1.0
        mask2[img2_region] = 1.0
        
        if overlap_region.sum() > 0:
            # Create smooth transition in overlap region using distance transform
            overlap_mask = overlap_region.astype(np.uint8)
            
            # Distance from img1's edge
            img1_only = np.logical_and(img1_region, np.logical_not(img2_region)).astype(np.uint8)
            dist1 = cv2.distanceTransform(overlap_mask, cv2.DIST_L2, 5)
            
            # Distance from img2's edge
            img2_only = np.logical_and(img2_region, np.logical_not(img1_region)).astype(np.uint8)
            dist2 = cv2.distanceTransform(overlap_mask, cv2.DIST_L2, 5)
            
            # Create smooth transition
            if dist1.max() > 0 and dist2.max() > 0:
                # Normalize distances
                weight1 = dist1 / (dist1 + dist2 + 1e-6)
                weight2 = 1.0 - weight1
                
                # Apply Gaussian smoothing for even smoother transition
                weight1 = cv2.GaussianBlur(weight1, (feather_amount*2+1, feather_amount*2+1), feather_amount/3)
                weight2 = cv2.GaussianBlur(weight2, (feather_amount*2+1, feather_amount*2+1), feather_amount/3)
                
                # Normalize
                total = weight1 + weight2
                weight1 = weight1 / (total + 1e-6)
                weight2 = weight2 / (total + 1e-6)
                
                mask1[overlap_region] = weight1[overlap_region]
                mask2[overlap_region] = weight2[overlap_region]
        
        return mask1, mask2
    
    def pyramid_blend(self, img1, img2, mask1, mask2, levels=None):
        """
        Multi-band blending using Laplacian pyramids
        This creates seamless blends by blending different frequency bands separately
        """
        if levels is None:
            levels = self.num_pyramid_levels
        
        # Ensure images are float
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # Build Laplacian pyramids for both images
        lap_pyr1 = self.build_laplacian_pyramid(img1, levels)
        lap_pyr2 = self.build_laplacian_pyramid(img2, levels)
        
        # Build Gaussian pyramid for masks
        mask1_pyramid = self.build_gaussian_pyramid(mask1, levels)
        mask2_pyramid = self.build_gaussian_pyramid(mask2, levels)
        
        # Blend each level
        blended_pyramid = []
        for i in range(levels):
            # Expand mask to 3 channels if needed
            if len(lap_pyr1[i].shape) == 3:
                m1 = np.dstack([mask1_pyramid[i]] * 3)
                m2 = np.dstack([mask2_pyramid[i]] * 3)
            else:
                m1 = mask1_pyramid[i]
                m2 = mask2_pyramid[i]
            
            # Blend at this level
            blended = lap_pyr1[i] * m1 + lap_pyr2[i] * m2
            blended_pyramid.append(blended)
        
        # Reconstruct from blended pyramid
        result = self.reconstruct_from_laplacian_pyramid(blended_pyramid)
        
        # Clip values to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def warp_images(self, img1, img2, H):
        """
        Warp img1 to img2's coordinate system using homography H
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Warp corners
        warped_corners1 = cv2.perspectiveTransform(corners1, H)
        
        # Get bounds
        all_corners = np.concatenate((warped_corners1, corners2), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Translation
        translation = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]])
        
        output_size = (x_max - x_min, y_max - y_min)
        
        # Warp img1
        warped_img1 = cv2.warpPerspective(img1, translation.dot(H), output_size)
        
        # Create canvas for img2
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2
        
        return warped_img1, canvas, (-x_min, -y_min)
    
    def blend_with_pyramids(self, warped_img1, canvas):
        """
        Blend two images using pyramid blending for seamless results
        """
        # Create masks
        mask1_binary = (warped_img1.sum(axis=2) > 0).astype(np.uint8)
        mask2_binary = (canvas.sum(axis=2) > 0).astype(np.uint8)
        
        # Find overlapping region
        overlap = np.logical_and(mask1_binary, mask2_binary)
        
        # Create smooth blending masks
        mask1, mask2 = self.create_blend_mask(
            warped_img1.shape, 
            mask1_binary.astype(bool), 
            mask2_binary.astype(bool),
            overlap,
            feather_amount=10
        )
        
        # Use pyramid blending
        result = self.pyramid_blend(warped_img1, canvas, mask1, mask2)
        
        return result
    
    def stitch_pair(self, img1, img2):
        """
        Stitch two images using pyramid-based approach
        """
        print(f"Image 1 shape: {img1.shape}")
        print(f"Image 2 shape: {img2.shape}")
        
        # 1. Multi-scale feature detection
        print("Detecting features with pyramid...")
        kp1, desc1 = self.detect_and_describe_multiscale(img1)
        kp2, desc2 = self.detect_and_describe_multiscale(img2)
        print(f"Found {len(kp1)} keypoints in image 1 (multi-scale)")
        print(f"Found {len(kp2)} keypoints in image 2 (multi-scale)")
        
        # 2. Match features
        print("Matching features...")
        matches = self.match_features(desc1, desc2)
        print(f"Found {len(matches)} good matches")
        
        if len(matches) < 4:
            print("Not enough matches found!")
            return None
        
        # 3. Find homography
        print("Computing homography...")
        H, mask = self.find_homography_ransac(kp1, kp2, matches)
        
        if H is None:
            print("Could not compute homography!")
            return None
        
        inliers = mask.ravel().sum()
        print(f"Homography computed with {inliers} inliers")
        
        # 4. Warp images
        print("Warping images...")
        warped_img1, canvas, offset = self.warp_images(img1, img2, H)
        
        # 5. Pyramid blending
        print("Blending with pyramid algorithm...")
        result = self.blend_with_pyramids(warped_img1, canvas)
        
        return result, matches, kp1, kp2, H
    
    def stitch_multiple(self, images: List[np.ndarray]):
        """
        Stitch multiple images using pyramid approach
        """
        if len(images) < 2:
            return images[0] if len(images) == 1 else None
        
        # Use middle image as reference
        middle_idx = len(images) // 2
        print(f"\n=== Using image {middle_idx + 1} as reference ===")
        result = images[middle_idx]
        
        # Stitch to the right
        for i in range(middle_idx + 1, len(images)):
            print(f"\n=== Stitching image {i + 1} (right) ===")
            stitch_output = self.stitch_pair(result, images[i])
            if stitch_output is None:
                print(f"Failed to stitch image {i + 1}")
                return None
            result = stitch_output[0]
        
        # Stitch to the left
        for i in range(middle_idx - 1, -1, -1):
            print(f"\n=== Stitching image {i + 1} (left) ===")
            stitch_output = self.stitch_pair(images[i], result)
            if stitch_output is None:
                print(f"Failed to stitch image {i + 1}")
                return None
            result = stitch_output[0]
        
        return result


