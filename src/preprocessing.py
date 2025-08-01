"""
Image preprocessing module for child malnutrition detection.
Handles image loading, preprocessing, and feature extraction.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for malnutrition detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target size for resizing images (width, height)
        """
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]  # ImageNet mean
        self.std = [0.229, 0.224, 0.225]   # ImageNet std
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image
        """
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        
        return image
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        return image
    
    def preprocess_single_image(self, image_path: str, augment: bool = False) -> Optional[np.ndarray]:
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to the image file
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed image ready for model input
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Resize image
        image = self.resize_image(image)
        
        # Apply augmentation if requested
        if augment:
            image = self.augment_image(image)
        
        # Normalize image
        image = self.normalize_image(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_batch(self, image_paths: List[str], augment: bool = False) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image file paths
            augment: Whether to apply data augmentation
            
        Returns:
            Batch of preprocessed images
        """
        processed_images = []
        
        for image_path in image_paths:
            processed_image = self.preprocess_single_image(image_path, augment)
            if processed_image is not None:
                processed_images.append(processed_image[0])  # Remove batch dimension
        
        if not processed_images:
            logger.error("No images were successfully processed")
            return np.array([])
        
        return np.array(processed_images)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract basic features from image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Extracted features
        """
        features = []
        
        # Color features
        features.extend([
            np.mean(image[:, :, 0]),  # Red channel mean
            np.mean(image[:, :, 1]),  # Green channel mean
            np.mean(image[:, :, 2]),  # Blue channel mean
            np.std(image[:, :, 0]),   # Red channel std
            np.std(image[:, :, 1]),   # Green channel std
            np.std(image[:, :, 2]),   # Blue channel std
        ])
        
        # Texture features (simple edge detection)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.mean(edges),           # Edge density
            np.std(edges),            # Edge variation
        ])
        
        # Shape features
        features.extend([
            image.shape[0] / image.shape[1],  # Aspect ratio
            np.sum(gray > 128) / gray.size,   # Brightness ratio
        ])
        
        return np.array(features)
    
    def save_processed_image(self, image: np.ndarray, output_path: str) -> bool:
        """
        Save processed image to file.
        
        Args:
            image: Processed image as numpy array
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Denormalize image
            image = image.copy()
            for i in range(3):
                image[:, :, i] = image[:, :, i] * self.std[i] + self.mean[i]
            
            # Scale back to [0, 255]
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            # Save image
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return True
            
        except Exception as e:
            logger.error(f"Error saving image {output_path}: {str(e)}")
            return False


def create_preprocessor(target_size: Tuple[int, int] = (224, 224)) -> ImagePreprocessor:
    """
    Factory function to create an image preprocessor.
    
    Args:
        target_size: Target size for resizing images
        
    Returns:
        Configured ImagePreprocessor instance
    """
    return ImagePreprocessor(target_size)


# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Example preprocessing
    image_path = "data/test/sample_image.jpg"
    if os.path.exists(image_path):
        processed_image = preprocessor.preprocess_single_image(image_path)
        if processed_image is not None:
            print(f"Processed image shape: {processed_image.shape}")
            features = preprocessor.extract_features(processed_image[0])
            print(f"Extracted features shape: {features.shape}") 