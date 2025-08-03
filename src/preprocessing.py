"""
Data preprocessing module for malnutrition detection pipeline.
Handles image preprocessing, data augmentation, and data generators.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io
import os
from typing import Tuple, Optional, Union
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Comprehensive image preprocessing for malnutrition detection.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image dimensions (width, height)
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def preprocess_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model prediction.
        
        Args:
            image_input: Image as file path, bytes, numpy array, or PIL Image
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                # File path
                image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                # Bytes data
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = Image.fromarray(image_input.astype('uint8'))
            elif isinstance(image_input, Image.Image):
                # Already PIL Image
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            logger.debug(f"Image preprocessed to shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def enhance_image(self, image: Image.Image, enhance_factor: float = 1.2) -> Image.Image:
        """
        Enhance image quality for better feature extraction.
        
        Args:
            image: PIL Image to enhance
            enhance_factor: Enhancement factor (>1 increases, <1 decreases)
            
        Returns:
            Enhanced PIL Image
        """
        # Enhance brightness
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(enhance_factor)
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(enhance_factor)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(enhance_factor)
        
        return image
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            CLAHE enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced

def create_data_generators(
    train_dir: str,
    validation_split: float = 0.2,
    batch_size: int = 8,
    target_size: Tuple[int, int] = (128, 128),
    seed: int = 42
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Create enhanced data generators with augmentation.
    
    Args:
        train_dir: Directory containing training data
        validation_split: Fraction of data to use for validation
        batch_size: Batch size for training
        target_size: Target image dimensions
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    logger.info(f"Creating data generators for directory: {train_dir}")
    
    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=validation_split,
        
        # Geometric transformations
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        
        # Intensity transformations
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        
        # Fill mode for transformations
        fill_mode='nearest'
    )
    
    # Validation generator (only rescaling)
    validation_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=validation_split
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=seed,
        shuffle=True
    )
    
    # Create validation generator
    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=seed,
        shuffle=False  # Don't shuffle validation data
    )
    
    logger.info(f"Training generator: {train_generator.samples} samples")
    logger.info(f"Validation generator: {validation_generator.samples} samples")
    logger.info(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, validation_generator

def create_test_generator(
    test_dir: str,
    batch_size: int = 8,
    target_size: Tuple[int, int] = (128, 128)
) -> ImageDataGenerator:
    """
    Create test data generator without augmentation.
    
    Args:
        test_dir: Directory containing test data
        batch_size: Batch size for testing
        target_size: Target image dimensions
        
    Returns:
        Test data generator
    """
    logger.info(f"Creating test generator for directory: {test_dir}")
    
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important: don't shuffle test data
    )
    
    logger.info(f"Test generator: {test_generator.samples} samples")
    return test_generator

class CustomDataSequence(Sequence):
    """
    Custom data sequence for more control over data loading and preprocessing.
    """
    
    def __init__(self, 
                 image_paths: list, 
                 labels: list, 
                 batch_size: int = 8,
                 target_size: Tuple[int, int] = (128, 128),
                 augment: bool = True):
        """
        Initialize custom data sequence.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Batch size
            target_size: Target image dimensions
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.preprocessor = ImagePreprocessor(target_size)
        
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data."""
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.image_paths))
        
        batch_paths = self.image_paths[batch_start:batch_end]
        batch_labels = self.labels[batch_start:batch_end]
        
        # Load and preprocess images
        batch_images = []
        for path in batch_paths:
            try:
                image = self.preprocessor.preprocess_image(path)
                batch_images.append(image[0])  # Remove batch dimension
            except Exception as e:
                logger.warning(f"Error loading image {path}: {e}")
                # Create dummy image as fallback
                dummy_image = np.zeros((*self.target_size, 3), dtype=np.float32)
                batch_images.append(dummy_image)
        
        return np.array(batch_images), np.array(batch_labels, dtype=np.float32)

def validate_data_directory(data_dir: str) -> dict:
    """
    Validate data directory structure and count samples.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": False,
        "total_samples": 0,
        "classes": {},
        "issues": []
    }
    
    try:
        if not os.path.exists(data_dir):
            validation_results["issues"].append(f"Directory does not exist: {data_dir}")
            return validation_results
        
        # Check for class subdirectories
        subdirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(subdirs) == 0:
            validation_results["issues"].append("No class subdirectories found")
            return validation_results
        
        # Count samples in each class
        total_samples = 0
        for subdir in subdirs:
            class_path = os.path.join(data_dir, subdir)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            class_count = len(image_files)
            validation_results["classes"][subdir] = class_count
            total_samples += class_count
            
            if class_count == 0:
                validation_results["issues"].append(f"No images found in class: {subdir}")
        
        validation_results["total_samples"] = total_samples
        validation_results["valid"] = len(validation_results["issues"]) == 0
        
        logger.info(f"Data validation completed for {data_dir}")
        logger.info(f"Total samples: {total_samples}, Classes: {validation_results['classes']}")
        
    except Exception as e:
        validation_results["issues"].append(f"Error during validation: {e}")
        logger.error(f"Error validating data directory: {e}")
    
    return validation_results

# Preprocessing configuration constants
PREPROCESSING_CONFIG = {
    "target_size": (128, 128),
    "batch_size": 8,
    "validation_split": 0.2,
    "augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "brightness_range": [0.8, 1.2],
        "zoom_range": 0.1,
        "horizontal_flip": True
    }
}

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test image preprocessor
    preprocessor = ImagePreprocessor()
    print("✅ ImagePreprocessor initialized successfully")
    
    # Test data directory validation
    if os.path.exists("../data/train"):
        results = validate_data_directory("../data/train")
        print(f"📊 Data validation results: {results}")
    else:
        print("⚠️ Training data directory not found") 