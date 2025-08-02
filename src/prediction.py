"""
Prediction module for child malnutrition detection.
Supports 3-class classification using confidence thresholds.
Optimized for MobileNetV2 model with 128x128 input images.
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Union
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalnutritionPredictor:
    """Predictor for malnutrition detection with 3-class output using confidence thresholds."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.80):
        """
        Initialize the malnutrition predictor.
        
        Args:
            model_path: Path to the trained model file
            confidence_threshold: Threshold for 3-class classification (default: 0.80)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = ['malnourished', 'overnourished', 'normal']
        self.binary_classes = ['malnourished', 'overnourished']
        
        # Load the model on initialization
        if not self.load_model():
            raise ValueError(f"Failed to load model from {model_path}")
        
    def load_model(self) -> bool:
        """Load the trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
        """
        Preprocess image for prediction (optimized for MobileNet 128x128).
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing (default 128x128 for MobileNet)
            
        Returns:
            Preprocessed image array or None if error
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, target_size)
            
            # Normalize to [0, 1] range
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def _classify_with_confidence(self, prob_malnourished: float, prob_overnourished: float) -> Tuple[str, float]:
        """
        Classify using confidence thresholds for 3-class output.
        Args:
            prob_malnourished: Probability of being malnourished
            prob_overnourished: Probability of being overnourished
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if prob_malnourished >= self.confidence_threshold:
            return "malnourished", prob_malnourished
        elif prob_overnourished >= self.confidence_threshold:
            return "overnourished", prob_overnourished
        else:
            # Both probabilities are < 0.80 - classify as normal
            # This represents the "uncertainty zone" where child appears normal
            max_prob = max(prob_malnourished, prob_overnourished)
            normal_confidence = 1.0 - max_prob  # Higher when model is more uncertain
            return "normal", normal_confidence

    def predict_single(self, image_input: Union[str, bytes]) -> Dict:
        """
        Predict malnutrition class for a single image.
        
        Args:
            image_input: Either a path to the image file (str) or image bytes (bytes)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Handle both file path and image bytes
            if isinstance(image_input, str):
                # File path provided
                processed_image = self.preprocess_image(image_input)
                image_identifier = image_input
            elif isinstance(image_input, bytes):
                # Image bytes provided (from API upload)
                processed_image = self._preprocess_image_bytes(image_input)
                image_identifier = "uploaded_image"
            else:
                return {
                    'error': 'Invalid image input type. Expected file path (str) or image bytes (bytes)',
                    'input_type': str(type(image_input))
                }
            
            if processed_image is None:
                return {
                    'error': 'Failed to preprocess image',
                    'image_identifier': image_identifier
                }
            
            # Get model prediction
            prediction = self.model.predict(processed_image, verbose=0)[0][0]
            
            # Convert to class probabilities
            prob_overnourished = float(prediction)
            prob_malnourished = 1.0 - prob_overnourished
            
            # Debug logging
            logger.info(f"Raw prediction: {prediction:.4f}")
            logger.info(f"Probabilities - Malnourished: {prob_malnourished:.4f}, Overnourished: {prob_overnourished:.4f}")
            
            # Apply confidence-based classification
            predicted_class, confidence = self._classify_with_confidence(
                prob_malnourished, prob_overnourished
            )
            
            logger.info(f"Final classification: {predicted_class} (confidence: {confidence:.4f}, threshold: {self.confidence_threshold})")
            
            # Get interpretation and recommendation
            interpretation = self._get_interpretation(predicted_class, confidence)
            recommendation = self._get_recommendation(predicted_class)
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'malnourished': float(prob_malnourished),
                    'overnourished': float(prob_overnourished),
                    'normal': float(1.0 - max(prob_malnourished, prob_overnourished))
                },
                'interpretation': interpretation,
                'recommendation': recommendation,
                'confidence_threshold': self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'image_identifier': image_identifier if 'image_identifier' in locals() else 'unknown'
            }

    def _preprocess_image_bytes(self, image_bytes: bytes, target_size: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
        """
        Preprocess image from bytes for model input.
        
        Args:
            image_bytes: Raw image bytes
            target_size: Target size for the image
            
        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image bytes: {str(e)}")
            return None

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict malnutrition classes for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict_single(image_path)
            results.append(result)
        return results

    def _get_interpretation(self, predicted_class: str, confidence: float) -> str:
        """Get human-readable interpretation of the prediction."""
        interpretations = {
            'malnourished': f"Child shows signs of malnutrition (confidence: {confidence:.2f})",
            'overnourished': f"Child shows signs of overnutrition (confidence: {confidence:.2f})",
            'normal': f"Child appears to have normal nutritional status (confidence: {confidence:.2f})"
        }
        return interpretations.get(predicted_class, "Unknown classification")

    def _get_recommendation(self, predicted_class: str) -> str:
        """Get actionable recommendation based on prediction."""
        recommendations = {
            'malnourished': "Immediate medical evaluation recommended. Assess for underlying causes and initiate appropriate nutritional intervention.",
            'overnourished': "Nutritional counseling recommended. Consider dietary modifications and increased physical activity.",
            'normal': "Continue current nutritional practices. Regular monitoring recommended."
        }
        return recommendations.get(predicted_class, "Consult healthcare professional for guidance.")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'error': 'No model loaded'}
        
        try:
            return {
                'model_path': self.model_path,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'total_parameters': self.model.count_params(),
                'confidence_threshold': self.confidence_threshold,
                'supported_classes': self.class_names
            }
        except Exception as e:
            return {'error': f'Failed to get model info: {str(e)}'}

    def set_confidence_threshold(self, threshold: float) -> bool:
        """
        Update the confidence threshold for classification.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold}")
            return True
        else:
            logger.error(f"Invalid threshold {threshold}. Must be between 0.0 and 1.0")
            return False


def create_predictor(model_path: str, confidence_threshold: float = 0.80) -> MalnutritionPredictor:
    """
    Factory function to create a MalnutritionPredictor instance.
    
    Args:
        model_path: Path to the trained model file
        confidence_threshold: Confidence threshold for classification (default: 0.80)
        
    Returns:
        MalnutritionPredictor instance
    """
    return MalnutritionPredictor(model_path, confidence_threshold)


if __name__ == "__main__":
    # Example usage
    model_path = "../models/malnutrition_model.h5"
    
    if os.path.exists(model_path):
        predictor = create_predictor(model_path, confidence_threshold=0.80)
        
        # Example prediction
        test_image = "../data/test/malnourished/malnourished-338_jpg.rf.e4084a36394a8785ffb48a82c7873a81.jpg"
        if os.path.exists(test_image):
            result = predictor.predict_single(test_image)
            print("🔍 Prediction Result:")
            print(f"   Image: {result.get('image_path', 'N/A')}")
            print(f"   Class: {result.get('predicted_class', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Interpretation: {result.get('interpretation', 'N/A')}")
        else:
            print(f"Test image not found: {test_image}")
    else:
        print(f"Model not found: {model_path}")
        print("Please run the notebook to train the model first.") 