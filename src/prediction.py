"""
Prediction module for child malnutrition detection.
Handles model predictions and result processing.
"""

import os
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import cv2
from PIL import Image
import io
import base64

# Import our modules
from .preprocessing import ImagePreprocessor
from .model import MalnutritionCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalnutritionPredictor:
    """Handles predictions for malnutrition detection."""
    
    def __init__(self, model_path: str = "models/malnutrition_model.pkl"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor()
        self.class_names = ['Normal', 'Malnourished']
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            # Create model instance
            self.model = MalnutritionCNN()
            
            # Load the trained model
            success = self.model.load_model(self.model_path)
            if success:
                logger.info("Model loaded successfully")
                return True
            else:
                logger.error("Failed to load model")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Predict malnutrition status for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the image
            processed_image = self.preprocessor.preprocess_single_image(image_path)
            if processed_image is None:
                return {
                    'success': False,
                    'error': 'Failed to process image'
                }
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Extract features for analysis
            features = self.preprocessor.extract_features(processed_image[0])
            
            result = {
                'success': True,
                'prediction': {
                    'class': self.class_names[predicted_class],
                    'class_id': int(predicted_class),
                    'confidence': confidence,
                    'probabilities': {
                        'normal': float(predictions[0][0]),
                        'malnourished': float(predictions[0][1])
                    }
                },
                'features': {
                    'color_features': features[:6].tolist(),
                    'texture_features': features[6:8].tolist(),
                    'shape_features': features[8:].tolist()
                },
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
            
            logger.info(f"Prediction completed: {result['prediction']['class']} "
                       f"(confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_batch_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Predict malnutrition status for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary containing batch prediction results
        """
        try:
            results = []
            successful_predictions = 0
            
            for image_path in image_paths:
                result = self.predict_single_image(image_path)
                results.append(result)
                
                if result['success']:
                    successful_predictions += 1
            
            # Calculate batch statistics
            if successful_predictions > 0:
                confidences = [r['prediction']['confidence'] 
                             for r in results if r['success']]
                avg_confidence = np.mean(confidences)
                
                class_counts = {}
                for result in results:
                    if result['success']:
                        class_name = result['prediction']['class']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            else:
                avg_confidence = 0.0
                class_counts = {}
            
            batch_result = {
                'success': True,
                'total_images': len(image_paths),
                'successful_predictions': successful_predictions,
                'failed_predictions': len(image_paths) - successful_predictions,
                'average_confidence': avg_confidence,
                'class_distribution': class_counts,
                'predictions': results,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Batch prediction completed: {successful_predictions}/{len(image_paths)} "
                       f"successful predictions")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_from_base64(self, image_base64: str) -> Dict[str, Any]:
        """
        Predict malnutrition status from base64 encoded image.
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert RGBA to RGB if necessary
            if image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Preprocess image
            processed_image = self.preprocessor.resize_image(image_array)
            processed_image = self.preprocessor.normalize_image(processed_image)
            processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Extract features
            features = self.preprocessor.extract_features(processed_image[0])
            
            result = {
                'success': True,
                'prediction': {
                    'class': self.class_names[predicted_class],
                    'class_id': int(predicted_class),
                    'confidence': confidence,
                    'probabilities': {
                        'normal': float(predictions[0][0]),
                        'malnourished': float(predictions[0][1])
                    }
                },
                'features': {
                    'color_features': features[:6].tolist(),
                    'texture_features': features[6:8].tolist(),
                    'shape_features': features[8:].tolist()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Base64 prediction completed: {result['prediction']['class']} "
                       f"(confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making base64 prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_prediction_interpretation(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide interpretation of prediction results.
        
        Args:
            prediction_result: Result from prediction methods
            
        Returns:
            Dictionary containing interpretation
        """
        if not prediction_result.get('success', False):
            return {'error': 'No valid prediction to interpret'}
        
        prediction = prediction_result['prediction']
        features = prediction_result.get('features', {})
        
        interpretation = {
            'risk_level': self._get_risk_level(prediction['confidence']),
            'recommendation': self._get_recommendation(prediction['class'], prediction['confidence']),
            'feature_analysis': self._analyze_features(features),
            'confidence_interpretation': self._interpret_confidence(prediction['confidence'])
        }
        
        return interpretation
    
    def _get_risk_level(self, confidence: float) -> str:
        """Get risk level based on confidence."""
        if confidence >= 0.9:
            return 'Very High'
        elif confidence >= 0.8:
            return 'High'
        elif confidence >= 0.7:
            return 'Medium'
        elif confidence >= 0.6:
            return 'Low'
        else:
            return 'Very Low'
    
    def _get_recommendation(self, predicted_class: str, confidence: float) -> str:
        """Get recommendation based on prediction."""
        if predicted_class == 'Malnourished':
            if confidence >= 0.8:
                return 'Immediate medical attention recommended'
            elif confidence >= 0.6:
                return 'Medical consultation advised'
            else:
                return 'Further assessment recommended'
        else:
            if confidence >= 0.8:
                return 'Continue regular monitoring'
            else:
                return 'Regular health check-ups recommended'
    
    def _analyze_features(self, features: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze extracted features."""
        analysis = {}
        
        if 'color_features' in features:
            color_features = features['color_features']
            if len(color_features) >= 3:
                # Analyze color distribution
                red_mean, green_mean, blue_mean = color_features[:3]
                if red_mean > green_mean and red_mean > blue_mean:
                    analysis['color_analysis'] = 'Reddish tones detected (may indicate health issues)'
                elif green_mean > red_mean and green_mean > blue_mean:
                    analysis['color_analysis'] = 'Greenish tones detected (normal skin tone)'
                else:
                    analysis['color_analysis'] = 'Balanced color distribution'
        
        if 'texture_features' in features:
            texture_features = features['texture_features']
            if len(texture_features) >= 2:
                edge_density = texture_features[0]
                if edge_density > 50:
                    analysis['texture_analysis'] = 'High texture complexity detected'
                else:
                    analysis['texture_analysis'] = 'Low texture complexity detected'
        
        return analysis
    
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence level."""
        if confidence >= 0.9:
            return 'Very confident prediction'
        elif confidence >= 0.8:
            return 'Confident prediction'
        elif confidence >= 0.7:
            return 'Moderately confident prediction'
        elif confidence >= 0.6:
            return 'Low confidence prediction'
        else:
            return 'Very low confidence prediction'
    
    def save_prediction_result(self, result: Dict[str, Any], output_path: str) -> bool:
        """
        Save prediction result to file.
        
        Args:
            result: Prediction result dictionary
            output_path: Path to save the result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Prediction result saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction result: {str(e)}")
            return False


def create_predictor(model_path: str = "models/malnutrition_model.pkl") -> MalnutritionPredictor:
    """
    Factory function to create a malnutrition predictor.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Configured MalnutritionPredictor instance
    """
    return MalnutritionPredictor(model_path)


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = create_predictor()
    
    # Example prediction
    image_path = "data/test/sample_image.jpg"
    if os.path.exists(image_path):
        result = predictor.predict_single_image(image_path)
        if result['success']:
            print(f"Prediction: {result['prediction']['class']}")
            print(f"Confidence: {result['prediction']['confidence']:.3f}")
            
            # Get interpretation
            interpretation = predictor.get_prediction_interpretation(result)
            print(f"Risk Level: {interpretation['risk_level']}")
            print(f"Recommendation: {interpretation['recommendation']}") 