"""
Custom model architecture for malnutrition detection using MobileNetV2.
This module defines the CNN model structure used throughout the pipeline.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def create_malnutrition_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 1,
    learning_rate: float = 0.0001,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create optimized MobileNetV2-based model for malnutrition detection.
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model ready for training
    """
    logger.info(f"Creating MobileNetV2 model with input shape: {input_shape}")
    
    # Load pre-trained MobileNetV2 base
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        alpha=0.75  # Width multiplier for efficiency
    )
    
    # Freeze base layers initially
    base_model.trainable = False
    logger.info(f"Base model loaded with {len(base_model.layers)} layers")
    
    # Create custom head architecture
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        # First dense layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Second dense layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid', name='predictions')
    ], name='malnutrition_detector')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info("Model compiled successfully")
    return model

def fine_tune_model(model: keras.Model, learning_rate: float = 0.00001) -> keras.Model:
    """
    Enable fine-tuning of the pre-trained layers.
    
    Args:
        model: Compiled model to fine-tune
        learning_rate: Lower learning rate for fine-tuning
        
    Returns:
        Model ready for fine-tuning
    """
    logger.info("Enabling fine-tuning of pre-trained layers")
    
    # Unfreeze the base model for fine-tuning
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Fine-tune from this layer onwards (last 20 layers)
    fine_tune_at = len(base_model.layers) - 20
    
    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info(f"Fine-tuning enabled for last {len(base_model.layers) - fine_tune_at} layers")
    return model

def get_training_callbacks(
    model_path: str,
    patience: int = 5,
    factor: float = 0.5,
    min_lr: float = 1e-7
) -> list:
    """
    Get training callbacks for model optimization.
    
    Args:
        model_path: Path to save best model
        patience: Patience for early stopping
        factor: Factor for learning rate reduction
        min_lr: Minimum learning rate
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    logger.info(f"Training callbacks configured, model will be saved to: {model_path}")
    return callbacks

def load_trained_model(model_path: str) -> keras.Model:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def get_model_summary(model: keras.Model) -> dict:
    """
    Get comprehensive model summary information.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(layer) for layer in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "model_layers": len(model.layers),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape
    }

# Model configuration constants
MODEL_CONFIG = {
    "input_shape": (128, 128, 3),
    "learning_rate": 0.0001,
    "fine_tune_learning_rate": 0.00001,
    "dropout_rate": 0.5,
    "batch_size": 8,
    "epochs": 50,
    "patience": 5
}

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create and display model
    model = create_malnutrition_model()
    print("\n📊 Model Architecture Summary:")
    model.summary()
    
    # Display model configuration
    summary = get_model_summary(model)
    print(f"\n🔧 Model Configuration:")
    for key, value in summary.items():
        print(f"  {key}: {value}") 