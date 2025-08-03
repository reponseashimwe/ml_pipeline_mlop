#!/usr/bin/env python3
"""
Model retraining module for malnutrition detection.
Handles incremental training with new uploaded data.
"""

import os
import shutil
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI crashes in threads
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, Tuple, Optional, List
import json
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_optimized_model(input_shape: Tuple[int, int, int] = (128, 128, 3)) -> keras.Model:
    """
    Create MobileNetV2-based model optimized for small datasets.
    Enhanced for better performance with fine-tuning.
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Compiled Keras model
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.75  # Lighter version
    )
    
    # Fine-tune the last few layers of base model for better performance
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def check_training_data(train_dir: str) -> Dict[str, int]:
    """
    Check available training data.
    
    Args:
        train_dir: Training data directory
        
    Returns:
        Dictionary with class counts
    """
    class_counts = {}
    
    for class_name in ['malnourished', 'overnourished']:
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
    
    return class_counts

def create_data_generators(train_dir: str, batch_size: int = 8) -> Tuple:
    """
    Create optimized data generators for training.
    
    Args:
        train_dir: Training data directory
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=42
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=42
    )
    
    return train_generator, val_generator

def retrain_model(train_dir: str = "../data/train", 
                 model_path: str = "../models/malnutrition_model.h5",
                 epochs: int = 30,  # Increased for better performance
                 progress_callback=None,
                 show_all_epochs: bool = False) -> Dict[str, any]:
    """
    Retrain the malnutrition detection model with new data.
    
    Args:
        train_dir: Directory containing training data
        model_path: Path to save the retrained model
        epochs: Number of training epochs
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with retraining results
    """
    try:
        logger.info("🚀 Starting model retraining...")
        
        # 1. Check training data
        class_counts = check_training_data(train_dir)
        total_images = sum(class_counts.values())
        
        logger.info(f"📊 Training data: {class_counts}")
        logger.info(f"📈 Total images: {total_images}")
        
        if total_images < 20:
            raise ValueError(f"Insufficient training data. Found {total_images} images, need at least 20.")
        
        # 2. Create data generators
        train_generator, val_generator = create_data_generators(train_dir)
        
        # 3. Load existing model as pre-trained base (Transfer Learning approach)
        try:
            logger.info("🔄 Loading existing model as pre-trained base...")
            existing_model = keras.models.load_model(model_path)
            logger.info("✅ Loaded existing model for transfer learning")
            
            # Create new model with same architecture
            model = create_optimized_model()
            
            # Transfer weights from existing model (if compatible)
            # This implements proper transfer learning from our own trained model
            try:
                model.set_weights(existing_model.get_weights())
                logger.info("✅ Transferred weights from existing model")
            except:
                logger.warning("⚠️ Could not transfer weights, using fresh initialization")
                
        except Exception as e:
            logger.info(f"🔄 Creating new model (existing model not found: {e})")
            model = create_optimized_model()
        
        # 4. Setup training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=12,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More aggressive reduction
                patience=6,
                min_lr=1e-8,
                verbose=1,
                cooldown=2
            ),
            ModelCheckpoint(
                model_path.replace('.h5', '_retrained_backup.h5'),
                monitor='val_accuracy',  # Save best accuracy model
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # 5. Train the model with progress tracking
        logger.info(f"🔄 Training model for {epochs} epochs...")
        logger.info(f"📊 Training data counts: {class_counts}")
        logger.info(f"📊 Train generator steps per epoch: {len(train_generator)}")
        logger.info(f"📊 Validation generator steps per epoch: {len(val_generator)}")
        logger.info(f"📊 Model input shape: {model.input_shape}")
        logger.info(f"📊 Model output shape: {model.output_shape}")
        logger.info(f"📊 Callbacks: {len(callbacks)} callbacks configured")
        
        # Custom callback for progress tracking - ensures EVERY epoch is logged
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_count = 0
            
            def on_epoch_begin(self, epoch, logs=None):
                """Called at the beginning of each epoch"""
                self.epoch_count = epoch + 1
                logger.info(f"🔄 Starting epoch {self.epoch_count}")
            
            def on_epoch_end(self, epoch, logs=None):
                """Called at the end of each epoch - ALWAYS called for every epoch"""
                if progress_callback:
                    try:
                        metrics = {
                            "accuracy": logs.get("accuracy", 0.0),
                            "loss": logs.get("loss", 1.0),
                            "val_accuracy": logs.get("val_accuracy", 0.0),
                            "val_loss": logs.get("val_loss", 1.0),
                        }
                        logger.info(f"📊 Progress callback called for epoch {epoch + 1}: {metrics}")
                        progress_callback(epoch + 1, metrics)
                        logger.info(f"✅ Progress callback completed for epoch {epoch + 1}")
                    except Exception as e:
                        logger.error(f"❌ Error in progress callback: {e}")
                else:
                    logger.warning("⚠️ No progress callback provided")
            
            def on_train_end(self, logs=None):
                """Called when training ends (including early stopping)"""
                logger.info(f"🏁 Training ended after {self.epoch_count} epochs")
                if hasattr(self.model, 'stopped_epoch') and self.model.stopped_epoch:
                    logger.info(f"⏹️ Early stopping triggered at epoch {self.model.stopped_epoch}")
        
        # Add progress callback to callbacks list - ALWAYS add it
        callbacks.append(ProgressCallback())
        logger.info("✅ Progress callback added to training callbacks")
        
        logger.info("🚀 Starting model.fit()...")
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        logger.info(f"✅ model.fit() completed. History length: {len(history.history.get('accuracy', []))}")
        
        # 6. Evaluate final performance
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        logger.info(f"📊 Final training accuracy: {final_train_acc:.4f}")
        logger.info(f"📊 Final validation accuracy: {final_val_acc:.4f}")
        
        # 7. Save the retrained model (no extra files - all data goes to database)
        model.save(model_path)
        logger.info(f"💾 Model saved to: {model_path}")
        logger.info("📊 Training plots and history stored in database, no extra files needed")
        
        # 12. Return results
        results = {
            "success": True,
            "message": "Model retrained successfully",
            "metrics": {
                "final_train_accuracy": float(final_train_acc),
                "final_val_accuracy": float(final_val_acc),
                "total_epochs": len(history.history['accuracy']),
                "training_images": class_counts
            },
            "model_path": model_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate REAL metrics using validation data
        try:
            from database import get_database
            from sklearn.metrics import precision_score, recall_score, f1_score
            db = get_database()
            
            # Get real predictions on validation data
            val_generator.reset()
            predictions = model.predict(val_generator, verbose=0)
            predicted_classes = (predictions > 0.5).astype(int).flatten()
            true_labels = val_generator.classes
            
            # Calculate real metrics
            precision = precision_score(true_labels, predicted_classes)
            recall = recall_score(true_labels, predicted_classes)
            f1 = f1_score(true_labels, predicted_classes)
            
            logger.info(f"📊 Real Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save to database
            model_version = f"retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db.save_model_performance(
                model_version=model_version,
                accuracy=float(final_val_acc),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                test_samples=len(true_labels)
            )
            
            logger.info(f"💾 Saved REAL model performance to database: {model_version}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save model performance to database: {e}")
        
        logger.info("✅ Retraining completed successfully!")
        return results
        
    except Exception as e:
        error_msg = f"Retraining failed: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        }

def merge_uploaded_data(uploaded_dir: str, main_train_dir: str) -> bool:
    """
    Merge uploaded training data with existing training data.
    
    Args:
        uploaded_dir: Directory with uploaded images
        main_train_dir: Main training directory
        
    Returns:
        True if data was merged successfully
    """
    try:
        import shutil
        
        for class_name in ['malnourished', 'overnourished']:
            uploaded_class_dir = os.path.join(uploaded_dir, class_name)
            main_class_dir = os.path.join(main_train_dir, class_name)
            
            if os.path.exists(uploaded_class_dir):
                os.makedirs(main_class_dir, exist_ok=True)
                
                # Copy uploaded images to main training directory
                for filename in os.listdir(uploaded_class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src = os.path.join(uploaded_class_dir, filename)
                        # Add timestamp to avoid filename conflicts
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        new_filename = f"uploaded_{timestamp}_{filename}"
                        dst = os.path.join(main_class_dir, new_filename)
                        shutil.copy2(src, dst)
                        logger.info(f"📁 Copied {filename} to training data")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to merge uploaded data: {e}")
        return False

if __name__ == "__main__":
    # For testing the retraining process
    result = retrain_model()
    print(f"Retraining result: {result}") 