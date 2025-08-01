#!/usr/bin/env python3
"""
Generate synthetic training data for malnutrition detection.
Based on medical research showing visual indicators of malnutrition in children.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import cv2
from pathlib import Path

def create_directories():
    """Create necessary directories for training data."""
    base_path = Path("data")
    directories = [
        "train/normal",
        "train/malnourished", 
        "test/normal",
        "test/malnourished"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created training data directories")

def generate_face_features(is_malnourished=False):
    """Generate facial features based on malnutrition indicators."""
    # Base face parameters
    face_width = random.randint(180, 220)
    face_height = random.randint(200, 240)
    
    if is_malnourished:
        # Malnourished characteristics from medical research
        skin_tone = random.choice([
            (160, 140, 120),  # Pale/grayish
            (140, 120, 100),  # Yellowish (jaundice-like)
            (120, 100, 90),   # Dark with poor circulation
        ])
        
        # Sunken features
        cheek_depression = random.randint(15, 25)
        eye_socket_depth = random.randint(10, 20)
        
        # Hair characteristics (sparse, discolored)
        hair_density = random.uniform(0.3, 0.6)
        hair_color = random.choice([
            (160, 140, 100),  # Discolored/reddish
            (120, 100, 80),   # Sparse brown
            (100, 90, 80),    # Thin gray
        ])
        
        # Facial proportions (wider forehead, smaller chin)
        forehead_ratio = random.uniform(1.1, 1.3)
        chin_ratio = random.uniform(0.7, 0.9)
        
    else:
        # Normal/healthy characteristics
        skin_tone = random.choice([
            (200, 180, 160),  # Healthy pink
            (180, 160, 140),  # Healthy brown
            (160, 140, 120),  # Healthy tan
            (220, 200, 180),  # Light healthy
        ])
        
        # Normal features
        cheek_depression = random.randint(5, 10)
        eye_socket_depth = random.randint(3, 8)
        
        # Healthy hair
        hair_density = random.uniform(0.8, 1.0)
        hair_color = random.choice([
            (80, 60, 40),     # Dark brown
            (120, 100, 60),   # Light brown
            (40, 30, 20),     # Black
            (160, 140, 100),  # Blonde
        ])
        
        # Normal proportions
        forehead_ratio = random.uniform(0.9, 1.1)
        chin_ratio = random.uniform(0.9, 1.1)
    
    return {
        'face_width': face_width,
        'face_height': face_height,
        'skin_tone': skin_tone,
        'cheek_depression': cheek_depression,
        'eye_socket_depth': eye_socket_depth,
        'hair_density': hair_density,
        'hair_color': hair_color,
        'forehead_ratio': forehead_ratio,
        'chin_ratio': chin_ratio,
        'is_malnourished': is_malnourished
    }

def create_synthetic_face(features):
    """Create a synthetic face image based on features."""
    # Create base image
    img_size = (224, 224)
    image = Image.new('RGB', img_size, (240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Face oval
    face_left = (img_size[0] - features['face_width']) // 2
    face_top = (img_size[1] - features['face_height']) // 2
    face_right = face_left + features['face_width']
    face_bottom = face_top + features['face_height']
    
    # Draw face shape
    draw.ellipse([face_left, face_top, face_right, face_bottom], 
                 fill=features['skin_tone'])
    
    # Eyes
    eye_y = face_top + features['face_height'] // 3
    left_eye_x = face_left + features['face_width'] // 4
    right_eye_x = face_right - features['face_width'] // 4
    
    eye_size = 15 if not features['is_malnourished'] else 12
    
    # Draw eyes (sunken if malnourished)
    if features['is_malnourished']:
        # Sunken eyes
        draw.ellipse([left_eye_x-eye_size, eye_y-eye_size-features['eye_socket_depth'], 
                     left_eye_x+eye_size, eye_y+eye_size-features['eye_socket_depth']], 
                     fill=(50, 50, 50))
        draw.ellipse([right_eye_x-eye_size, eye_y-eye_size-features['eye_socket_depth'],
                     right_eye_x+eye_size, eye_y+eye_size-features['eye_socket_depth']], 
                     fill=(50, 50, 50))
    else:
        # Normal eyes
        draw.ellipse([left_eye_x-eye_size, eye_y-eye_size, 
                     left_eye_x+eye_size, eye_y+eye_size], fill=(50, 50, 50))
        draw.ellipse([right_eye_x-eye_size, eye_y-eye_size,
                     right_eye_x+eye_size, eye_y+eye_size], fill=(50, 50, 50))
    
    # Nose
    nose_x = face_left + features['face_width'] // 2
    nose_y = face_top + features['face_height'] // 2
    nose_width = 8 if not features['is_malnourished'] else 6
    
    draw.ellipse([nose_x-nose_width, nose_y-nose_width//2,
                 nose_x+nose_width, nose_y+nose_width*2], 
                 fill=tuple(c-20 for c in features['skin_tone']))
    
    # Mouth
    mouth_y = face_top + features['face_height'] * 2 // 3
    mouth_width = 20 if not features['is_malnourished'] else 15
    
    draw.ellipse([nose_x-mouth_width, mouth_y-5,
                 nose_x+mouth_width, mouth_y+5], fill=(150, 100, 100))
    
    # Cheeks (depression for malnourished)
    if features['is_malnourished']:
        # Sunken cheeks
        left_cheek_x = face_left + features['face_width'] // 6
        right_cheek_x = face_right - features['face_width'] // 6
        cheek_y = nose_y + 10
        
        # Draw shadow for sunken appearance
        shadow_color = tuple(max(0, c-40) for c in features['skin_tone'])
        draw.ellipse([left_cheek_x-15, cheek_y-10, left_cheek_x+15, cheek_y+20], 
                     fill=shadow_color)
        draw.ellipse([right_cheek_x-15, cheek_y-10, right_cheek_x+15, cheek_y+20], 
                     fill=shadow_color)
    
    # Hair (sparse if malnourished)
    hair_density = int(features['hair_density'] * 100)
    for _ in range(hair_density):
        hair_x = random.randint(face_left, face_right)
        hair_y = random.randint(face_top-20, face_top+20)
        hair_size = random.randint(2, 5)
        
        draw.ellipse([hair_x-hair_size, hair_y-hair_size,
                     hair_x+hair_size, hair_y+hair_size], 
                     fill=features['hair_color'])
    
    # Add noise and variations
    img_array = np.array(image)
    
    # Add some noise
    noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur for realism
    image = Image.fromarray(img_array)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Adjust brightness and contrast based on health status
    if features['is_malnourished']:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.8)  # Slightly darker
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.9)  # Lower contrast
    else:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)  # Slightly brighter
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)  # Higher contrast
    
    return image

def generate_dataset(num_samples_per_class=500):
    """Generate complete dataset with realistic variations."""
    print("🎨 Generating synthetic training data...")
    
    # Training data
    print("📚 Creating training data...")
    for class_name, is_malnourished in [("normal", False), ("malnourished", True)]:
        print(f"   Creating {num_samples_per_class} {class_name} samples...")
        
        for i in range(num_samples_per_class):
            features = generate_face_features(is_malnourished)
            image = create_synthetic_face(features)
            
            # Save image
            filename = f"{class_name}_{i:04d}.jpg"
            filepath = f"data/train/{class_name}/{filename}"
            image.save(filepath, quality=95)
            
            if (i + 1) % 100 == 0:
                print(f"      ✓ Generated {i + 1}/{num_samples_per_class} {class_name} images")
    
    # Test data (smaller set)
    test_samples = num_samples_per_class // 5
    print(f"🧪 Creating test data ({test_samples} per class)...")
    
    for class_name, is_malnourished in [("normal", False), ("malnourished", True)]:
        print(f"   Creating {test_samples} {class_name} test samples...")
        
        for i in range(test_samples):
            features = generate_face_features(is_malnourished)
            image = create_synthetic_face(features)
            
            # Save image
            filename = f"{class_name}_test_{i:04d}.jpg"
            filepath = f"data/test/{class_name}/{filename}"
            image.save(filepath, quality=95)
    
    print("✅ Dataset generation completed!")
    print(f"📊 Generated:")
    print(f"   - Training: {num_samples_per_class * 2} images ({num_samples_per_class} per class)")
    print(f"   - Testing: {test_samples * 2} images ({test_samples} per class)")

def create_sample_real_images():
    """Create a few sample 'real' images by downloading from research datasets."""
    print("🌐 Note: For production use, consider downloading real datasets from:")
    print("   - https://www.kaggle.com/code/masterofall/notebook1ed813e60a")
    print("   - https://yanweifu.github.io/FG_NET_data/")
    print("   - WHO medical image databases")
    print("   - Research institutions with ethical approval")

if __name__ == "__main__":
    print("🚀 Starting malnutrition detection dataset creation...")
    
    # Create directories
    create_directories()
    
    # Generate synthetic dataset
    generate_dataset(num_samples_per_class=300)  # Smaller for demo
    
    # Show info about real datasets
    create_sample_real_images()
    
    print("\n🎉 Training data creation completed!")
    print("📁 Data structure:")
    print("   data/")
    print("   ├── train/")
    print("   │   ├── normal/     (300 images)")
    print("   │   └── malnourished/ (300 images)")
    print("   └── test/")
    print("       ├── normal/     (60 images)")
    print("       └── malnourished/ (60 images)") 