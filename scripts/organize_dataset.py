#!/usr/bin/env python3
"""
Organize the downloaded Roboflow malnutrition dataset into proper class folders.
Reads _classes.csv files and moves images to normal/malnourished/overnourished folders.
"""

import pandas as pd
import shutil
from pathlib import Path
import os

def organize_dataset():
    """Organize images into class-based directory structure."""
    
    print("🗂️  Organizing Malnutrition Dataset")
    print("=" * 40)
    
    # Create class directories
    for split in ['train', 'test']:
        for class_name in ['normal', 'malnourished', 'overnourished']:
            class_dir = Path(f"data/{split}/{class_name}")
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and test splits
    for split in ['train', 'test']:
        csv_file = Path(f"data/{split}/_classes.csv")
        
        if not csv_file.exists():
            print(f"⚠️  No CSV file found for {split} split")
            continue
            
        print(f"\n📊 Processing {split} split...")
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        print(f"   Found {len(df)} images")
        
        # Count classes
        normal_count = 0
        malnourished_count = 0 
        overnourished_count = 0
        
        # Process each image
        for _, row in df.iterrows():
            filename = row['filename']
            source_path = Path(f"data/{split}/{filename}")
            
            if not source_path.exists():
                print(f"   ⚠️  Missing file: {filename}")
                continue
            
            # Determine class based on labels
            if row['malnourished'] == 1:
                class_name = 'malnourished'
                malnourished_count += 1
            elif row['overnourished'] == 1:
                class_name = 'overnourished'
                overnourished_count += 1
            else:
                class_name = 'normal'
                normal_count += 1
            
            # Move to class folder
            target_dir = Path(f"data/{split}/{class_name}")
            target_path = target_dir / filename
            
            if not target_path.exists():
                shutil.move(str(source_path), str(target_path))
        
        print(f"   ✅ {split} split organized:")
        print(f"      📗 Normal: {normal_count}")
        print(f"      📕 Malnourished: {malnourished_count}")
        print(f"      📘 Overnourished: {overnourished_count}")
    
    # Clean up - move CSV files to backup
    backup_dir = Path("data/backup")
    backup_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'test']:
        csv_file = Path(f"data/{split}/_classes.csv")
        if csv_file.exists():
            shutil.move(str(csv_file), str(backup_dir / f"{split}_classes.csv"))
    
    print(f"\n🎉 Dataset organization complete!")
    print(f"📁 Check organized structure:")
    print(f"   ls -la data/train/*/")
    print(f"   ls -la data/test/*/")

def check_organization():
    """Check the final organization of the dataset."""
    
    print("\n📋 Final Dataset Structure:")
    print("=" * 30)
    
    total_images = 0
    
    for split in ['train', 'test']:
        print(f"\n{split.upper()} SET:")
        split_total = 0
        
        for class_name in ['normal', 'malnourished', 'overnourished']:
            class_dir = Path(f"data/{split}/{class_name}")
            if class_dir.exists():
                count = len(list(class_dir.glob("*.jpg")))
                print(f"  {class_name:12}: {count:3d} images")
                split_total += count
            else:
                print(f"  {class_name:12}:   0 images (no folder)")
        
        print(f"  {'TOTAL':12}: {split_total:3d} images")
        total_images += split_total
    
    print(f"\n🎯 GRAND TOTAL: {total_images} images")
    
    # Calculate class distribution 
    print(f"\n📊 Class Distribution Analysis:")
    
    all_normal = len(list(Path("data/train/normal").glob("*.jpg"))) + len(list(Path("data/test/normal").glob("*.jpg")))
    all_malnourished = len(list(Path("data/train/malnourished").glob("*.jpg"))) + len(list(Path("data/test/malnourished").glob("*.jpg")))
    all_overnourished = len(list(Path("data/train/overnourished").glob("*.jpg"))) + len(list(Path("data/test/overnourished").glob("*.jpg")))
    
    total = all_normal + all_malnourished + all_overnourished
    
    if total > 0:
        print(f"   Normal:        {all_normal:3d} ({all_normal/total*100:.1f}%)")
        print(f"   Malnourished:  {all_malnourished:3d} ({all_malnourished/total*100:.1f}%)")  
        print(f"   Overnourished: {all_overnourished:3d} ({all_overnourished/total*100:.1f}%)")

if __name__ == "__main__":
    organize_dataset()
    check_organization() 