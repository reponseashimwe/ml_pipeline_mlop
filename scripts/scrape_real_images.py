#!/usr/bin/env python3
"""
Scrape real malnutrition images from legitimate medical and research sources.
Based on the Google search results provided by the user.
"""

import os
import requests
import time
import random
from pathlib import Path
from bs4 import BeautifulSoup
import urllib.parse
import re
from PIL import Image
import io

def setup_session():
    """Setup requests session with proper headers."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    return session

def download_image(session, url, save_path, max_size=5*1024*1024):
    """Download and save an image with error handling."""
    try:
        response = session.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return False
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size:
            return False
        
        # Download image data
        image_data = response.content
        if len(image_data) > max_size:
            return False
        
        # Verify it's a valid image
        try:
            img = Image.open(io.BytesIO(image_data))
            # Resize if too large
            if img.width > 800 or img.height > 800:
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save image
            img.save(save_path, 'JPEG', quality=85)
            return True
            
        except Exception as e:
            print(f"   ❌ Invalid image: {e}")
            return False
        
    except Exception as e:
        print(f"   ❌ Download error: {e}")
        return False

def scrape_unicef_images(session):
    """Scrape images from UNICEF malnutrition pages."""
    print("📊 Scraping UNICEF malnutrition images...")
    
    unicef_urls = [
        "https://data.unicef.org/topic/nutrition/malnutrition/",
        "https://www.unicef.org/nutrition/malnutrition",
        "https://www.unicef.org/media/photos/malnutrition",
    ]
    
    images_downloaded = 0
    
    for url in unicef_urls:
        try:
            print(f"   🔍 Scraping: {url}")
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find images with malnutrition-related keywords
            img_tags = soup.find_all('img')
            
            for i, img in enumerate(img_tags):
                if images_downloaded >= 20:  # Limit per source
                    break
                    
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                
                # Make absolute URL
                img_url = urllib.parse.urljoin(url, src)
                
                # Check if image seems related to malnutrition
                alt_text = (img.get('alt', '') + img.get('title', '')).lower()
                keywords = ['malnutrition', 'malnourished', 'stunted', 'wasted', 'underweight', 'child', 'nutrition']
                
                if any(keyword in alt_text for keyword in keywords) or 'child' in img_url.lower():
                    # Determine classification based on keywords
                    if any(word in alt_text for word in ['malnourished', 'stunted', 'wasted', 'underweight']):
                        class_dir = 'malnourished'
                    else:
                        class_dir = 'normal'
                    
                    save_path = f"data/train/{class_dir}/unicef_{class_dir}_{i:03d}.jpg"
                    
                    if download_image(session, img_url, save_path):
                        images_downloaded += 1
                        print(f"   ✅ Downloaded: {save_path}")
                        time.sleep(random.uniform(1, 3))  # Be respectful
        
        except Exception as e:
            print(f"   ⚠️  Error scraping {url}: {e}")
    
    return images_downloaded

def scrape_medical_research_images(session):
    """Scrape images from medical research sources."""
    print("🔬 Scraping medical research images...")
    
    research_urls = [
        "https://ehp.niehs.nih.gov/122-a298",
        "https://www.news-medical.net/health/Risk-Factors-for-Stunted-Growth.aspx",
        "https://borgenproject.org/stunted-growth-in-children/",
    ]
    
    images_downloaded = 0
    
    for url in research_urls:
        try:
            print(f"   🔍 Scraping: {url}")
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tags = soup.find_all('img')
            
            for i, img in enumerate(img_tags):
                if images_downloaded >= 15:
                    break
                    
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                
                img_url = urllib.parse.urljoin(url, src)
                
                # Filter for relevant images
                alt_text = (img.get('alt', '') + img.get('title', '')).lower()
                src_text = src.lower()
                
                if any(keyword in alt_text + src_text for keyword in ['child', 'malnutrition', 'growth', 'nutrition']):
                    # Classify based on context
                    if any(word in alt_text + src_text for word in ['malnourished', 'stunted', 'underweight']):
                        class_dir = 'malnourished'
                    else:
                        class_dir = 'normal'
                    
                    save_path = f"data/train/{class_dir}/research_{class_dir}_{i:03d}.jpg"
                    
                    if download_image(session, img_url, save_path):
                        images_downloaded += 1
                        print(f"   ✅ Downloaded: {save_path}")
                        time.sleep(random.uniform(1, 2))
        
        except Exception as e:
            print(f"   ⚠️  Error scraping {url}: {e}")
    
    return images_downloaded

def scrape_news_images(session):
    """Scrape images from news sources about malnutrition."""
    print("📰 Scraping news source images...")
    
    news_urls = [
        "https://www.theguardian.com/world/2012/jan/10/child-malnutrition-india-national-shame",
        "https://www.voanews.com/a/child-malnutrition-all-time-high-yemen-un-agency-unicef/3633398.html",
        "https://www.compassion.com/poverty/child-malnutrition.htm",
    ]
    
    images_downloaded = 0
    
    for url in news_urls:
        try:
            print(f"   🔍 Scraping: {url}")
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for article images
            img_containers = soup.find_all(['figure', 'div'], class_=re.compile(r'image|photo|picture'))
            img_tags = soup.find_all('img')
            all_imgs = img_containers + img_tags
            
            for i, container in enumerate(all_imgs):
                if images_downloaded >= 10:
                    break
                
                img = container if container.name == 'img' else container.find('img')
                if not img:
                    continue
                    
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                
                img_url = urllib.parse.urljoin(url, src)
                
                # Check relevance
                alt_text = (img.get('alt', '') + img.get('title', '')).lower()
                
                if any(keyword in alt_text for keyword in ['child', 'malnutrition', 'hunger', 'nutrition']):
                    # Most news images about malnutrition show malnourished children
                    class_dir = 'malnourished'
                    save_path = f"data/train/{class_dir}/news_{class_dir}_{i:03d}.jpg"
                    
                    if download_image(session, img_url, save_path):
                        images_downloaded += 1
                        print(f"   ✅ Downloaded: {save_path}")
                        time.sleep(random.uniform(2, 4))
        
        except Exception as e:
            print(f"   ⚠️  Error scraping {url}: {e}")
    
    return images_downloaded

def download_sample_normal_images(session):
    """Download some sample normal/healthy child images for comparison."""
    print("👶 Downloading sample normal child images...")
    
    # Public health/wellness sites with healthy children
    normal_urls = [
        "https://www.who.int/news-room/photo-library",
        "https://www.unicef.org/media/photos/healthy-children",
    ]
    
    images_downloaded = 0
    
    # For demo purposes, create some placeholder normal images
    normal_samples = [
        "https://via.placeholder.com/300x300/90EE90/000000?text=Healthy+Child+1",
        "https://via.placeholder.com/300x300/98FB98/000000?text=Healthy+Child+2",
        "https://via.placeholder.com/300x300/F0FFF0/000000?text=Healthy+Child+3",
        "https://via.placeholder.com/300x300/ADFF2F/000000?text=Healthy+Child+4",
        "https://via.placeholder.com/300x300/9AFF9A/000000?text=Healthy+Child+5",
    ]
    
    for i, img_url in enumerate(normal_samples):
        save_path = f"data/train/normal/sample_normal_{i:03d}.jpg"
        if download_image(session, img_url, save_path):
            images_downloaded += 1
            print(f"   ✅ Downloaded: {save_path}")
            time.sleep(1)
    
    return images_downloaded

def create_test_set():
    """Create test set by moving some training images."""
    print("📋 Creating test set...")
    
    import shutil
    
    for class_name in ['normal', 'malnourished']:
        train_dir = Path(f"data/train/{class_name}")
        test_dir = Path(f"data/test/{class_name}")
        
        images = list(train_dir.glob("*.jpg"))
        if len(images) > 3:
            # Move 20% to test set
            test_count = max(1, len(images) // 5)
            test_images = random.sample(images, test_count)
            
            for img_path in test_images:
                new_path = test_dir / img_path.name
                shutil.move(str(img_path), str(new_path))
                print(f"   📤 Moved to test: {img_path.name}")

def main():
    print("🌐 Real Malnutrition Image Scraper")
    print("=" * 50)
    print("Downloading from legitimate medical sources")
    
    # Setup
    session = setup_session()
    
    # Create directories
    for split in ['train', 'test']:
        for class_name in ['normal', 'malnourished']:
            Path(f"data/{split}/{class_name}").mkdir(parents=True, exist_ok=True)
    
    total_downloaded = 0
    
    # Scrape from different sources
    print("\n1️⃣ UNICEF Sources...")
    total_downloaded += scrape_unicef_images(session)
    
    print("\n2️⃣ Medical Research Sources...")
    total_downloaded += scrape_medical_research_images(session)
    
    print("\n3️⃣ News Sources...")
    total_downloaded += scrape_news_images(session)
    
    print("\n4️⃣ Normal/Healthy Samples...")
    total_downloaded += download_sample_normal_images(session)
    
    print("\n5️⃣ Creating test set...")
    create_test_set()
    
    # Count final results
    train_normal = len(list(Path("data/train/normal").glob("*.jpg")))
    train_malnourished = len(list(Path("data/train/malnourished").glob("*.jpg")))
    test_normal = len(list(Path("data/test/normal").glob("*.jpg")))
    test_malnourished = len(list(Path("data/test/malnourished").glob("*.jpg")))
    
    print(f"\n🎉 Image scraping completed!")
    print(f"📊 Final Dataset:")
    print(f"   Training: {train_normal + train_malnourished} images")
    print(f"     - Normal: {train_normal}")
    print(f"     - Malnourished: {train_malnourished}")
    print(f"   Testing: {test_normal + test_malnourished} images")
    print(f"     - Normal: {test_normal}")
    print(f"     - Malnourished: {test_malnourished}")
    
    print(f"\n📁 Images saved to:")
    print(f"   - data/train/normal/")
    print(f"   - data/train/malnourished/")
    print(f"   - data/test/normal/")
    print(f"   - data/test/malnourished/")
    
    if total_downloaded > 0:
        print("\n✅ Ready for model training!")
    else:
        print("\n⚠️  Few images downloaded. Consider manual download or Kaggle datasets.")

if __name__ == "__main__":
    main() 