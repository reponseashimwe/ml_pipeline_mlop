"""
Locust load testing configuration for the malnutrition detection API.
"""

from locust import HttpUser, task, between
import json
import base64
import os
import numpy as np
from PIL import Image
import io

class MalnutritionAPIUser(HttpUser):
    """Load testing user for the malnutrition detection API."""
    
    wait_time = between(0.5, 2)  # Faster requests for higher volume
    
    def on_start(self):
        """Initialize user session."""
        # Create a dummy image for testing
        self.dummy_image = self.create_dummy_image()
    
    def create_dummy_image(self):
        """Create a dummy image for testing."""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    
    @task(5)
    def health_check(self):
        """Test health check endpoint."""
        self.client.get("/health")
    
    @task(4)
    def get_status(self):
        """Test model status endpoint."""
        self.client.get("/status")
    
    @task(8)
    def predict_single_image(self):
        """Test single image prediction endpoint."""
        files = {'image': ('test_image.jpg', self.dummy_image, 'image/jpeg')}
        self.client.post("/predict/image", files=files)
    

    
    @task(3)
    def get_metrics(self):
        """Test performance metrics endpoint."""
        self.client.get("/metrics")
    
    @task(2)
    def upload_data(self):
        """Test data upload endpoint."""
        files = [
            ('files', ('train_image1.jpg', self.dummy_image, 'image/jpeg')),
            ('files', ('train_image2.jpg', self.dummy_image, 'image/jpeg'))
        ]
        self.client.post("/upload/data", files=files)
    
    @task(2)
    def trigger_retraining(self):
        """Test model retraining endpoint."""
        self.client.post("/retrain")


class HighLoadUser(HttpUser):
    """High load testing user for stress testing."""
    
    wait_time = between(0.05, 0.2)  # Ultra fast requests for maximum load
    
    def on_start(self):
        """Initialize user session."""
        self.dummy_image = self.create_dummy_image()
    
    def create_dummy_image(self):
        """Create a dummy image for testing."""
        img = Image.new('RGB', (224, 224), color='blue')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    
    @task(10)
    def rapid_predictions(self):
        """Rapid fire predictions for stress testing."""
        files = {'image': ('stress_test.jpg', self.dummy_image, 'image/jpeg')}
        self.client.post("/predict/image", files=files)
    
    @task(5)
    def health_check(self):
        """Frequent health checks."""
        self.client.get("/health")


class APIStressTest(HttpUser):
    """Stress testing for API endpoints."""
    
    wait_time = between(0.2, 1)  # Faster stress testing
    
    def on_start(self):
        """Initialize user session."""
        self.dummy_image = self.create_dummy_image()
    
    def create_dummy_image(self):
        """Create a dummy image for testing."""
        img = Image.new('RGB', (224, 224), color='green')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    
    @task(8)
    def mixed_requests(self):
        """Mix of different request types."""
        # Health check
        self.client.get("/health")
        
        # Status check
        self.client.get("/status")
        
        # Prediction
        files = {'image': ('mixed_test.jpg', self.dummy_image, 'image/jpeg')}
        self.client.post("/predict/image", files=files)
    



# Custom events for monitoring
from locust import events

@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Custom request handler for monitoring."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request successful: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a test is starting."""
    print("Load test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is ending."""
    print("Load test ending...") 