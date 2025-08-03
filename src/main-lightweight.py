"""
Lightweight FastAPI application for child malnutrition detection ML pipeline.
Optimized for 512MB memory limit on Render.
"""

import os
import shutil
import tempfile
from typing import List, Optional
from datetime import datetime
import logging
import json
import asyncio
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Malnutrition Detection API",
    description="ML Pipeline for child malnutrition detection using image classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for test images
app.mount("/static", StaticFiles(directory="test-files"), name="static")

# Lazy imports to save memory
predictor = None
db = None

def get_predictor():
    """Lazy load predictor to save memory"""
    global predictor
    if predictor is None:
        try:
            # Import only when needed
            from prediction import create_predictor
            logger.info("🔄 Loading predictor (lazy initialization)")
            predictor = create_predictor("../models/malnutrition_model.h5", confidence_threshold=0.70)
            logger.info("✅ Predictor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    return predictor

def get_database():
    """Lazy load database to save memory"""
    global db
    if db is None:
        try:
            # Import only when needed
            from database import get_database
            db = get_database()
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            db = None
    return db

# Global variables for model status
model_status = {
    "loaded": False,
    "last_updated": None,
    "performance_metrics": None,
    "is_loaded": False,
    "model_path": "",
    "uptime_start": datetime.now()
}

# Training data storage
uploads_temp_dir = "../data/uploads_temp"
main_train_dir = "../data/train"

# Create required directories
os.makedirs(uploads_temp_dir, exist_ok=True)
os.makedirs(f"{uploads_temp_dir}/malnourished", exist_ok=True)
os.makedirs(f"{uploads_temp_dir}/overnourished", exist_ok=True)
os.makedirs(main_train_dir, exist_ok=True)
os.makedirs(f"{main_train_dir}/malnourished", exist_ok=True)
os.makedirs(f"{main_train_dir}/overnourished", exist_ok=True)

# Global variables for training progress
training_jobs = {}

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[dict] = None
    error: Optional[str] = None
    timestamp: str

class ModelStatusResponse(BaseModel):
    model_loaded: bool
    model_path: str
    last_updated: Optional[str] = None
    performance_metrics: Optional[dict] = None

class RetrainingResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Child Malnutrition Detection API (Lightweight)...")
    
    # Check if model exists (but don't load it yet to save memory)
    model_path = "../models/malnutrition_model.h5"
    if os.path.exists(model_path):
        model_status["loaded"] = False  # Will be loaded on first prediction
        model_status["is_loaded"] = False
        model_status["model_path"] = model_path
        logger.info("Model file found - will load on first prediction (lazy loading)")
    else:
        logger.info("No existing model found. Please train the model first using the notebook.")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Child Malnutrition Detection API (Lightweight)",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": predictor is not None,
        "model_available": os.path.exists("../models/malnutrition_model.h5"),
        "memory_optimized": True,
        "endpoints": {
            "predict_single": "/predict/image",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.post("/predict/image")
async def predict_image(image: UploadFile = File(...)):
    """
    Predict malnutrition from uploaded image.
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Lazy load predictor
        current_predictor = get_predictor()
        
        # Make prediction
        result = current_predictor.predict_single(image_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get model and system status."""
    try:
        # Lazy load database
        current_db = get_database()
        
        # Get system info
        import psutil
        memory_info = psutil.virtual_memory()
        
        status_data = {
            "model_loaded": predictor is not None,
            "model_path": model_status.get("model_path", ""),
            "last_updated": model_status.get("last_updated"),
            "uptime": str(datetime.now() - model_status["uptime_start"]),
            "system": {
                "memory_used_mb": round(memory_info.used / 1024 / 1024, 2),
                "memory_total_mb": round(memory_info.total / 1024 / 1024, 2),
                "memory_percent": round(memory_info.percent, 2),
                "cpu_percent": psutil.cpu_percent()
            },
            "training_jobs": len(training_jobs),
            "api_version": "1.0.0",
            "memory_optimized": True
        }
        
        return status_data
        
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return {
            "error": str(e),
            "model_loaded": False,
            "memory_optimized": True
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_optimized": True
    }

@app.get("/performance-report")
async def get_performance_report():
    """Get the latest performance report from Locust testing."""
    try:
        report_path = "../tests/performance_report.html"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        else:
            return HTMLResponse(content="""
                <html>
                <head><title>Performance Report</title></head>
                <body>
                    <h1>Performance Report Not Available</h1>
                    <p>No performance report found. Run Locust load testing first.</p>
                    <p><a href="/">Back to API</a></p>
                </body>
                </html>
            """, status_code=404)
    except Exception as e:
        logger.error(f"Error reading performance report: {e}")
        return HTMLResponse(content=f"Error reading report: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000))) 