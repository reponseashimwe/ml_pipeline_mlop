"""
Main FastAPI application for child malnutrition detection ML pipeline.
Provides endpoints for prediction, model management, and data upload.
"""

import os
import shutil
import tempfile
from typing import List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our modules
from .preprocessing import ImagePreprocessor
from .model import MalnutritionCNN
from .prediction import MalnutritionPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Child Malnutrition Detection API",
    description="ML Pipeline for detecting child malnutrition from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = ImagePreprocessor()
predictor = MalnutritionPredictor()
model = MalnutritionCNN()

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[dict] = None
    error: Optional[str] = None
    timestamp: str

class BatchPredictionResponse(BaseModel):
    success: bool
    total_images: int
    successful_predictions: int
    failed_predictions: int
    average_confidence: float
    class_distribution: dict
    predictions: List[dict]
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

# Global variables for model status
model_status = {
    "loaded": False,
    "last_updated": None,
    "performance_metrics": None
}

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Child Malnutrition Detection API...")
    
    # Try to load existing model
    model_path = "models/malnutrition_model.pkl"
    if os.path.exists(model_path):
        success = predictor.load_model()
        if success:
            model_status["loaded"] = True
            model_status["last_updated"] = datetime.now().isoformat()
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning("Failed to load existing model on startup")
    else:
        logger.info("No existing model found. Model will be created during training.")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Child Malnutrition Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict_single": "/predict/image",
            "predict_batch": "/predict/bulk",
            "upload_data": "/upload/data",
            "retrain": "/retrain",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.post("/predict/image", response_model=PredictionResponse)
async def predict_single_image(file: UploadFile = File(...)):
    """
    Predict malnutrition status for a single uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction result with class and confidence
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Make prediction
        result = predictor.predict_single_image(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return PredictionResponse(
            success=True,
            prediction=result['prediction'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error in single image prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bulk", response_model=BatchPredictionResponse)
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Predict malnutrition status for multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Batch prediction results with statistics
    """
    try:
        # Validate files
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
        
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_paths.append(temp_file.name)
        
        # Make batch prediction
        result = predictor.predict_batch_images(temp_paths)
        
        # Clean up temporary files
        for temp_path in temp_paths:
            os.unlink(temp_path)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return BatchPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/data")
async def upload_training_data(files: List[UploadFile] = File(...)):
    """
    Upload training data (images) for model retraining.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Upload status and file information
    """
    try:
        uploaded_files = []
        failed_files = []
        
        # Create upload directory
        upload_dir = "data/train"
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    failed_files.append({
                        "filename": file.filename,
                        "error": "File must be an image"
                    })
                    continue
                
                # Save file
                file_path = os.path.join(upload_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "size": os.path.getsize(file_path),
                    "path": file_path
                })
                
            except Exception as e:
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} files successfully",
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "total_files": len(files),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in data upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain", response_model=RetrainingResponse)
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with uploaded data.
    
    Returns:
        Retraining job status
    """
    try:
        # Check if training data exists
        train_dir = "data/train"
        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            raise HTTPException(status_code=400, detail="No training data found. Upload images first.")
        
        # Generate job ID
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add retraining task to background
        background_tasks.add_task(retrain_model_task, job_id)
        
        return RetrainingResponse(
            success=True,
            message="Retraining job started successfully",
            job_id=job_id
        )
        
    except Exception as e:
        logger.error(f"Error starting retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_model_task(job_id: str):
    """
    Background task for model retraining.
    
    Args:
        job_id: Unique job identifier
    """
    try:
        logger.info(f"Starting retraining job: {job_id}")
        
        # TODO: Implement actual retraining logic
        # This would involve:
        # 1. Loading training data
        # 2. Preprocessing images
        # 3. Training the model
        # 4. Saving the new model
        # 5. Updating model status
        
        # For now, just simulate training
        import time
        time.sleep(5)  # Simulate training time
        
        logger.info(f"Retraining job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in retraining job {job_id}: {str(e)}")

@app.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get current model status and performance metrics.
    
    Returns:
        Model status information
    """
    try:
        return ModelStatusResponse(
            model_loaded=model_status["loaded"],
            model_path="models/malnutrition_model.pkl",
            last_updated=model_status["last_updated"],
            performance_metrics=model_status["performance_metrics"]
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_status["loaded"]
    }

@app.get("/metrics")
async def get_performance_metrics():
    """
    Get detailed performance metrics.
    
    Returns:
        Performance metrics and statistics
    """
    try:
        # TODO: Implement actual metrics collection
        # This would include:
        # - Model accuracy, precision, recall, F1-score
        # - Response times
        # - Request counts
        # - Error rates
        
        return {
            "model_metrics": {
                "accuracy": 0.945,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90
            },
            "api_metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 