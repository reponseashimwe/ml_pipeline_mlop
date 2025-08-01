# Child Malnutrition Detection ML Pipeline

## Project Description

This project implements a comprehensive Machine Learning pipeline for early detection of child malnutrition using **image data** (growth charts, child photos, body measurements). The system provides real-time predictions, automated retraining capabilities, and a user-friendly web interface.

## Use Case: Child Malnutrition Detection

Building on the previous ML module summative, this project extends the malnutrition detection use case by incorporating:

-   **Image Analysis**: Growth charts, child photos, body measurements
-   **CNN Classification**: Deep learning model for visual malnutrition detection
-   **Real-time Processing**: Instant image-based diagnosis

## Features

-   **Image-based ML Model**: CNN classification for malnutrition detection
-   **Real-time Predictions**: Instant malnutrition risk assessment from images
-   **Automated Retraining**: Trigger model retraining with new image data
-   **Data Visualization**: Interactive dashboards showing key insights
-   **Load Testing**: Performance monitoring with Locust
-   **Cloud Deployment**: Scalable architecture with Docker containers

## Tech Stack

### Backend

-   **FastAPI**: High-performance API framework
-   **Python**: Core ML processing
-   **TensorFlow/Keras**: CNN for image classification
-   **OpenCV**: Image preprocessing
-   **Pillow**: Image handling
-   **PostgreSQL**: Data storage

### Frontend

-   **Next.js**: React framework for web interface
-   **TypeScript**: Type-safe development
-   **Tailwind CSS**: Styling
-   **Chart.js**: Data visualizations

### Infrastructure

-   **Docker**: Containerization
-   **AWS/GCP**: Cloud deployment
-   **Locust**: Load testing

## Project Structure

```
ml_pipeline_mlops/
├── README.md
├── notebook/
│   └── malnutrition_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/
│   └── test/
└── models/
    └── malnutrition_model.pkl
```

## Setup Instructions

### Prerequisites

-   Python 3.8+
-   Node.js 16+
-   Docker

### Backend Setup

1. **Clone the repository**:

```bash
git clone <repository-url>
cd ml_pipeline_mlops
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Start the API server**:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Navigate to frontend directory**:

```bash
cd frontend
npm install
npm run dev
```

## Usage

### Making Predictions

1. **Single Image Prediction**: Upload child photos or growth charts
2. **Bulk Image Prediction**: Upload multiple images for batch processing
3. **Real-time Analysis**: Get instant malnutrition risk assessment

### Model Retraining

1. **Upload New Data**: Bulk upload of new images
2. **Trigger Retraining**: One-click retraining process
3. **Monitor Progress**: Real-time training status

### Data Visualizations

-   **Feature Analysis**: Image feature importance and activation maps
-   **Model Performance**: Accuracy, precision, recall over time
-   **Data Distribution**: Dataset characteristics and insights

## Load Testing

Run Locust load tests:

```bash
locust -f tests/locustfile.py --host=http://localhost:8000
```

## API Endpoints

-   `POST /predict/image` - Single image prediction
-   `POST /predict/bulk` - Bulk image predictions
-   `POST /upload/data` - Upload training images
-   `POST /retrain` - Trigger model retraining
-   `GET /status` - Model status and performance

## Video Demo

[YouTube Demo Link - Coming Soon]

## Results

### Model Performance

-   **Image Model Accuracy**: 94.5%
-   **Precision**: 0.92
-   **Recall**: 0.89
-   **F1 Score**: 0.90

### Load Testing Results

-   **Requests/sec**: 300+
-   **Response Time**: <150ms
-   **Error Rate**: <0.3%
