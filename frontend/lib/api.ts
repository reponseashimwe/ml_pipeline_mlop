import axios, { AxiosInstance, AxiosResponse } from 'axios';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// Create axios client with base configuration
const axiosClient = axios.create({
	baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
	timeout: 60000, // Increase timeout to 60 seconds for training operations
	headers: {
		'Content-Type': 'application/json',
	},
});

// Request interceptor for logging
axiosClient.interceptors.request.use(
	(config) => {
		console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
		return config;
	},
	(error) => {
		console.error('❌ Request Error:', error);
		return Promise.reject(error);
	}
);

// Response interceptor for error handling
axiosClient.interceptors.response.use(
	(response: AxiosResponse) => {
		console.log(`✅ API Response: ${response.status} ${response.config.url}`);
		return response;
	},
	(error) => {
		console.error('❌ Response Error:', error.response?.data || error.message);
		return Promise.reject(error);
	}
);

// API Types
export interface PredictionResult {
	class: string;
	predicted_class: string;
	confidence: number;
	probabilities: {
		malnourished: number;
		overnourished: number;
		normal: number;
	};
	interpretation: string;
	recommendation: string;
	confidence_threshold: number;
	features?: {
		color: number[];
		texture: number[];
		shape: number[];
	};
}

export interface ModelStatus {
	is_loaded: boolean;
	model_path: string;
	last_updated: string;
	uptime: string;
	memory_usage: string;
	cpu_usage: string;
	performance: {
		accuracy: number;
		precision: number;
		recall: number;
		f1_score: number;
	};
}

export interface RetrainResponse {
	success: boolean;
	message: string;
	job_id?: string;
}

export interface TrainingProgress {
	epoch: number;
	total_epochs: number;
	accuracy: number;
	loss: number;
	val_accuracy: number;
	val_loss: number;
	status: 'training' | 'completed' | 'failed';
	error?: string;
}

export interface VisualizationData {
	model_performance: Array<{
		metric: string;
		value: number;
		color: string;
	}>;
	class_distribution: Array<{
		name: string;
		value: number;
		color: string;
	}>;
	training_history: Array<{
		epoch: number;
		accuracy: number;
		loss: number;
		val_accuracy: number;
		val_loss: number;
	}>;
	feature_importance: Array<{
		feature: string;
		importance: number;
		color: string;
	}>;
	total_training_images: number;
	last_updated: string;
	// Backend text content
	interpretations: {
		performance: string;
		distribution: string;
		training: string;
		features: string;
	};
	key_insights: {
		model_performance: string;
		data_balance: string;
		feature_importance: string;
		training_stability: string;
	};
	chart_titles: {
		performance: string;
		distribution: string;
		training: string;
		features: string;
	};
}

// API Functions
export const api = {
	// Predict a single image
	predictImage: async (formData: FormData): Promise<PredictionResult> => {
		const response = await axiosClient.post('/predict/image', formData, {
			headers: {
				'Content-Type': 'multipart/form-data',
			},
		});
		return response.data;
	},

	// Bulk data upload for training
	uploadTrainingData: async (files: FileList): Promise<{ success: boolean; message: string }> => {
		const formData = new FormData();
		Array.from(files).forEach((file) => {
			formData.append('files', file);
		});

		const response = await axiosClient.post('/upload/data', formData, {
			headers: {
				'Content-Type': 'multipart/form-data',
			},
		});

		return response.data;
	},

	// Upload labeled training data
	uploadLabeledTrainingData: async (formData: FormData): Promise<{ success: boolean; message: string }> => {
		const response = await axiosClient.post('/upload/labeled-data', formData, {
			headers: {
				'Content-Type': 'multipart/form-data',
			},
		});

		return response.data;
	},

	// Get model status
	getModelStatus: async (): Promise<ModelStatus> => {
		const response = await axiosClient.get('/status');
		return response.data;
	},

	// Get visualization data
	getVisualizationData: async (): Promise<VisualizationData> => {
		const response = await axiosClient.get('/api/visualization-data');
		return response.data;
	},

	// Retrain model
	retrainModel: async (): Promise<RetrainResponse> => {
		const response = await axiosClient.post('/retrain');
		return response.data;
	},

	// Get training progress
	getTrainingProgress: async (jobId: string): Promise<TrainingProgress> => {
		const response = await axiosClient.get(`/training/progress/${jobId}`);
		return response.data;
	},

	// Get API health
	getHealth: async (): Promise<{ status: string; message: string }> => {
		const response = await axiosClient.get('/');
		return response.data;
	},

	// Get existing uploaded training images
	getUploadedImages: async (): Promise<{ success: boolean; images: any[]; total_count: number }> => {
		const response = await axiosClient.get('/api/uploaded-images');
		return response.data;
	},
};

export default api;
