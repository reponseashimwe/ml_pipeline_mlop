'use client';

import React, { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';
import PredictionResult from '@/components/PredictionResult';
import ModelStatus from '@/components/ModelStatus';
import DataVisualization from '@/components/DataVisualization';
import RetrainingPanel from '@/components/RetrainingPanel';

export default function Home() {
	const [activeTab, setActiveTab] = useState('prediction');
	const [predictionResult, setPredictionResult] = useState(null);

	const tabs = [
		{ id: 'prediction', label: 'Prediction', icon: '🔮' },
		{ id: 'upload', label: 'Upload Data', icon: '📁' },
		{ id: 'retrain', label: 'Retrain Model', icon: '🔄' },
		{ id: 'visualize', label: 'Visualizations', icon: '📊' },
		{ id: 'status', label: 'Model Status', icon: '📈' },
	];

	return (
		<div className='min-h-screen bg-gray-50'>
			{/* Header */}
			<header className='bg-white shadow-sm border-b'>
				<div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
					<div className='flex justify-between items-center py-6'>
						<div>
							<h1 className='text-3xl font-bold text-gray-900'>ML Pipeline - Malnutrition Detection</h1>
							<p className='text-gray-600 mt-1'>
								End-to-end machine learning pipeline for image-based malnutrition detection
							</p>
						</div>
						<div className='flex items-center space-x-4'>
							<div className='flex items-center space-x-2'>
								<div className='w-3 h-3 bg-green-500 rounded-full animate-pulse'></div>
								<span className='text-sm text-gray-600'>System Online</span>
							</div>
						</div>
					</div>
				</div>
			</header>

			{/* Navigation Tabs */}
			<nav className='bg-white border-b'>
				<div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
					<div className='flex space-x-8'>
						{tabs.map((tab) => (
							<button
								key={tab.id}
								onClick={() => setActiveTab(tab.id)}
								className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-200 ${
									activeTab === tab.id
										? 'border-blue-500 text-blue-600'
										: 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
								}`}
							>
								<span className='mr-2'>{tab.icon}</span>
								{tab.label}
							</button>
						))}
					</div>
				</div>
			</nav>

			{/* Main Content */}
			<main className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>
				{activeTab === 'prediction' && (
					<div className='space-y-6'>
						<div className='card'>
							<h2 className='text-2xl font-semibold text-gray-900 mb-4'>Image Prediction</h2>
							<ImageUpload onPrediction={setPredictionResult} />
						</div>
						{predictionResult && (
							<div className='card'>
								<PredictionResult result={predictionResult} />
							</div>
						)}
					</div>
				)}

				{activeTab === 'upload' && (
					<div className='card'>
						<h2 className='text-2xl font-semibold text-gray-900 mb-4'>Upload Training Data</h2>
						<p className='text-gray-600 mb-4'>
							Upload new images to improve the model's performance through retraining.
						</p>
						<ImageUpload onPrediction={setPredictionResult} isTrainingData={true} />
					</div>
				)}

				{activeTab === 'retrain' && (
					<div className='card'>
						<RetrainingPanel />
					</div>
				)}

				{activeTab === 'visualize' && (
					<div className='card'>
						<DataVisualization />
					</div>
				)}

				{activeTab === 'status' && (
					<div className='card'>
						<ModelStatus />
					</div>
				)}
			</main>
		</div>
	);
}
