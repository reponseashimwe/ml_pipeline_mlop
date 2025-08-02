'use client';

import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import UploadData from '../components/UploadData';
import RetrainingPanel from '../components/RetrainingPanel';
import DataVisualization from '../components/DataVisualization';
import ModelStatus from '../components/ModelStatus';
import PredictionResult from '../components/PredictionResult';
import { PredictionResult as PredictionResultType } from '../lib/api';

export default function Home() {
	const [activeTab, setActiveTab] = useState('prediction');
	const [predictionResult, setPredictionResult] = useState<PredictionResultType | null>(null);

	const tabs = [
		{ id: 'prediction', name: '🔍 Prediction', icon: '🔍' },
		{ id: 'upload', name: '📚 Upload Data', icon: '📚' },
		{ id: 'retrain', name: '🔄 Retrain Model', icon: '🔄' },
		{ id: 'visualizations', name: '📊 Visualizations', icon: '📊' },
		{ id: 'status', name: '📈 Model Status', icon: '📈' },
	];

	return (
		<div className='min-h-screen bg-gray-50'>
			{/* Header */}
			<header className='bg-white shadow-sm border-b'>
				<div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
					<div className='flex justify-between items-center py-4'>
						<div>
							<h1 className='text-2xl font-bold text-gray-900'>🏥 Malnutrition Detection ML Pipeline</h1>
							<p className='text-sm text-gray-600'>
								End-to-end machine learning system for child malnutrition detection
							</p>
						</div>
						<div className='text-right'>
							<div className='text-sm text-gray-500'>MLOps Pipeline</div>
							<div className='text-xs text-gray-400'>Production Ready</div>
						</div>
					</div>
				</div>
			</header>

			{/* Navigation Tabs */}
			<div className='bg-white border-b'>
				<div className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
					<nav className='flex space-x-8'>
						{tabs.map((tab) => (
							<button
								key={tab.id}
								onClick={() => setActiveTab(tab.id)}
								className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
									activeTab === tab.id
										? 'border-blue-500 text-blue-600'
										: 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
								}`}
							>
								{tab.name}
							</button>
						))}
					</nav>
				</div>
			</div>

			{/* Main Content */}
			<main className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8'>
				{/* Tab Content */}
				{activeTab === 'prediction' && (
					<div className='space-y-6'>
						<ImageUpload onPrediction={setPredictionResult} />
						{predictionResult && <PredictionResult result={predictionResult} />}
					</div>
				)}

				{activeTab === 'upload' && <UploadData />}

				{activeTab === 'retrain' && <RetrainingPanel />}

				{activeTab === 'visualizations' && <DataVisualization />}

				{activeTab === 'status' && <ModelStatus />}
			</main>
		</div>
	);
}
