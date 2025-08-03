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
		{ id: 'performance', name: '⚡ Performance', icon: '⚡' },
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
						<div className='text-right flex flex-col items-end'>
							<div className='text-sm text-gray-500'>MLOps Pipeline</div>
							<div className='text-xs text-gray-400 flex items-center gap-2'>
								<span className='text-green-500 bg-green-100 px-2 py-1 rounded-md'>Production</span>
							</div>
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

				{activeTab === 'performance' && (
					<div className='space-y-6'>
						<div className='bg-white rounded-lg shadow p-6'>
							<h2 className='text-xl font-semibold text-gray-900 mb-4'>⚡ Performance Testing</h2>
							<div className='space-y-4'>
								<div className='bg-blue-50 border border-blue-200 rounded-lg p-4'>
									<h3 className='text-lg font-medium text-blue-900 mb-2'>Load Testing Results</h3>
									<p className='text-blue-700 mb-4'>
										View the latest performance report from Locust load testing.
									</p>
									<a
										href={`${process.env.NEXT_PUBLIC_API_URL}/performance-report`}
										target='_blank'
										rel='noopener noreferrer'
										className='inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
									>
										📊 View Performance Report
									</a>
								</div>
								<div className='bg-green-50 border border-green-200 rounded-lg p-4'>
									<h3 className='text-lg font-medium text-green-900 mb-2'>Run Load Test</h3>
									<p className='text-green-700 mb-4'>
										Execute a new load test to generate fresh performance metrics.
									</p>
									<code className='block bg-gray-100 p-3 rounded text-sm'>
										cd tests && locust -f locustfile.py --host=http://localhost:8000 --users=10
										--spawn-rate=2 --run-time=30s --headless --html=performance_report.html
									</code>
								</div>
							</div>
						</div>
					</div>
				)}
			</main>
		</div>
	);
}
