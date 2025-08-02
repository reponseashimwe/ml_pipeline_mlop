'use client';

import React, { useState, useEffect } from 'react';
import {
	BarChart,
	Bar,
	XAxis,
	YAxis,
	CartesianGrid,
	Tooltip,
	Legend,
	ResponsiveContainer,
	PieChart,
	Pie,
	Cell,
	LineChart,
	Line,
} from 'recharts';
import api, { VisualizationData } from '../lib/api';

interface VisualizationDataProps {
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

const DataVisualization: React.FC = () => {
	const [activeTab, setActiveTab] = useState('performance');
	const [data, setData] = useState<VisualizationDataProps | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

	useEffect(() => {
		fetchVisualizationData();
	}, []);

	const fetchVisualizationData = async () => {
		try {
			setIsLoading(true);
			setError(null);

			// Fetch real visualization data from backend
			const visualizationData = await api.getVisualizationData();
			setData(visualizationData);
			setLastUpdated(new Date());
		} catch (err) {
			console.error('Error fetching visualization data:', err);
			setError('Failed to load visualization data from backend');
		} finally {
			setIsLoading(false);
		}
	};

	const tabs = [
		{ id: 'performance', label: 'Model Performance', icon: '📊' },
		{ id: 'distribution', label: 'Class Distribution', icon: '🥧' },
		{ id: 'training', label: 'Training History', icon: '📈' },
		{ id: 'features', label: 'Feature Importance', icon: '🔍' },
		{ id: 'confusion', label: 'Confusion Matrix', icon: '🎯' },
		{ id: 'correlation', label: 'Correlation Matrix', icon: '🔗' },
	];

	if (isLoading) {
		return (
			<div className='flex items-center justify-center py-12'>
				<div className='animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600'></div>
				<span className='ml-3 text-gray-600'>Loading real-time data...</span>
			</div>
		);
	}

	if (error) {
		return (
			<div className='bg-red-50 border border-red-200 rounded-lg p-6'>
				<div className='flex items-center'>
					<div className='text-red-500 mr-3'>⚠️</div>
					<div>
						<h3 className='text-red-800 font-medium'>Data Loading Error</h3>
						<p className='text-red-700 text-sm mt-1'>{error}</p>
						<button
							onClick={fetchVisualizationData}
							className='mt-2 text-sm text-red-600 hover:text-red-800 underline'
						>
							Retry
						</button>
					</div>
				</div>
			</div>
		);
	}

	if (!data) {
		return (
			<div className='bg-gray-50 border border-gray-200 rounded-lg p-6'>
				<div className='text-center'>
					<p className='text-gray-500'>No data available</p>
					<button
						onClick={fetchVisualizationData}
						className='mt-2 text-sm text-blue-600 hover:text-blue-800 underline'
					>
						Load Data
					</button>
				</div>
			</div>
		);
	}

	return (
		<div className='bg-white border rounded-lg p-6'>
			{/* Header with refresh button */}
			<div className='flex justify-between items-center mb-6'>
				<h3 className='text-lg font-semibold text-gray-900'>Data Visualizations</h3>
				<div className='flex items-center space-x-3'>
					{lastUpdated && (
						<p className='text-xs text-gray-500'>Last updated: {lastUpdated.toLocaleTimeString()}</p>
					)}
					<button
						onClick={fetchVisualizationData}
						disabled={isLoading}
						className='text-sm text-blue-600 hover:text-blue-800 disabled:text-gray-400'
					>
						🔄 Refresh
					</button>
				</div>
			</div>

			{/* Navigation Tabs */}
			<div className='mb-6'>
				<div className='border-b border-gray-200'>
					<nav className='-mb-px flex space-x-8'>
						{tabs.map((tab) => (
							<button
								key={tab.id}
								onClick={() => setActiveTab(tab.id)}
								className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors duration-200 ${
									activeTab === tab.id
										? 'border-blue-500 text-blue-600'
										: 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
								}`}
							>
								<span className='mr-2'>{tab.icon}</span>
								{tab.label}
							</button>
						))}
					</nav>
				</div>
			</div>

			{/* Data Source Info */}
			<div className='mb-4 p-3 bg-blue-50 rounded-lg'>
				<div className='flex items-center justify-between text-sm'>
					<div className='flex items-center space-x-2'>
						<span className='text-blue-600'>📊</span>
						<span className='text-blue-800 font-medium'>Real-time Data</span>
					</div>
					<div className='text-blue-600'>Total Training Images: {data.total_training_images}</div>
				</div>
				<div className='mt-2 text-xs text-blue-600'>
					<p>• Model Performance: Real metrics from trained model</p>
					<p>• Class Distribution: Actual file counts from training data</p>
					<p>• Training History: Based on MobileNetV2 training patterns</p>
					<p>• Feature Importance: Analysis of malnutrition detection features</p>
				</div>
			</div>

			{/* Visualization Content */}
			<div className='bg-white border rounded-lg p-6'>
				{activeTab === 'performance' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>{data.chart_titles.performance}</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<BarChart data={data.model_performance}>
									<CartesianGrid strokeDasharray='3 3' />
									<XAxis dataKey='metric' />
									<YAxis domain={[0, 1]} />
									<Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
									<Bar dataKey='value' fill='#3B82F6' />
								</BarChart>
							</ResponsiveContainer>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> {data.interpretations.performance}
							</p>
						</div>
					</div>
				)}

				{activeTab === 'distribution' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>{data.chart_titles.distribution}</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<PieChart>
									<Pie
										data={data.class_distribution}
										cx='50%'
										cy='50%'
										labelLine={false}
										label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
										outerRadius={80}
										fill='#8884d8'
										dataKey='value'
									>
										{data.class_distribution.map((entry, index) => (
											<Cell key={`cell-${index}`} fill={entry.color} />
										))}
									</Pie>
									<Tooltip />
								</PieChart>
							</ResponsiveContainer>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> {data.interpretations.distribution}
							</p>
						</div>
					</div>
				)}

				{activeTab === 'training' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>{data.chart_titles.training}</h4>
						<div className='h-80'>
							{data.training_history.length > 0 ? (
								<ResponsiveContainer width='100%' height='100%'>
									<LineChart data={data.training_history}>
										<CartesianGrid strokeDasharray='3 3' />
										<XAxis dataKey='epoch' />
										<YAxis domain={[0, 1]} />
										<Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
										<Legend />
										<Line
											type='monotone'
											dataKey='accuracy'
											stroke='#3B82F6'
											name='Training Accuracy'
										/>
										<Line
											type='monotone'
											dataKey='val_accuracy'
											stroke='#10B981'
											name='Validation Accuracy'
										/>
										<Line type='monotone' dataKey='loss' stroke='#EF4444' name='Training Loss' />
										<Line
											type='monotone'
											dataKey='val_loss'
											stroke='#F59E0B'
											name='Validation Loss'
										/>
									</LineChart>
								</ResponsiveContainer>
							) : (
								<div className='h-80 flex items-center justify-center bg-gray-50 rounded-lg'>
									<img
										src='/api/training-plots'
										alt='Training Plots'
										className='max-w-full max-h-full object-contain'
										onError={(e) => {
											const target = e.target as HTMLImageElement;
											target.style.display = 'none';
											target.nextElementSibling?.classList.remove('hidden');
										}}
									/>
									<div className='hidden text-center text-gray-500'>
										<p>Training plots not available</p>
										<p className='text-sm'>Train the model to generate training plots</p>
									</div>
								</div>
							)}
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> {data.interpretations.training}
							</p>
						</div>
					</div>
				)}

				{activeTab === 'features' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>{data.chart_titles.features}</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<BarChart data={data.feature_importance} layout='horizontal'>
									<CartesianGrid strokeDasharray='3 3' />
									<XAxis type='number' domain={[0, 1]} />
									<YAxis dataKey='feature' type='category' width={120} />
									<Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
									<Bar dataKey='importance' fill='#3B82F6' />
								</BarChart>
							</ResponsiveContainer>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> {data.interpretations.features}
							</p>
						</div>
					</div>
				)}

				{activeTab === 'confusion' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Confusion Matrix</h4>
						<div className='h-80 flex items-center justify-center bg-gray-50 rounded-lg'>
							<img
								src='/api/confusion-matrix'
								alt='Confusion Matrix'
								className='max-w-full max-h-full object-contain'
								onError={(e) => {
									const target = e.target as HTMLImageElement;
									target.style.display = 'none';
									target.nextElementSibling?.classList.remove('hidden');
								}}
							/>
							<div className='hidden text-center text-gray-500'>
								<p>Confusion matrix not available</p>
								<p className='text-sm'>Train the model to generate confusion matrix</p>
							</div>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> The confusion matrix shows the model's prediction
								accuracy for each class. Diagonal values represent correct predictions, while
								off-diagonal values show misclassifications.
							</p>
						</div>
					</div>
				)}

				{activeTab === 'correlation' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Correlation Matrix</h4>
						<div className='h-80 flex items-center justify-center bg-gray-50 rounded-lg'>
							<img
								src='/api/correlation-matrix'
								alt='Correlation Matrix'
								className='max-w-full max-h-full object-contain'
								onError={(e) => {
									const target = e.target as HTMLImageElement;
									target.style.display = 'none';
									target.nextElementSibling?.classList.remove('hidden');
								}}
							/>
							<div className='hidden text-center text-gray-500'>
								<p>Correlation matrix not available</p>
								<p className='text-sm'>Train the model to generate correlation matrix</p>
							</div>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> The correlation matrix shows relationships between
								different features. High correlation values indicate strong relationships, while low
								values suggest independence.
							</p>
						</div>
					</div>
				)}
			</div>

			{/* Insights Summary */}
			<div className='bg-blue-50 border border-blue-200 rounded-lg p-6'>
				<h4 className='text-lg font-medium text-blue-900 mb-3'>Key Insights</h4>
				<div className='grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800'>
					<div>
						<p>
							<strong>🎯 Model Performance:</strong> {data.key_insights.model_performance}
						</p>
						<p>
							<strong>📊 Data Balance:</strong> {data.key_insights.data_balance}
						</p>
					</div>
					<div>
						<p>
							<strong>🔍 Feature Importance:</strong> {data.key_insights.feature_importance}
						</p>
						<p>
							<strong>📈 Training Stability:</strong> {data.key_insights.training_stability}
						</p>
					</div>
				</div>
			</div>
		</div>
	);
};

export default DataVisualization;
