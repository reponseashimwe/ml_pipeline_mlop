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
	const [activeTab, setActiveTab] = useState('overview');
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
		{ id: 'overview', label: 'Model Overview', icon: '📊' },
		{ id: 'training', label: 'Training History', icon: '📈' },
		{ id: 'features', label: 'Feature Analysis', icon: '🔍' },
	];

	if (isLoading && !data) {
		return (
			<div className='flex items-center justify-center py-12'>
				<div className='animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600'></div>
				<span className='ml-3 text-gray-600'>Loading real-time data...</span>
			</div>
		);
	}

	if (error && !data) {
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
					{isLoading && (
						<div className='flex items-center text-sm text-gray-500'>
							<div className='animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 mr-2'></div>
							Refreshing...
						</div>
					)}
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
					<p>• Model Overview: Performance metrics & class distribution</p>
					<p>• Training History: MobileNetV2 training patterns & progress</p>
					<p>• Feature Analysis: Malnutrition detection feature importance</p>
				</div>
			</div>

			{/* Visualization Content */}
			<div className='bg-white border rounded-lg p-6'>
				{activeTab === 'overview' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Model Overview</h4>
						<div className='grid grid-cols-1 lg:grid-cols-2 gap-6'>
							{/* Model Performance */}
							<div className='bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200'>
								<h5 className='text-sm font-semibold text-blue-900 mb-3'>📊 Model Performance</h5>
								<div className='h-64'>
									<ResponsiveContainer width='100%' height='100%'>
										<BarChart
											data={data.model_performance}
											margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
										>
											<CartesianGrid strokeDasharray='3 3' stroke='#e5e7eb' opacity={0.6} />
											<XAxis
												dataKey='metric'
												tick={{ fontSize: 11, fill: '#6b7280' }}
												axisLine={{ stroke: '#d1d5db' }}
												tickLine={{ stroke: '#d1d5db' }}
											/>
											<YAxis
												domain={[0, 1]}
												tick={{ fontSize: 11, fill: '#6b7280' }}
												axisLine={{ stroke: '#d1d5db' }}
												tickLine={{ stroke: '#d1d5db' }}
												tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
											/>
											<Tooltip
												formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
												contentStyle={{
													backgroundColor: '#ffffff',
													border: '1px solid #e5e7eb',
													borderRadius: '8px',
													boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
												}}
											/>
											<Bar dataKey='value' fill='#3B82F6' radius={[4, 4, 0, 0]} />
										</BarChart>
									</ResponsiveContainer>
								</div>
								<div className='mt-3 text-xs text-blue-700'>
									<p>
										<strong>Interpretation:</strong> {data.interpretations.performance}
									</p>
								</div>
							</div>

							{/* Class Distribution */}
							<div className='bg-gradient-to-br from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200'>
								<h5 className='text-sm font-semibold text-green-900 mb-3'>🥧 Class Distribution</h5>
								<div className='h-64'>
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
											<Tooltip
												contentStyle={{
													backgroundColor: '#ffffff',
													border: '1px solid #e5e7eb',
													borderRadius: '8px',
													boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
												}}
											/>
										</PieChart>
									</ResponsiveContainer>
								</div>
								<div className='mt-3 text-xs text-green-700'>
									<p>
										<strong>Interpretation:</strong> {data.interpretations.distribution}
									</p>
								</div>
							</div>
						</div>
					</div>
				)}

				{activeTab === 'training' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Training History</h4>
						<div className='bg-gradient-to-br from-orange-50 to-red-50 p-6 rounded-lg border border-orange-200'>
							<h5 className='text-sm font-semibold text-orange-900 mb-4'>📈 Model Training Progress</h5>
							<div className='h-80'>
								{data.training_history.length > 0 ? (
									<ResponsiveContainer width='100%' height='100%'>
										<LineChart
											data={data.training_history}
											margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
										>
											<CartesianGrid strokeDasharray='3 3' stroke='#e5e7eb' opacity={0.6} />
											<XAxis
												dataKey='epoch'
												tick={{ fontSize: 11, fill: '#6b7280' }}
												axisLine={{ stroke: '#d1d5db' }}
												tickLine={{ stroke: '#d1d5db' }}
											/>
											<YAxis
												domain={[0, 1]}
												tick={{ fontSize: 11, fill: '#6b7280' }}
												axisLine={{ stroke: '#d1d5db' }}
												tickLine={{ stroke: '#d1d5db' }}
												tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
											/>
											<Tooltip
												formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
												contentStyle={{
													backgroundColor: '#ffffff',
													border: '1px solid #e5e7eb',
													borderRadius: '8px',
													boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
												}}
											/>
											<Legend />
											<Line
												type='monotone'
												dataKey='accuracy'
												stroke='#3B82F6'
												strokeWidth={3}
												dot={{ r: 3, fill: '#3B82F6', strokeWidth: 2, stroke: '#ffffff' }}
												activeDot={{ r: 5, stroke: '#3B82F6', strokeWidth: 2 }}
												name='Training Accuracy'
											/>
											<Line
												type='monotone'
												dataKey='val_accuracy'
												stroke='#10B981'
												strokeWidth={3}
												dot={{ r: 3, fill: '#10B981', strokeWidth: 2, stroke: '#ffffff' }}
												activeDot={{ r: 5, stroke: '#10B981', strokeWidth: 2 }}
												name='Validation Accuracy'
											/>
											<Line
												type='monotone'
												dataKey='loss'
												stroke='#EF4444'
												strokeWidth={2}
												dot={{ r: 2, fill: '#EF4444', strokeWidth: 1, stroke: '#ffffff' }}
												name='Training Loss'
											/>
											<Line
												type='monotone'
												dataKey='val_loss'
												stroke='#F59E0B'
												strokeWidth={2}
												dot={{ r: 2, fill: '#F59E0B', strokeWidth: 1, stroke: '#ffffff' }}
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
							<div className='mt-4 text-sm text-orange-700'>
								<p>
									<strong>Interpretation:</strong> {data.interpretations.training}
								</p>
							</div>
						</div>
					</div>
				)}

				{activeTab === 'features' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Feature Analysis</h4>
						<div className='bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-lg border border-purple-200'>
							<h5 className='text-sm font-semibold text-purple-900 mb-4'>
								🔍 Malnutrition Detection Features
							</h5>
							<div className='h-80'>
								<ResponsiveContainer width='100%' height='100%'>
									<BarChart
										data={data.feature_importance}
										layout='horizontal'
										margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
									>
										<CartesianGrid strokeDasharray='3 3' stroke='#e5e7eb' opacity={0.6} />
										<XAxis
											type='number'
											domain={[0, 1]}
											tick={{ fontSize: 11, fill: '#6b7280' }}
											axisLine={{ stroke: '#d1d5db' }}
											tickLine={{ stroke: '#d1d5db' }}
											tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
										/>
										<YAxis
											dataKey='feature'
											type='category'
											width={140}
											tick={{ fontSize: 11, fill: '#6b7280' }}
											axisLine={{ stroke: '#d1d5db' }}
											tickLine={{ stroke: '#d1d5db' }}
										/>
										<Tooltip
											formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
											contentStyle={{
												backgroundColor: '#ffffff',
												border: '1px solid #e5e7eb',
												borderRadius: '8px',
												boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
											}}
										/>
										<Bar
											dataKey='importance'
											fill='#8B5CF6'
											radius={[0, 4, 4, 0]}
											background={{ fill: '#f3f4f6' }}
										/>
									</BarChart>
								</ResponsiveContainer>
							</div>
							<div className='mt-4 text-sm text-purple-700'>
								<p>
									<strong>Interpretation:</strong> {data.interpretations.features}
								</p>
							</div>
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
