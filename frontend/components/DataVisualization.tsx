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

interface VisualizationData {
	modelPerformance: Array<{
		metric: string;
		value: number;
		color: string;
	}>;
	classDistribution: Array<{
		name: string;
		value: number;
		color: string;
	}>;
	trainingHistory: Array<{
		epoch: number;
		accuracy: number;
		loss: number;
		val_accuracy: number;
		val_loss: number;
	}>;
	featureImportance: Array<{
		feature: string;
		importance: number;
		color: string;
	}>;
}

const DataVisualization: React.FC = () => {
	const [activeTab, setActiveTab] = useState('performance');
	const [data, setData] = useState<VisualizationData | null>(null);
	const [isLoading, setIsLoading] = useState(true);

	useEffect(() => {
		// Simulate loading data
		setTimeout(() => {
			setData({
				modelPerformance: [
					{ metric: 'Accuracy', value: 0.89, color: '#3B82F6' },
					{ metric: 'Precision', value: 0.87, color: '#10B981' },
					{ metric: 'Recall', value: 0.91, color: '#8B5CF6' },
					{ metric: 'F1-Score', value: 0.89, color: '#F59E0B' },
				],
				classDistribution: [
					{ name: 'Normal', value: 65, color: '#10B981' },
					{ name: 'Malnourished', value: 35, color: '#EF4444' },
				],
				trainingHistory: [
					{ epoch: 1, accuracy: 0.65, loss: 0.8, val_accuracy: 0.62, val_loss: 0.85 },
					{ epoch: 5, accuracy: 0.72, loss: 0.6, val_accuracy: 0.7, val_loss: 0.65 },
					{ epoch: 10, accuracy: 0.78, loss: 0.5, val_accuracy: 0.76, val_loss: 0.55 },
					{ epoch: 15, accuracy: 0.82, loss: 0.4, val_accuracy: 0.8, val_loss: 0.45 },
					{ epoch: 20, accuracy: 0.85, loss: 0.35, val_accuracy: 0.83, val_loss: 0.4 },
					{ epoch: 25, accuracy: 0.87, loss: 0.3, val_accuracy: 0.85, val_loss: 0.35 },
					{ epoch: 30, accuracy: 0.89, loss: 0.25, val_accuracy: 0.87, val_loss: 0.3 },
				],
				featureImportance: [
					{ feature: 'Color Features', importance: 0.35, color: '#3B82F6' },
					{ feature: 'Texture Features', importance: 0.28, color: '#10B981' },
					{ feature: 'Shape Features', importance: 0.22, color: '#8B5CF6' },
					{ feature: 'Edge Features', importance: 0.15, color: '#F59E0B' },
				],
			});
			setIsLoading(false);
		}, 1000);
	}, []);

	const tabs = [
		{ id: 'performance', label: 'Model Performance', icon: '📊' },
		{ id: 'distribution', label: 'Class Distribution', icon: '🥧' },
		{ id: 'training', label: 'Training History', icon: '📈' },
		{ id: 'features', label: 'Feature Importance', icon: '🔍' },
	];

	if (isLoading) {
		return (
			<div className='flex items-center justify-center py-12'>
				<div className='animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600'></div>
				<span className='ml-3 text-gray-600'>Loading visualizations...</span>
			</div>
		);
	}

	if (!data) {
		return (
			<div className='bg-yellow-50 border border-yellow-200 rounded-lg p-6'>
				<p className='text-yellow-700'>No visualization data available</p>
			</div>
		);
	}

	return (
		<div className='space-y-6'>
			<h3 className='text-xl font-semibold text-gray-900'>Data Visualizations & Insights</h3>

			{/* Navigation Tabs */}
			<div className='flex space-x-1 bg-gray-100 p-1 rounded-lg'>
				{tabs.map((tab) => (
					<button
						key={tab.id}
						onClick={() => setActiveTab(tab.id)}
						className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors duration-200 ${
							activeTab === tab.id
								? 'bg-white text-blue-600 shadow-sm'
								: 'text-gray-600 hover:text-gray-900'
						}`}
					>
						<span className='mr-2'>{tab.icon}</span>
						{tab.label}
					</button>
				))}
			</div>

			{/* Visualization Content */}
			<div className='bg-white border rounded-lg p-6'>
				{activeTab === 'performance' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Model Performance Metrics</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<BarChart data={data.modelPerformance}>
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
								<strong>Interpretation:</strong> The model shows strong performance across all metrics,
								with particularly high recall indicating good detection of malnourished cases.
							</p>
						</div>
					</div>
				)}

				{activeTab === 'distribution' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Class Distribution</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<PieChart>
									<Pie
										data={data.classDistribution}
										cx='50%'
										cy='50%'
										labelLine={false}
										label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
										outerRadius={80}
										fill='#8884d8'
										dataKey='value'
									>
										{data.classDistribution.map((entry, index) => (
											<Cell key={`cell-${index}`} fill={entry.color} />
										))}
									</Pie>
									<Tooltip />
								</PieChart>
							</ResponsiveContainer>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> The dataset shows a 65:35 split between normal and
								malnourished cases, indicating a slight class imbalance that the model handles well.
							</p>
						</div>
					</div>
				)}

				{activeTab === 'training' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Training History</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<LineChart data={data.trainingHistory}>
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
									<Line type='monotone' dataKey='val_loss' stroke='#F59E0B' name='Validation Loss' />
								</LineChart>
							</ResponsiveContainer>
						</div>
						<div className='mt-4 text-sm text-gray-600'>
							<p>
								<strong>Interpretation:</strong> The training shows good convergence with validation
								metrics closely following training metrics, indicating no overfitting.
							</p>
						</div>
					</div>
				)}

				{activeTab === 'features' && (
					<div>
						<h4 className='text-lg font-medium text-gray-900 mb-4'>Feature Importance Analysis</h4>
						<div className='h-80'>
							<ResponsiveContainer width='100%' height='100%'>
								<BarChart data={data.featureImportance} layout='horizontal'>
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
								<strong>Interpretation:</strong> Color features are most important (35%), followed by
								texture (28%), indicating that visual characteristics are key for malnutrition
								detection.
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
							<strong>🎯 Model Performance:</strong> High accuracy (89%) with balanced precision and
							recall
						</p>
						<p>
							<strong>📊 Data Balance:</strong> Slight class imbalance handled well by the model
						</p>
					</div>
					<div>
						<p>
							<strong>🔍 Feature Importance:</strong> Color and texture features are most predictive
						</p>
						<p>
							<strong>📈 Training Stability:</strong> No overfitting observed during training
						</p>
					</div>
				</div>
			</div>
		</div>
	);
};

export default DataVisualization;
