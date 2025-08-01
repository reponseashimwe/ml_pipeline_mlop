'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Activity, Clock, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';

interface ModelStatusData {
	is_loaded: boolean;
	model_path: string;
	last_updated: string;
	performance_metrics: {
		accuracy: number;
		precision: number;
		recall: number;
		f1_score: number;
	};
	system_status: {
		uptime: string;
		memory_usage: string;
		cpu_usage: string;
	};
}

const ModelStatus: React.FC = () => {
	const [status, setStatus] = useState<ModelStatusData | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		fetchStatus();
		const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds
		return () => clearInterval(interval);
	}, []);

	const fetchStatus = async () => {
		try {
			const response = await axios.get('http://localhost:8000/status');
			setStatus(response.data);
			setError(null);
		} catch (err) {
			setError('Failed to fetch model status');
			toast.error('Failed to fetch model status');
		} finally {
			setIsLoading(false);
		}
	};

	if (isLoading) {
		return (
			<div className='flex items-center justify-center py-12'>
				<div className='animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600'></div>
				<span className='ml-3 text-gray-600'>Loading model status...</span>
			</div>
		);
	}

	if (error) {
		return (
			<div className='bg-red-50 border border-red-200 rounded-lg p-6'>
				<div className='flex items-center'>
					<AlertCircle className='w-5 h-5 text-red-500 mr-2' />
					<span className='text-red-700'>{error}</span>
				</div>
			</div>
		);
	}

	if (!status) {
		return (
			<div className='bg-yellow-50 border border-yellow-200 rounded-lg p-6'>
				<div className='flex items-center'>
					<AlertCircle className='w-5 h-5 text-yellow-500 mr-2' />
					<span className='text-yellow-700'>No status data available</span>
				</div>
			</div>
		);
	}

	return (
		<div className='space-y-6'>
			<h3 className='text-xl font-semibold text-gray-900'>Model Status & Performance</h3>

			{/* Model Status Overview */}
			<div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
				<div className='bg-white border rounded-lg p-4'>
					<div className='flex items-center'>
						<div
							className={`w-3 h-3 rounded-full mr-3 ${status.is_loaded ? 'bg-green-500' : 'bg-red-500'}`}
						></div>
						<div>
							<p className='text-sm font-medium text-gray-900'>Model Status</p>
							<p className={`text-sm ${status.is_loaded ? 'text-green-600' : 'text-red-600'}`}>
								{status.is_loaded ? 'Loaded' : 'Not Loaded'}
							</p>
						</div>
					</div>
				</div>

				<div className='bg-white border rounded-lg p-4'>
					<div className='flex items-center'>
						<Clock className='w-5 h-5 text-gray-400 mr-3' />
						<div>
							<p className='text-sm font-medium text-gray-900'>Last Updated</p>
							<p className='text-sm text-gray-600'>
								{new Date(status.last_updated).toLocaleDateString()}
							</p>
						</div>
					</div>
				</div>

				<div className='bg-white border rounded-lg p-4'>
					<div className='flex items-center'>
						<Activity className='w-5 h-5 text-blue-500 mr-3' />
						<div>
							<p className='text-sm font-medium text-gray-900'>Uptime</p>
							<p className='text-sm text-gray-600'>{status.system_status.uptime}</p>
						</div>
					</div>
				</div>

				<div className='bg-white border rounded-lg p-4'>
					<div className='flex items-center'>
						<TrendingUp className='w-5 h-5 text-green-500 mr-3' />
						<div>
							<p className='text-sm font-medium text-gray-900'>Memory Usage</p>
							<p className='text-sm text-gray-600'>{status.system_status.memory_usage}</p>
						</div>
					</div>
				</div>
			</div>

			{/* Performance Metrics */}
			<div className='bg-white border rounded-lg p-6'>
				<h4 className='text-lg font-medium text-gray-900 mb-4'>Performance Metrics</h4>
				<div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
					<div className='text-center'>
						<p className='text-2xl font-bold text-blue-600'>
							{(status.performance_metrics.accuracy * 100).toFixed(1)}%
						</p>
						<p className='text-sm text-gray-600'>Accuracy</p>
					</div>
					<div className='text-center'>
						<p className='text-2xl font-bold text-green-600'>
							{(status.performance_metrics.precision * 100).toFixed(1)}%
						</p>
						<p className='text-sm text-gray-600'>Precision</p>
					</div>
					<div className='text-center'>
						<p className='text-2xl font-bold text-purple-600'>
							{(status.performance_metrics.recall * 100).toFixed(1)}%
						</p>
						<p className='text-sm text-gray-600'>Recall</p>
					</div>
					<div className='text-center'>
						<p className='text-2xl font-bold text-orange-600'>
							{(status.performance_metrics.f1_score * 100).toFixed(1)}%
						</p>
						<p className='text-sm text-gray-600'>F1 Score</p>
					</div>
				</div>
			</div>

			{/* System Resources */}
			<div className='bg-white border rounded-lg p-6'>
				<h4 className='text-lg font-medium text-gray-900 mb-4'>System Resources</h4>
				<div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
					<div>
						<div className='flex justify-between items-center mb-2'>
							<span className='text-sm font-medium text-gray-700'>CPU Usage</span>
							<span className='text-sm text-gray-600'>{status.system_status.cpu_usage}</span>
						</div>
						<div className='w-full bg-gray-200 rounded-full h-2'>
							<div
								className='bg-blue-600 h-2 rounded-full transition-all duration-300'
								style={{ width: `${parseFloat(status.system_status.cpu_usage)}%` }}
							/>
						</div>
					</div>

					<div>
						<div className='flex justify-between items-center mb-2'>
							<span className='text-sm font-medium text-gray-700'>Memory Usage</span>
							<span className='text-sm text-gray-600'>{status.system_status.memory_usage}</span>
						</div>
						<div className='w-full bg-gray-200 rounded-full h-2'>
							<div
								className='bg-green-600 h-2 rounded-full transition-all duration-300'
								style={{ width: `${parseFloat(status.system_status.memory_usage)}%` }}
							/>
						</div>
					</div>
				</div>
			</div>

			{/* Model Information */}
			<div className='bg-gray-50 border rounded-lg p-6'>
				<h4 className='text-lg font-medium text-gray-900 mb-4'>Model Information</h4>
				<div className='space-y-2'>
					<div className='flex justify-between'>
						<span className='text-sm text-gray-600'>Model Path:</span>
						<span className='text-sm font-medium text-gray-900'>{status.model_path}</span>
					</div>
					<div className='flex justify-between'>
						<span className='text-sm text-gray-600'>Model Type:</span>
						<span className='text-sm font-medium text-gray-900'>CNN - Malnutrition Detection</span>
					</div>
					<div className='flex justify-between'>
						<span className='text-sm text-gray-600'>Input Format:</span>
						<span className='text-sm font-medium text-gray-900'>224x224 RGB Images</span>
					</div>
				</div>
			</div>

			{/* Refresh Button */}
			<div className='flex justify-end'>
				<button onClick={fetchStatus} className='btn-primary flex items-center space-x-2'>
					<Activity className='w-4 h-4' />
					<span>Refresh Status</span>
				</button>
			</div>
		</div>
	);
};

export default ModelStatus;
