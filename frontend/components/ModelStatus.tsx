'use client';

import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import api, { ModelStatus as ModelStatusType } from '../lib/api';

const ModelStatus: React.FC = () => {
	const [status, setStatus] = useState<ModelStatusType | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [uptimeSeconds, setUptimeSeconds] = useState(0);

	const fetchStatus = async () => {
		try {
			setLoading(true);
			const data = await api.getModelStatus();
			setStatus(data);
			setError(null);
			setLoading(false);

			// Parse uptime from the backend response
			if (data.uptime) {
				const uptimeMatch = data.uptime.match(/(\d+):(\d+):(\d+)/);
				if (uptimeMatch) {
					const hours = parseInt(uptimeMatch[1]);
					const minutes = parseInt(uptimeMatch[2]);
					const seconds = parseInt(uptimeMatch[3]);
					setUptimeSeconds(hours * 3600 + minutes * 60 + seconds);
				}
			}
		} catch (error) {
			console.error('Error fetching status:', error);
			setError('Failed to fetch model status');
		} finally {
			setLoading(false);
		}
	};

	useEffect(() => {
		fetchStatus();
		const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds
		return () => clearInterval(interval);
	}, []);

	// Live uptime counter
	useEffect(() => {
		const uptimeInterval = setInterval(() => {
			setUptimeSeconds((prev) => prev + 1);
		}, 1000);

		return () => clearInterval(uptimeInterval);
	}, []);

	// Format uptime
	const formatUptime = (seconds: number) => {
		const hours = Math.floor(seconds / 3600);
		const minutes = Math.floor((seconds % 3600) / 60);
		const secs = seconds % 60;
		return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs
			.toString()
			.padStart(2, '0')}`;
	};

	if (error && !status) {
		return (
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<div className='text-center'>
					<div className='text-red-500 mb-2'>⚠️</div>
					<p className='text-red-600'>{error}</p>
					<button
						onClick={fetchStatus}
						className='mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700'
					>
						Retry
					</button>
				</div>
			</div>
		);
	}

	if (!status) {
		return (
			<div className='bg-white p-6 rounded-lg shadow-md flex flex-col gap-4'>
				<div className='h-10 w-full bg-gray-200 animate-pulse max-w-md' />
				<div className='h-10 w-full bg-gray-200 animate-pulse rounded-lg' />
				<div className='h-10 w-full bg-gray-200 animate-pulse rounded-lg' />
			</div>
		);
	}

	return (
		<div className='bg-white p-6 rounded-lg shadow-md'>
			<div className='flex justify-between items-center mb-4'>
				<h3 className='text-lg font-semibold'>Model Status</h3>
				{loading && (
					<div className='flex items-center text-sm text-gray-500'>
						<div className='animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 mr-2'></div>
						Refreshing...
					</div>
				)}
			</div>

			<div className='space-y-4'>
				{/* Model Loading Status */}
				<div className='flex items-center justify-between p-3 bg-gray-50 rounded-lg'>
					<span className='font-medium'>Model Status:</span>
					<span
						className={`px-2 py-1 rounded-full text-sm font-medium ${
							status.is_loaded ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
						}`}
					>
						{status.is_loaded ? 'Loaded' : 'Not Loaded'}
					</span>
				</div>

				{/* Model Path */}
				<div className='flex items-center justify-between p-3 bg-gray-50 rounded-lg'>
					<span className='font-medium'>Model Path:</span>
					<span className='text-sm text-gray-600 font-mono'>{status.model_path}</span>
				</div>

				{/* Last Updated */}
				<div className='flex items-center justify-between p-3 bg-gray-50 rounded-lg'>
					<span className='font-medium'>Last Updated:</span>
					<span className='text-sm text-gray-600'>{new Date(status.last_updated).toLocaleString()}</span>
				</div>

				{/* System Metrics */}
				<div className='grid grid-cols-1 md:grid-cols-3 gap-4'>
					<div className='p-3 bg-blue-50 rounded-lg'>
						<div className='text-sm text-blue-600 font-medium'>Uptime</div>
						<div className='text-lg font-semibold text-blue-800'>{formatUptime(uptimeSeconds)}</div>
					</div>
					<div className='p-3 bg-green-50 rounded-lg'>
						<div className='text-sm text-green-600 font-medium'>Memory Usage</div>
						<div className='text-lg font-semibold text-green-800'>{status.memory_usage}</div>
					</div>
					<div className='p-3 bg-purple-50 rounded-lg'>
						<div className='text-sm text-purple-600 font-medium'>CPU Usage</div>
						<div className='text-lg font-semibold text-purple-800'>{status.cpu_usage}</div>
					</div>
				</div>

				{/* Performance Metrics */}
				<div className='p-4 bg-yellow-50 rounded-lg'>
					<h4 className='font-medium text-yellow-900 mb-3'>Performance Metrics</h4>
					<div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
						<div className='text-center'>
							<div className='text-2xl font-bold text-yellow-800'>
								{status.performance?.accuracy ? (status.performance.accuracy * 100).toFixed(1) : 'N/A'}%
							</div>
							<div className='text-sm text-yellow-700'>Accuracy</div>
						</div>
						<div className='text-center'>
							<div className='text-2xl font-bold text-yellow-800'>
								{status.performance?.precision
									? (status.performance.precision * 100).toFixed(1)
									: 'N/A'}
								%
							</div>
							<div className='text-sm text-yellow-700'>Precision</div>
						</div>
						<div className='text-center'>
							<div className='text-2xl font-bold text-yellow-800'>
								{status.performance?.recall ? (status.performance.recall * 100).toFixed(1) : 'N/A'}%
							</div>
							<div className='text-sm text-yellow-700'>Recall</div>
						</div>
						<div className='text-center'>
							<div className='text-2xl font-bold text-yellow-800'>
								{status.performance?.f1_score ? (status.performance.f1_score * 100).toFixed(1) : 'N/A'}%
							</div>
							<div className='text-sm text-yellow-700'>F1 Score</div>
						</div>
					</div>
				</div>

				{/* Refresh Button */}
				<div className='text-center'>
					<button
						onClick={fetchStatus}
						className='px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors'
					>
						{loading ? (
							<div className='animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 mr-2'></div>
						) : (
							'Refresh Status'
						)}
					</button>
				</div>
			</div>
		</div>
	);
};

export default ModelStatus;
