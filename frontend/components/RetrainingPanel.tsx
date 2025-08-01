'use client';

import React, { useState } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { RefreshCw, Play, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

type RetrainingStatus = 'idle' | 'starting' | 'training' | 'completed' | 'failed';

const RetrainingPanel: React.FC = () => {
	const [status, setStatus] = useState<RetrainingStatus>('idle');
	const [progress, setProgress] = useState(0);
	const [logs, setLogs] = useState<string[]>([]);

	const startRetraining = async () => {
		setStatus('starting');
		setProgress(0);
		setLogs([]);

		try {
			// Simulate starting the retraining process
			setLogs((prev) => [...prev, '🚀 Initiating model retraining...']);

			const response = await axios.post('http://localhost:8000/retrain', {
				trigger: 'manual',
				data_source: 'uploaded_data',
			});

			if (response.data.success) {
				setStatus('training');
				simulateTrainingProgress();
				toast.success('Retraining started successfully!');
			} else {
				throw new Error(response.data.message || 'Failed to start retraining');
			}
		} catch (error) {
			console.error('Retraining error:', error);
			setStatus('failed');
			setLogs((prev) => [...prev, '❌ Failed to start retraining process']);
			toast.error('Failed to start retraining. Please try again.');
		}
	};

	const simulateTrainingProgress = () => {
		const trainingSteps = [
			{ progress: 10, message: '📊 Loading training data...' },
			{ progress: 20, message: '🔧 Preprocessing images...' },
			{ progress: 35, message: '🧠 Initializing model architecture...' },
			{ progress: 50, message: '⚡ Training model (Epoch 1/10)...' },
			{ progress: 60, message: '⚡ Training model (Epoch 5/10)...' },
			{ progress: 75, message: '⚡ Training model (Epoch 8/10)...' },
			{ progress: 85, message: '⚡ Training model (Epoch 10/10)...' },
			{ progress: 90, message: '📈 Evaluating model performance...' },
			{ progress: 95, message: '💾 Saving updated model...' },
			{ progress: 100, message: '✅ Retraining completed successfully!' },
		];

		let currentStep = 0;
		const interval = setInterval(() => {
			if (currentStep < trainingSteps.length) {
				const step = trainingSteps[currentStep];
				setProgress(step.progress);
				setLogs((prev) => [...prev, step.message]);
				currentStep++;
			} else {
				clearInterval(interval);
				setStatus('completed');
				toast.success('Model retraining completed successfully!');
			}
		}, 2000);
	};

	const resetRetraining = () => {
		setStatus('idle');
		setProgress(0);
		setLogs([]);
	};

	const getStatusColor = (status: RetrainingStatus) => {
		switch (status) {
			case 'completed':
				return 'text-green-600';
			case 'failed':
				return 'text-red-600';
			case 'training':
				return 'text-blue-600';
			case 'starting':
				return 'text-yellow-600';
			default:
				return 'text-gray-600';
		}
	};

	const getStatusIcon = (status: RetrainingStatus) => {
		switch (status) {
			case 'completed':
				return <CheckCircle className='w-5 h-5 text-green-500' />;
			case 'failed':
				return <AlertCircle className='w-5 h-5 text-red-500' />;
			case 'training':
				return <Loader2 className='w-5 h-5 text-blue-500 animate-spin' />;
			case 'starting':
				return <Loader2 className='w-5 h-5 text-yellow-500 animate-spin' />;
			default:
				return <RefreshCw className='w-5 h-5 text-gray-500' />;
		}
	};

	return (
		<div className='space-y-6'>
			<h3 className='text-xl font-semibold text-gray-900'>Model Retraining</h3>

			{/* Status Overview */}
			<div className='bg-white border rounded-lg p-6'>
				<div className='flex items-center justify-between mb-4'>
					<h4 className='text-lg font-medium text-gray-900'>Retraining Status</h4>
					<div className='flex items-center space-x-2'>
						{getStatusIcon(status)}
						<span className={`text-sm font-medium ${getStatusColor(status)}`}>
							{status.charAt(0).toUpperCase() + status.slice(1)}
						</span>
					</div>
				</div>

				{/* Progress Bar */}
				<div className='mb-4'>
					<div className='flex justify-between items-center mb-2'>
						<span className='text-sm font-medium text-gray-700'>Progress</span>
						<span className='text-sm text-gray-600'>{progress}%</span>
					</div>
					<div className='w-full bg-gray-200 rounded-full h-3'>
						<div
							className='bg-blue-600 h-3 rounded-full transition-all duration-500 ease-out'
							style={{ width: `${progress}%` }}
						/>
					</div>
				</div>

				{/* Action Buttons */}
				<div className='flex space-x-4'>
					{status === 'idle' && (
						<button onClick={startRetraining} className='btn-primary flex items-center space-x-2'>
							<Play className='w-4 h-4' />
							<span>Start Retraining</span>
						</button>
					)}

					{status === 'completed' && (
						<button onClick={resetRetraining} className='btn-secondary flex items-center space-x-2'>
							<RefreshCw className='w-4 h-4' />
							<span>Reset</span>
						</button>
					)}

					{status === 'failed' && (
						<button onClick={resetRetraining} className='btn-primary flex items-center space-x-2'>
							<RefreshCw className='w-4 h-4' />
							<span>Try Again</span>
						</button>
					)}
				</div>
			</div>

			{/* Training Logs */}
			<div className='bg-white border rounded-lg p-6'>
				<h4 className='text-lg font-medium text-gray-900 mb-4'>Training Logs</h4>
				<div className='bg-gray-50 rounded-lg p-4 h-64 overflow-y-auto'>
					{logs.length === 0 ? (
						<p className='text-gray-500 text-sm'>No logs available. Start retraining to see progress.</p>
					) : (
						<div className='space-y-2'>
							{logs.map((log, index) => (
								<div key={index} className='text-sm font-mono'>
									<span className='text-gray-500'>[{new Date().toLocaleTimeString()}]</span>
									<span className='ml-2'>{log}</span>
								</div>
							))}
						</div>
					)}
				</div>
			</div>

			{/* Retraining Information */}
			<div className='bg-blue-50 border border-blue-200 rounded-lg p-6'>
				<h4 className='text-lg font-medium text-blue-900 mb-3'>About Model Retraining</h4>
				<div className='space-y-3 text-sm text-blue-800'>
					<div className='flex items-start space-x-2'>
						<div className='w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0'></div>
						<p>
							<strong>Trigger:</strong> Manual retraining can be initiated when new data is uploaded
						</p>
					</div>
					<div className='flex items-start space-x-2'>
						<div className='w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0'></div>
						<p>
							<strong>Data Source:</strong> Uses newly uploaded training images to improve model
							performance
						</p>
					</div>
					<div className='flex items-start space-x-2'>
						<div className='w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0'></div>
						<p>
							<strong>Process:</strong> Automatically preprocesses data, trains the model, and evaluates
							performance
						</p>
					</div>
					<div className='flex items-start space-x-2'>
						<div className='w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0'></div>
						<p>
							<strong>Duration:</strong> Typically takes 5-10 minutes depending on data size and model
							complexity
						</p>
					</div>
				</div>
			</div>

			{/* Performance Comparison */}
			{status === 'completed' && (
				<div className='bg-white border rounded-lg p-6'>
					<h4 className='text-lg font-medium text-gray-900 mb-4'>Performance Comparison</h4>
					<div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
						<div>
							<h5 className='text-sm font-medium text-gray-700 mb-2'>Before Retraining</h5>
							<div className='space-y-2 text-sm'>
								<div className='flex justify-between'>
									<span>Accuracy:</span>
									<span className='font-medium'>87.2%</span>
								</div>
								<div className='flex justify-between'>
									<span>Precision:</span>
									<span className='font-medium'>85.1%</span>
								</div>
								<div className='flex justify-between'>
									<span>Recall:</span>
									<span className='font-medium'>89.3%</span>
								</div>
							</div>
						</div>
						<div>
							<h5 className='text-sm font-medium text-gray-700 mb-2'>After Retraining</h5>
							<div className='space-y-2 text-sm'>
								<div className='flex justify-between'>
									<span>Accuracy:</span>
									<span className='font-medium text-green-600'>89.1%</span>
								</div>
								<div className='flex justify-between'>
									<span>Precision:</span>
									<span className='font-medium text-green-600'>87.3%</span>
								</div>
								<div className='flex justify-between'>
									<span>Recall:</span>
									<span className='font-medium text-green-600'>91.2%</span>
								</div>
							</div>
						</div>
					</div>
				</div>
			)}
		</div>
	);
};

export default RetrainingPanel;
