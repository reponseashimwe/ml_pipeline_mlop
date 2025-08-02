'use client';

import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import api from '../lib/api';

type RetrainingStatus = 'idle' | 'starting' | 'training' | 'completed' | 'failed';

const RetrainingPanel: React.FC = () => {
	const [status, setStatus] = useState<RetrainingStatus>('idle');
	const [logs, setLogs] = useState<string[]>([]);

	const handleRetrain = async () => {
		setStatus('starting');
		setLogs([]);

		try {
			setLogs((prev) => [...prev, '🚀 Initiating model retraining...']);

			const response = await api.retrainModel();

			if (response.success) {
				setStatus('training');
				setLogs((prev) => [...prev, '✅ Retraining job started successfully']);
				setLogs((prev) => [...prev, '⏳ Training in progress... (this may take several minutes)']);
				toast.success('Model retraining started successfully!');
			} else {
				setStatus('failed');
				setLogs((prev) => [...prev, `❌ Failed to start retraining: ${response.message}`]);
				toast.error('Failed to start retraining');
			}
		} catch (error) {
			setStatus('failed');
			setLogs((prev) => [...prev, `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`]);
			toast.error('Failed to start retraining');
		}
	};

	const resetStatus = () => {
		setStatus('idle');
		setLogs([]);
	};

	return (
		<div className='bg-white p-6 rounded-lg shadow-md'>
			<h3 className='text-lg font-semibold mb-4'>Model Retraining</h3>

			<div className='space-y-4'>
				<div className='flex items-center justify-between'>
					<div>
						<p className='text-gray-600 mb-2'>
							Retrain the model with newly uploaded data to improve performance.
						</p>
						<p className='text-sm text-gray-500'>
							This process will use both original and newly uploaded training data.
						</p>
					</div>

					<button
						onClick={handleRetrain}
						disabled={status === 'starting' || status === 'training'}
						className={`px-6 py-2 rounded-lg font-medium transition-colors ${
							status === 'starting' || status === 'training'
								? 'bg-gray-300 text-gray-500 cursor-not-allowed'
								: 'bg-blue-600 text-white hover:bg-blue-700'
						}`}
					>
						{status === 'starting'
							? 'Starting...'
							: status === 'training'
							? 'Training...'
							: 'Start Retraining'}
					</button>
				</div>

				{/* Status Indicator */}
				{status !== 'idle' && (
					<div className='p-4 bg-gray-50 rounded-lg'>
						<div className='flex items-center mb-2'>
							<div
								className={`w-3 h-3 rounded-full mr-2 ${
									status === 'starting'
										? 'bg-yellow-500'
										: status === 'training'
										? 'bg-blue-500'
										: status === 'completed'
										? 'bg-green-500'
										: 'bg-red-500'
								}`}
							></div>
							<span className='font-medium capitalize'>{status}</span>
						</div>

						{/* Logs */}
						{logs.length > 0 && (
							<div className='mt-3'>
								<h4 className='text-sm font-medium mb-2'>Progress Log:</h4>
								<div className='bg-black text-green-400 p-3 rounded text-sm font-mono max-h-40 overflow-y-auto'>
									{logs.map((log, index) => (
										<div key={index}>{log}</div>
									))}
								</div>
							</div>
						)}

						{/* Reset Button */}
						{(status === 'completed' || status === 'failed') && (
							<button
								onClick={resetStatus}
								className='mt-3 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700'
							>
								Reset
							</button>
						)}
					</div>
				)}

				{/* Information */}
				<div className='bg-blue-50 p-4 rounded-lg'>
					<h4 className='font-medium text-blue-900 mb-2'>What happens during retraining?</h4>
					<ul className='text-sm text-blue-800 space-y-1'>
						<li>• Uses existing model as a pre-trained base</li>
						<li>• Incorporates newly uploaded training data</li>
						<li>• Applies advanced data augmentation</li>
						<li>• Optimizes model weights for better performance</li>
						<li>• Automatically saves the improved model</li>
					</ul>
				</div>
			</div>
		</div>
	);
};

export default RetrainingPanel;
