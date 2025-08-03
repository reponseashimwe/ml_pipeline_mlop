'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { toast } from 'react-hot-toast';
import api from '../lib/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

type RetrainingStatus = 'idle' | 'starting' | 'training' | 'completed' | 'failed';

interface TrainingProgress {
	epoch: number;
	total_epochs: number;
	accuracy: number;
	loss: number;
	val_accuracy: number;
	val_loss: number;
	status: string;
}

interface TrainingHistoryPoint {
	epoch: number;
	accuracy: number;
	loss: number;
	val_accuracy: number;
	val_loss: number;
}

const RetrainingPanel: React.FC = () => {
	const [status, setStatus] = useState<RetrainingStatus>('idle');
	const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
	const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryPoint[]>([]);
	const [logs, setLogs] = useState<string[]>([]);
	const [isPolling, setIsPolling] = useState(false);
	const lastDisplayedEpochRef = useRef<number>(-1);
	const logsEndRef = useRef<HTMLDivElement>(null);

	// Load existing training history on mount
	useEffect(() => {
		loadExistingTrainingHistory();
	}, []);

	const loadExistingTrainingHistory = async () => {
		try {
			const visualizationData = await api.getVisualizationData();
			if (visualizationData.training_history && visualizationData.training_history.length > 0) {
				setTrainingHistory(visualizationData.training_history);
			}
		} catch (error) {
			console.error('Failed to load existing training history:', error);
		}
	};

	// Auto-scroll logs to bottom
	const scrollToBottom = () => {
		logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	};

	useEffect(() => {
		scrollToBottom();
	}, [logs]);

	// Start retraining
	const handleRetrain = async () => {
		setStatus('starting');
		setLogs([]);
		setTrainingProgress(null);
		lastDisplayedEpochRef.current = -1;

		try {
			setLogs((prev) => [...prev, '🚀 Initiating model retraining...']);

			const response = await api.retrainModel();

			if (response.success) {
				setStatus('training');
				setLogs((prev) => [...prev, '✅ Retraining job started successfully']);
				setLogs((prev) => [...prev, '⏳ Training in progress... (this may take several minutes)']);

				// Start polling for progress
				setIsPolling(true);
				if (response.job_id) {
					startProgressPolling(response.job_id);
				}

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

	// Poll for training progress
	const startProgressPolling = async (jobId: string) => {
		const pollInterval = setInterval(async () => {
			try {
				console.log(`🔄 Polling progress for job: ${jobId}`);
				const progress = await api.getTrainingProgress(jobId);
				console.log(`📊 Progress received:`, progress);

				if (progress.status === 'completed') {
					setStatus('completed');
					setTrainingProgress(progress);
					setLogs((prev) => [...prev, '✅ Training completed successfully!']);
					setLogs((prev) => [...prev, '🎉 Model updated and ready for use!']);
					setIsPolling(false);
					clearInterval(pollInterval);
					toast.success('Model retraining completed!', { duration: 5000 });
				} else if (progress.status === 'failed') {
					setStatus('failed');
					setLogs((prev) => [...prev, `❌ Training failed: ${progress.error}`]);
					setIsPolling(false);
					clearInterval(pollInterval);
					toast.error('Model retraining failed');
				} else if (progress.status === 'training' || progress.status === 'starting') {
					setTrainingProgress(progress);

					// Add to training history
					setTrainingHistory((prev) => {
						const newPoint: TrainingHistoryPoint = {
							epoch: progress.epoch,
							accuracy: progress.accuracy,
							loss: progress.loss,
							val_accuracy: progress.val_accuracy,
							val_loss: progress.val_loss,
						};

						// Check if we already have this epoch
						const existingIndex = prev.findIndex((p) => p.epoch === progress.epoch);
						if (existingIndex >= 0) {
							// Update existing point
							const updated = [...prev];
							updated[existingIndex] = newPoint;
							return updated;
						} else {
							// Add new point
							return [...prev, newPoint];
						}
					});

					// Check if this is a new epoch (not already displayed)
					const isNewEpoch = progress.epoch !== lastDisplayedEpochRef.current;

					if (isNewEpoch) {
						lastDisplayedEpochRef.current = progress.epoch;
						// Enhanced terminal-like logging - only for new epochs
						const epochLog = `📊 Epoch ${progress.epoch}/${progress.total_epochs}`;
						const accuracyLog = `   Accuracy: ${(progress.accuracy * 100).toFixed(2)}%`;
						const lossLog = `   Loss: ${progress.loss.toFixed(4)}`;
						const valAccuracyLog = `   Val Accuracy: ${(progress.val_accuracy * 100).toFixed(2)}%`;
						const valLossLog = `   Val Loss: ${progress.val_loss.toFixed(4)}`;

						// Check if validation accuracy improved
						const prevEpoch = trainingHistory.find((p) => p.epoch === progress.epoch - 1);
						let improvementLog = '';
						if (prevEpoch) {
							if (progress.val_accuracy > prevEpoch.val_accuracy) {
								improvementLog = `   ✅ Val accuracy improved from ${(
									prevEpoch.val_accuracy * 100
								).toFixed(2)}%`;
							} else if (progress.val_accuracy < prevEpoch.val_accuracy) {
								improvementLog = `   ⚠️ Val accuracy decreased from ${(
									prevEpoch.val_accuracy * 100
								).toFixed(2)}%`;
							} else {
								improvementLog = `   ➡️ Val accuracy unchanged`;
							}
						}

						setLogs((prev) => [
							...prev,
							epochLog,
							accuracyLog,
							lossLog,
							valAccuracyLog,
							valLossLog,
							...(improvementLog ? [improvementLog] : []),
							'',
						]);
					}
				} else {
					console.log(`⚠️ Unknown status: ${progress.status}`);
				}
			} catch (error) {
				console.error('❌ Error polling training progress:', error);
				setLogs((prev) => [
					...prev,
					`⚠️ Progress polling error: ${error instanceof Error ? error.message : 'Unknown error'}`,
				]);

				// If we get too many errors, stop polling
				const errorCount = logs.filter((log) => log.includes('Progress polling error')).length;
				if (errorCount > 5) {
					setLogs((prev) => [...prev, '❌ Too many polling errors, stopping progress updates']);
					setIsPolling(false);
					clearInterval(pollInterval);
				}
			}
		}, 3000); // Poll every 3 seconds

		// Cleanup on unmount
		return () => clearInterval(pollInterval);
	};

	// Reset status
	const resetStatus = () => {
		setStatus('idle');
		setTrainingProgress(null);
		setTrainingHistory([]);
		setLogs([]);
		setIsPolling(false);
	};

	return (
		<div className='space-y-6'>
			{/* Retraining Section */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>🔄 Model Retraining</h3>
				<p className='text-gray-600 mb-4 text-sm'>
					Start the retraining process with uploaded labeled data to improve model performance.
				</p>

				<div className='flex items-center justify-between mb-4'>
					<div>
						<p className='text-sm text-gray-600'>
							This process will use both original and newly uploaded training data.
						</p>
						<p className='text-xs text-gray-500 mt-1'>
							💡 Tip: Upload at least 20 images per class for better results
						</p>
					</div>

					<button
						onClick={handleRetrain}
						disabled={status === 'starting' || status === 'training'}
						className={`px-6 py-3 rounded-lg font-medium transition-colors ${
							status === 'starting' || status === 'training'
								? 'bg-gray-300 text-gray-500 cursor-not-allowed'
								: 'bg-blue-600 text-white hover:bg-blue-700'
						}`}
					>
						{status === 'starting'
							? '🚀 Starting...'
							: status === 'training'
							? '⏳ Training...'
							: '🔄 Start Retraining'}
					</button>
				</div>

				{/* Status and Progress */}
				{status !== 'idle' && (
					<div className='p-4 bg-gray-50 rounded-lg'>
						<div className='flex items-center mb-3'>
							<div
								className={`w-3 h-3 rounded-full mr-2 ${
									status === 'starting'
										? 'bg-yellow-500 animate-pulse'
										: status === 'training'
										? 'bg-blue-500 animate-pulse'
										: status === 'completed'
										? 'bg-green-500'
										: 'bg-red-500'
								}`}
							></div>
							<span className='font-medium capitalize'>{status}</span>
							{status === 'training' && (
								<span className='ml-2 text-sm text-gray-600'>
									(Est.{' '}
									{Math.max(
										5,
										(trainingProgress?.total_epochs || 20) - (trainingProgress?.epoch || 0)
									)}{' '}
									min remaining)
								</span>
							)}
						</div>

						{/* Training Progress */}
						{trainingProgress && (
							<div className='mb-4 p-3 bg-white rounded border'>
								<div className='flex justify-between items-center mb-2'>
									<span className='text-sm font-medium'>Training Progress</span>
									<span className='text-sm text-gray-600'>
										Epoch {trainingProgress.epoch}/{trainingProgress.total_epochs}
									</span>
								</div>

								{/* Progress Bar */}
								<div className='w-full bg-gray-200 rounded-full h-3 mb-3'>
									<div
										className='bg-blue-600 h-3 rounded-full transition-all duration-300'
										style={{
											width: `${(trainingProgress.epoch / trainingProgress.total_epochs) * 100}%`,
										}}
									></div>
								</div>

								{/* Metrics */}
								<div className='grid grid-cols-2 gap-4 text-sm'>
									<div className='bg-blue-50 p-2 rounded'>
										<span className='text-gray-600'>Training Accuracy:</span>
										<span className='ml-2 font-bold text-blue-700'>
											{(trainingProgress.accuracy * 100).toFixed(1)}%
										</span>
									</div>
									<div className='bg-green-50 p-2 rounded'>
										<span className='text-gray-600'>Validation Accuracy:</span>
										<span className='ml-2 font-bold text-green-700'>
											{(trainingProgress.val_accuracy * 100).toFixed(1)}%
										</span>
									</div>
									<div className='bg-orange-50 p-2 rounded'>
										<span className='text-gray-600'>Training Loss:</span>
										<span className='ml-2 font-medium text-orange-700'>
											{trainingProgress.loss.toFixed(4)}
										</span>
									</div>
									<div className='bg-purple-50 p-2 rounded'>
										<span className='text-gray-600'>Validation Loss:</span>
										<span className='ml-2 font-medium text-purple-700'>
											{trainingProgress.val_loss.toFixed(4)}
										</span>
									</div>
								</div>
							</div>
						)}

						{/* Logs */}
						{logs.length > 0 && (
							<div className='mt-3'>
								<div className='flex items-center justify-between mb-2'>
									<h4 className='text-sm font-medium'>🖥️ Training Terminal</h4>
									<div className='flex items-center space-x-2'>
										<div className='flex items-center space-x-1'>
											<div className='w-2 h-2 bg-red-500 rounded-full'></div>
											<div className='w-2 h-2 bg-yellow-500 rounded-full'></div>
											<div className='w-2 h-2 bg-green-500 rounded-full'></div>
										</div>
										<span className='text-xs text-gray-500'>Live Output</span>
									</div>
								</div>
								<div className='bg-gray-900 text-green-400 p-4 rounded-lg text-sm font-mono h-96 overflow-y-auto border border-gray-700 shadow-inner'>
									<div className='mb-2 text-gray-400 text-xs'>
										<span className='text-yellow-400'>$</span> python retrain_model.py
									</div>
									{logs.map((log, index) => (
										<div
											key={index}
											className={`mb-1 ${
												log.startsWith('📊')
													? 'text-blue-400 font-semibold'
													: log.startsWith('✅')
													? 'text-green-400'
													: log.startsWith('❌')
													? 'text-red-400'
													: log.startsWith('⚠️')
													? 'text-yellow-400'
													: log.startsWith('🚀')
													? 'text-purple-400'
													: log.startsWith('⏳')
													? 'text-cyan-400'
													: log.startsWith('   ')
													? 'text-gray-300 ml-4'
													: log === ''
													? 'text-transparent'
													: 'text-green-400'
											}`}
										>
											{log}
										</div>
									))}
									<div ref={logsEndRef} />
									{status === 'training' && (
										<div className='text-green-400 animate-pulse'>
											<span className='inline-block w-2 h-4 bg-green-400'></span>
										</div>
									)}
								</div>
							</div>
						)}

						{/* Reset Button */}
						{(status === 'completed' || status === 'failed') && (
							<button
								onClick={resetStatus}
								className='mt-3 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700'
							>
								🔄 Reset
							</button>
						)}
					</div>
				)}

				{/* Information */}
				<div className='bg-blue-50 p-4 rounded-lg mt-4'>
					<h4 className='font-medium text-blue-900 mb-2'>🎯 Retraining Process</h4>
					<div className='text-sm text-blue-800'>
						<p>• Uses existing model + new uploaded data</p>
						<p>• Applies data augmentation & optimization</p>
						<p>• Saves improved model automatically</p>
						<p>• Clears uploaded data after completion</p>
					</div>
				</div>
			</div>
		</div>
	);
};

export default RetrainingPanel;
