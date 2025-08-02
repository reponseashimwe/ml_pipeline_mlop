'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import api from '../lib/api';

type RetrainingStatus = 'idle' | 'starting' | 'training' | 'completed' | 'failed';
type ImageLabel = 'malnourished' | 'overnourished' | 'unlabeled';

interface PendingImage {
	id: string;
	file: File;
	preview: string;
	label: ImageLabel;
	name: string;
	size: number;
}

interface TrainingProgress {
	epoch: number;
	total_epochs: number;
	accuracy: number;
	loss: number;
	val_accuracy: number;
	val_loss: number;
	status: string;
}

const RetrainingPanel: React.FC = () => {
	const [status, setStatus] = useState<RetrainingStatus>('idle');
	const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
	const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
	const [logs, setLogs] = useState<string[]>([]);
	const [isPolling, setIsPolling] = useState(false);

	// Dropzone for image upload
	const onDrop = useCallback((acceptedFiles: File[]) => {
		const newImages: PendingImage[] = acceptedFiles.map((file) => ({
			id: Math.random().toString(36).substr(2, 9),
			file,
			preview: URL.createObjectURL(file),
			label: 'unlabeled' as ImageLabel,
			name: file.name,
			size: file.size,
		}));

		setPendingImages((prev) => [...prev, ...newImages]);
		toast.success(`Added ${acceptedFiles.length} images for labeling`);
	}, []);

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: {
			'image/*': ['.jpeg', '.jpg', '.png'],
		},
		multiple: true,
	});

	// Label an image
	const labelImage = (imageId: string, label: ImageLabel) => {
		setPendingImages((prev) => prev.map((img) => (img.id === imageId ? { ...img, label } : img)));
	};

	// Remove an image
	const removeImage = (imageId: string) => {
		setPendingImages((prev) => {
			const image = prev.find((img) => img.id === imageId);
			if (image) {
				URL.revokeObjectURL(image.preview);
			}
			return prev.filter((img) => img.id !== imageId);
		});
	};

	// Upload labeled images
	const uploadLabeledImages = async () => {
		const labeledImages = pendingImages.filter((img) => img.label !== 'unlabeled');

		if (labeledImages.length === 0) {
			toast.error('Please label at least one image before uploading');
			return;
		}

		try {
			setLogs((prev) => [...prev, '📤 Uploading labeled images...']);

			// Create FormData with labeled images
			const formData = new FormData();
			labeledImages.forEach((img) => {
				formData.append('files', img.file);
				formData.append('labels', img.label);
			});

			const response = await api.uploadLabeledTrainingData(formData);

			if (response.success) {
				// Clear uploaded images
				labeledImages.forEach((img) => URL.revokeObjectURL(img.preview));
				setPendingImages((prev) => prev.filter((img) => img.label === 'unlabeled'));

				setLogs((prev) => [...prev, `✅ Uploaded ${labeledImages.length} labeled images`]);
				toast.success(`Successfully uploaded ${labeledImages.length} labeled images`);
			} else {
				throw new Error(response.message);
			}
		} catch (error) {
			setLogs((prev) => [
				...prev,
				`❌ Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
			]);
			toast.error('Failed to upload labeled images');
		}
	};

	// Start retraining
	const handleRetrain = async () => {
		setStatus('starting');
		setLogs([]);
		setTrainingProgress(null);

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
				const progress = await api.getTrainingProgress(jobId);

				if (progress.status === 'completed') {
					setStatus('completed');
					setTrainingProgress(progress);
					setLogs((prev) => [...prev, '✅ Training completed successfully!']);
					setIsPolling(false);
					clearInterval(pollInterval);
					toast.success('Model retraining completed!');
				} else if (progress.status === 'failed') {
					setStatus('failed');
					setLogs((prev) => [...prev, `❌ Training failed: ${progress.error}`]);
					setIsPolling(false);
					clearInterval(pollInterval);
					toast.error('Model retraining failed');
				} else {
					setTrainingProgress(progress);
					setLogs((prev) => [
						...prev,
						`📊 Epoch ${progress.epoch}/${progress.total_epochs} - Accuracy: ${(
							progress.accuracy * 100
						).toFixed(2)}%`,
					]);
				}
			} catch (error) {
				console.error('Error polling training progress:', error);
			}
		}, 2000); // Poll every 2 seconds

		// Cleanup on unmount
		return () => clearInterval(pollInterval);
	};

	// Reset status
	const resetStatus = () => {
		setStatus('idle');
		setLogs([]);
		setTrainingProgress(null);
		setIsPolling(false);
	};

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			pendingImages.forEach((img) => URL.revokeObjectURL(img.preview));
		};
	}, [pendingImages]);

	const labeledCount = pendingImages.filter((img) => img.label !== 'unlabeled').length;
	const unlabeledCount = pendingImages.filter((img) => img.label === 'unlabeled').length;

	return (
		<div className='space-y-6'>
			{/* Image Upload and Labeling Section */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>📚 Upload Training Images</h3>
				<p className='text-gray-600 mb-4 text-sm'>
					Upload images and label them for model retraining. Images will be organized by class.
				</p>

				{/* Drag & Drop Area */}
				<div
					{...getRootProps()}
					className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
						isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
					}`}
				>
					<input {...getInputProps()} />
					<div className='space-y-2'>
						<p className='text-gray-600'>
							{isDragActive ? 'Drop images here...' : 'Drag & drop images here, or click to select'}
						</p>
						<p className='text-xs text-gray-500'>Supports: JPG, JPEG, PNG (Multiple files)</p>
					</div>
				</div>

				{/* Pending Images */}
				{pendingImages.length > 0 && (
					<div className='mt-6'>
						<div className='flex justify-between items-center mb-4'>
							<h4 className='font-semibold'>📋 Pending Images ({pendingImages.length})</h4>
							<div className='text-sm text-gray-600'>
								<span className='text-green-600'>{labeledCount} labeled</span>
								{' • '}
								<span className='text-orange-600'>{unlabeledCount} unlabeled</span>
							</div>
						</div>

						<div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto'>
							{pendingImages.map((image) => (
								<div key={image.id} className='border rounded-lg p-3 bg-gray-50'>
									<div className='relative mb-3'>
										<img
											src={image.preview}
											alt={image.name}
											className='w-full h-32 object-cover rounded'
										/>
										<button
											onClick={() => removeImage(image.id)}
											className='absolute top-1 right-1 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600'
										>
											×
										</button>
									</div>

									<div className='space-y-2'>
										<p className='text-xs text-gray-600 truncate'>{image.name}</p>
										<p className='text-xs text-gray-500'>{(image.size / 1024).toFixed(1)} KB</p>

										{/* Labeling Buttons */}
										<div className='flex space-x-1'>
											<button
												onClick={() => labelImage(image.id, 'malnourished')}
												className={`flex-1 px-2 py-1 text-xs rounded ${
													image.label === 'malnourished'
														? 'bg-red-500 text-white'
														: 'bg-red-100 text-red-700 hover:bg-red-200'
												}`}
											>
												Malnourished
											</button>
											<button
												onClick={() => labelImage(image.id, 'overnourished')}
												className={`flex-1 px-2 py-1 text-xs rounded ${
													image.label === 'overnourished'
														? 'bg-orange-500 text-white'
														: 'bg-orange-100 text-orange-700 hover:bg-orange-200'
												}`}
											>
												Overnourished
											</button>
										</div>

										{image.label !== 'unlabeled' && (
											<div className='text-xs text-green-600 font-medium'>
												✓ Labeled as {image.label}
											</div>
										)}
									</div>
								</div>
							))}
						</div>

						{/* Upload Button */}
						{labeledCount > 0 && (
							<div className='mt-4 flex justify-end'>
								<button
									onClick={uploadLabeledImages}
									className='px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors'
								>
									📤 Upload {labeledCount} Labeled Images
								</button>
							</div>
						)}
					</div>
				)}
			</div>

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

				{/* Status and Progress */}
				{status !== 'idle' && (
					<div className='p-4 bg-gray-50 rounded-lg'>
						<div className='flex items-center mb-3'>
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
								<div className='w-full bg-gray-200 rounded-full h-2 mb-2'>
									<div
										className='bg-blue-600 h-2 rounded-full transition-all duration-300'
										style={{
											width: `${(trainingProgress.epoch / trainingProgress.total_epochs) * 100}%`,
										}}
									></div>
								</div>

								{/* Metrics */}
								<div className='grid grid-cols-2 gap-4 text-sm'>
									<div>
										<span className='text-gray-600'>Accuracy:</span>
										<span className='ml-2 font-medium'>
											{(trainingProgress.accuracy * 100).toFixed(2)}%
										</span>
									</div>
									<div>
										<span className='text-gray-600'>Loss:</span>
										<span className='ml-2 font-medium'>{trainingProgress.loss.toFixed(4)}</span>
									</div>
									<div>
										<span className='text-gray-600'>Val Accuracy:</span>
										<span className='ml-2 font-medium'>
											{(trainingProgress.val_accuracy * 100).toFixed(2)}%
										</span>
									</div>
									<div>
										<span className='text-gray-600'>Val Loss:</span>
										<span className='ml-2 font-medium'>{trainingProgress.val_loss.toFixed(4)}</span>
									</div>
								</div>
							</div>
						)}

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
						<li>• Incorporates newly uploaded labeled training data</li>
						<li>• Applies advanced data augmentation</li>
						<li>• Optimizes model weights for better performance</li>
						<li>• Automatically saves the improved model</li>
						<li>• Provides real-time training progress</li>
					</ul>
				</div>
			</div>
		</div>
	);
};

export default RetrainingPanel;
