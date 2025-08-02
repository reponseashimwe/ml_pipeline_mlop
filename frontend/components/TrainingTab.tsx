'use client';

import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import api from '../lib/api';

const TrainingTab: React.FC = () => {
	const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
	const [isRetraining, setIsRetraining] = useState(false);

	const handleBulkUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = event.target.files;
		if (!files || files.length === 0) return;

		try {
			setUploadedFiles(Array.from(files));
			const result = await api.uploadTrainingData(files);
			toast.success(`Successfully uploaded ${files.length} files for training!`);
		} catch (error) {
			console.error('Error uploading files:', error);
			toast.error('Failed to upload files. Please try again.');
			setUploadedFiles([]);
		}
	};

	const handleRetrain = async () => {
		if (uploadedFiles.length === 0) {
			toast.error('Please upload some training data first!');
			return;
		}

		try {
			setIsRetraining(true);
			const result = await api.retrainModel();
			toast.success('Model retraining started! Check the logs for progress.');
		} catch (error) {
			console.error('Error starting retraining:', error);
			toast.error('Failed to start retraining. Please try again.');
		} finally {
			setIsRetraining(false);
		}
	};

	const clearUploadedFiles = () => {
		setUploadedFiles([]);
		toast.success('Uploaded files cleared');
	};

	return (
		<div className='space-y-6'>
			{/* Training Data Upload */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>📚 Training Data Upload</h3>
				<p className='text-gray-600 mb-4'>
					Upload multiple images to add to the training dataset. These images will be used for model
					retraining.
				</p>

				<div className='border-2 border-dashed border-gray-300 rounded-lg p-6'>
					<input
						type='file'
						multiple
						accept='image/*'
						onChange={handleBulkUpload}
						className='block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100'
					/>
					<p className='text-sm text-gray-500 mt-2'>
						Select multiple images. They will be stored for retraining purposes.
					</p>
				</div>

				{/* Show uploaded files */}
				{uploadedFiles.length > 0 && (
					<div className='mt-4 p-4 bg-green-50 rounded-lg'>
						<div className='flex justify-between items-center mb-2'>
							<h4 className='font-semibold text-green-800'>
								📁 Uploaded Training Files ({uploadedFiles.length})
							</h4>
							<button onClick={clearUploadedFiles} className='text-red-600 hover:text-red-800 text-sm'>
								Clear All
							</button>
						</div>
						<div className='space-y-1 max-h-40 overflow-y-auto'>
							{uploadedFiles.map((file, index) => (
								<div key={index} className='text-sm text-green-700 flex items-center'>
									<span className='mr-2'>📄</span>
									{file.name} ({(file.size / 1024).toFixed(1)} KB)
								</div>
							))}
						</div>
						<p className='text-xs text-green-600 mt-2'>
							These files are ready for model retraining. Use the retraining button below to start the
							process.
						</p>
					</div>
				)}
			</div>

			{/* Model Retraining */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>🔄 Model Retraining</h3>
				<p className='text-gray-600 mb-4'>
					Retrain the model with the uploaded training data to improve its performance.
				</p>

				<div className='bg-blue-50 p-4 rounded-lg mb-4'>
					<h4 className='font-semibold text-blue-800 mb-2'>⚠️ Important Notes:</h4>
					<ul className='text-sm text-blue-700 space-y-1'>
						<li>• Retraining will use all available training data (original + uploaded)</li>
						<li>• The process may take several minutes</li>
						<li>• The model will be automatically updated after retraining</li>
						<li>• Check the backend logs for detailed progress</li>
					</ul>
				</div>

				<button
					onClick={handleRetrain}
					disabled={isRetraining || uploadedFiles.length === 0}
					className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
						isRetraining || uploadedFiles.length === 0
							? 'bg-gray-300 text-gray-500 cursor-not-allowed'
							: 'bg-blue-600 text-white hover:bg-blue-700'
					}`}
				>
					{isRetraining ? (
						<div className='flex items-center'>
							<div className='animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2'></div>
							Retraining...
						</div>
					) : (
						'🚀 Start Model Retraining'
					)}
				</button>

				{uploadedFiles.length === 0 && (
					<p className='text-sm text-gray-500 mt-2'>Upload some training data first to enable retraining.</p>
				)}
			</div>

			{/* Training Status */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>📊 Training Status</h3>
				<p className='text-gray-600 mb-4'>Monitor the current status of your model and training process.</p>

				<div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
					<div className='bg-gray-50 p-4 rounded-lg'>
						<h4 className='font-semibold text-gray-800 mb-2'>Model Status</h4>
						<p className='text-sm text-gray-600'>
							Check the Model Status panel for current model information and performance metrics.
						</p>
					</div>
					<div className='bg-gray-50 p-4 rounded-lg'>
						<h4 className='font-semibold text-gray-800 mb-2'>Training Logs</h4>
						<p className='text-sm text-gray-600'>
							Monitor training progress through the backend console logs.
						</p>
					</div>
				</div>
			</div>
		</div>
	);
};

export default TrainingTab;
