'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import api, { PredictionResult } from '../lib/api';

interface ImageUploadProps {
	onPrediction?: (result: PredictionResult) => void;
	isTrainingData?: boolean;
}

// Test files from the test data
const TEST_FILES = [
	{
		name: 'malnourished-001.jpg',
		url: 'malnourished-001.jpg',
		class: 'malnourished',
		description: 'Test malnourished child',
	},
	{
		name: 'malnourished-002.jpg',
		url: 'malnourished-002.jpg',
		class: 'malnourished',
		description: 'Test malnourished child',
	},
	{
		name: 'malnourished-003.jpg',
		url: 'malnourished-003.jpg',
		class: 'malnourished',
		description: 'Test malnourished child',
	},
	{
		name: 'normal-001.jpg',
		url: 'normal-001.jpg',
		class: 'normal',
		description: 'Test normal child',
	},
	{
		name: 'normal-002.jpg',
		url: 'normal-002.jpg',
		class: 'normal',
		description: 'Test normal child',
	},
	{
		name: 'normal-003.jpg',
		url: 'normal-003.jpg',
		class: 'normal',
		description: 'Test normal child',
	},
	{
		name: 'overnourished-001.jpg',
		url: 'overnourished-001.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
	{
		name: 'overnourished-002.jpg',
		url: 'overnourished-002.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
	{
		name: 'overnourished-003.jpg',
		url: 'overnourished-003.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
	{
		name: 'overnourished-004.jpg',
		url: 'overnourished-004.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
	{
		name: 'overnourished-005.jpg',
		url: 'overnourished-005.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
	{
		name: 'overnourished-006.jpg',
		url: 'overnourished-006.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
	{
		name: 'overnourished-007.jpg',
		url: 'overnourished-007.jpg',
		class: 'overnourished',
		description: 'Test overnourished child',
	},
];

const ImageUpload: React.FC<ImageUploadProps> = ({ onPrediction, isTrainingData = false }) => {
	const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [selectedFile, setSelectedFile] = useState<File | null>(null);
	const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

	const onDrop = useCallback(
		async (acceptedFiles: File[]) => {
			if (acceptedFiles.length === 0) return;

			const file = acceptedFiles[0];
			setSelectedFile(file);
			setIsUploading(true);

			// Create preview
			const reader = new FileReader();
			reader.onload = () => {
				setPreviewUrl(reader.result as string);
			};
			reader.readAsDataURL(file);

			try {
				const formData = new FormData();
				formData.append('image', file);
				const result = await api.predictImage(formData);
				setPredictionResult(result);
				onPrediction?.(result);
				toast.success('Prediction completed successfully!');
			} catch (error) {
				console.error('Error predicting image:', error);
				toast.error('Failed to predict image. Please try again.');
			} finally {
				setIsUploading(false);
			}
		},
		[onPrediction]
	);

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: {
			'image/*': ['.jpeg', '.jpg', '.png'],
		},
		multiple: false,
	});

	const handleTestFileClick = async (testFile: (typeof TEST_FILES)[0]) => {
		try {
			setIsUploading(true);

			// Construct the full API URL
			const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
			const fullImageUrl = `${apiBaseUrl}/api/test-images/${testFile.url}`;

			setPreviewUrl(fullImageUrl);
			setSelectedFile(null); // Clear selected file since this is a test file

			// Fetch the test image and convert to File object
			const response = await fetch(fullImageUrl);
			const blob = await response.blob();
			const file = new File([blob], testFile.name, { type: 'image/jpeg' });

			const formData = new FormData();
			formData.append('image', file);
			const result = await api.predictImage(formData);
			setPredictionResult(result);
			onPrediction?.(result);
			toast.success(`Test prediction completed for ${testFile.name}!`);
		} catch (error) {
			console.error('Error with test file:', error);
			toast.error('Failed to process test file. Please try again.');
		} finally {
			setIsUploading(false);
		}
	};

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

	const clearPreview = () => {
		setPreviewUrl(null);
		setSelectedFile(null);
		setPredictionResult(null);
	};

	const clearUploadedFiles = () => {
		setUploadedFiles([]);
		toast.success('All uploaded training files cleared.');
	};

	return (
		<div className='space-y-4'>
			{/* Single Image Prediction */}
			{!isTrainingData && (
				<div className='bg-white p-4 rounded-lg shadow-md'>
					<h3 className='text-lg font-semibold mb-3'>🔍 Single Image Prediction</h3>
					<p className='text-gray-600 mb-3 text-sm'>
						Upload a single image to get instant malnutrition prediction results.
					</p>

					<div className='grid grid-cols-1 md:grid-cols-4 lg:grid-cols-6 gap-4'>
						{/* Left Column - Drag & Drop Area */}
						<div className='md:col-span-3 lg:col-span-5'>
							<div
								{...getRootProps()}
								className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors h-32 flex items-center justify-center ${
									isDragActive
										? 'border-blue-500 bg-blue-50'
										: 'border-gray-300 hover:border-gray-400'
								}`}
							>
								<input {...getInputProps()} />
								{isUploading ? (
									<div className='flex items-center justify-center'>
										<div className='animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500'></div>
										<span className='ml-2 text-sm'>Processing...</span>
									</div>
								) : (
									<div>
										<p className='text-gray-600 text-sm'>
											{isDragActive
												? 'Drop the image here...'
												: 'Drag & drop an image here, or click to select'}
										</p>
										<p className='text-xs text-gray-500 mt-1'>Supports: JPG, JPEG, PNG</p>
									</div>
								)}
							</div>
						</div>

						{/* Right Column - Preview */}
						<div className='lg:col-span-1'>
							{previewUrl ? (
								<div className='relative'>
									<img
										src={previewUrl}
										alt='Preview'
										className='w-full h-32 object-cover rounded-lg border'
									/>
									<button
										onClick={clearPreview}
										className='absolute top-1 right-1 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs hover:bg-red-600'
									>
										×
									</button>
									{selectedFile && (
										<p className='text-xs text-gray-600 mt-1'>
											{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
										</p>
									)}
								</div>
							) : (
								<div className='h-32 bg-gray-100 rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center'>
									<p className='text-gray-500 text-center text-sm'>Image preview</p>
								</div>
							)}
						</div>
					</div>

					{/* Prediction Result */}
					{predictionResult && (
						<div className='mt-4 p-3 bg-gray-50 rounded-lg'>
							<h4 className='font-semibold mb-2 text-sm'>Prediction Result:</h4>
							<div className='grid grid-cols-2 gap-3 text-sm'>
								<div>
									<strong>Class:</strong> {predictionResult.class}
								</div>
								<div>
									<strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(2)}%
								</div>
								<div className='col-span-2'>
									<strong>Probabilities:</strong>
									<div className='mt-1 space-y-1'>
										<div>
											Malnourished:{' '}
											{(predictionResult.probabilities.malnourished * 100).toFixed(1)}%
										</div>
										<div>
											Overnourished:{' '}
											{(predictionResult.probabilities.overnourished * 100).toFixed(1)}%
										</div>
										<div>Normal: {(predictionResult.probabilities.normal * 100).toFixed(1)}%</div>
									</div>
								</div>
							</div>
						</div>
					)}
				</div>
			)}

			{/* Test Files */}
			{!isTrainingData && (
				<div className='bg-white p-4 rounded-lg shadow-md'>
					<h3 className='text-lg font-semibold mb-3'>🧪 Test Files</h3>
					<p className='text-gray-600 mb-3 text-sm'>
						Click on any test image below to quickly test the prediction system:
					</p>

					<div className='grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2'>
						{TEST_FILES.map((testFile, index) => (
							<button
								key={index}
								onClick={() => handleTestFileClick(testFile)}
								disabled={isUploading}
								className='group aspect-square relative overflow-hidden rounded-lg border hover:border-blue-500 transition-colors disabled:opacity-50'
							>
								<img
									src={`${
										process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'
									}/api/test-images/${testFile.url}`}
									alt={testFile.name}
									className='w-full h-full object-cover group-hover:scale-105 transition-transform'
								/>
								<div className='absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white text-xs p-1'>
									<div className='font-medium truncate'>{testFile.class}</div>
									<div className='truncate'>{testFile.name}</div>
								</div>
							</button>
						))}
					</div>
				</div>
			)}

			{/* Training Data Upload */}
			{isTrainingData && (
				<div className='bg-white p-4 rounded-lg shadow-md'>
					<h3 className='text-lg font-semibold mb-3'>📚 Training Data Upload</h3>
					<p className='text-gray-600 mb-3 text-sm'>
						Upload multiple images to add to the training dataset. These images will be used for model
						retraining.
					</p>

					<div className='border-2 border-dashed border-gray-300 rounded-lg p-4'>
						<input
							type='file'
							multiple
							accept='image/*'
							onChange={handleBulkUpload}
							className='block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100'
						/>
						<p className='text-xs text-gray-500 mt-2'>
							Select multiple images. They will be stored for retraining purposes.
						</p>
					</div>

					{/* Show uploaded files */}
					{uploadedFiles.length > 0 && (
						<div className='mt-3 p-3 bg-green-50 rounded-lg'>
							<div className='flex justify-between items-center mb-2'>
								<h4 className='font-semibold text-green-800 text-sm'>
									📁 Uploaded Training Files ({uploadedFiles.length})
								</h4>
								<button
									onClick={clearUploadedFiles}
									className='text-red-600 hover:text-red-800 text-xs'
								>
									Clear All
								</button>
							</div>
							<div className='space-y-1 max-h-24 overflow-y-auto'>
								{uploadedFiles.map((file, index) => (
									<div key={index} className='text-xs text-green-700 flex items-center'>
										<span className='mr-1'>📄</span>
										{file.name} ({(file.size / 1024).toFixed(1)} KB)
									</div>
								))}
							</div>
							<p className='text-xs text-green-600 mt-2'>
								These files are ready for model retraining. Use the retraining panel to start the
								process.
							</p>
						</div>
					)}
				</div>
			)}
		</div>
	);
};

export default ImageUpload;
