'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import api, { PredictionResult } from '../lib/api';

interface PredictionTabProps {
	onPredictionComplete?: (result: PredictionResult) => void;
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
];

const PredictionTab: React.FC<PredictionTabProps> = ({ onPredictionComplete }) => {
	const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [selectedFile, setSelectedFile] = useState<File | null>(null);

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
				onPredictionComplete?.(result);
				toast.success('Prediction completed successfully!');
			} catch (error) {
				console.error('Error predicting image:', error);
				toast.error('Failed to predict image. Please try again.');
			} finally {
				setIsUploading(false);
			}
		},
		[onPredictionComplete]
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
			onPredictionComplete?.(result);
			toast.success(`Test prediction completed for ${testFile.name}!`);
		} catch (error) {
			console.error('Error with test file:', error);
			toast.error('Failed to process test file. Please try again.');
		} finally {
			setIsUploading(false);
		}
	};

	const clearPreview = () => {
		setPreviewUrl(null);
		setSelectedFile(null);
		setPredictionResult(null);
	};

	return (
		<div className='space-y-6'>
			{/* Single Image Prediction */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>🔍 Single Image Prediction</h3>
				<p className='text-gray-600 mb-4'>
					Upload a single image to get instant malnutrition prediction results.
				</p>

				<div className='grid grid-cols-1 lg:grid-cols-3 gap-6'>
					{/* Left Column - Drag & Drop Area */}
					<div className='lg:col-span-2'>
						<div
							{...getRootProps()}
							className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors h-48 flex items-center justify-center ${
								isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
							}`}
						>
							<input {...getInputProps()} />
							{isUploading ? (
								<div className='flex items-center justify-center'>
									<div className='animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500'></div>
									<span className='ml-2'>Processing...</span>
								</div>
							) : (
								<div>
									<p className='text-gray-600'>
										{isDragActive
											? 'Drop the image here...'
											: 'Drag & drop an image here, or click to select'}
									</p>
									<p className='text-sm text-gray-500 mt-2'>Supports: JPG, JPEG, PNG</p>
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
									className='w-full h-48 object-cover rounded-lg border'
								/>
								<button
									onClick={clearPreview}
									className='absolute top-2 right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm hover:bg-red-600'
								>
									×
								</button>
								{selectedFile && (
									<p className='text-sm text-gray-600 mt-2'>
										File: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
									</p>
								)}
							</div>
						) : (
							<div className='h-48 bg-gray-100 rounded-lg border-2 border-dashed border-gray-300 flex items-center justify-center'>
								<p className='text-gray-500 text-center'>Image preview will appear here</p>
							</div>
						)}
					</div>
				</div>

				{/* Prediction Result */}
				{predictionResult && (
					<div className='mt-6 p-4 bg-gray-50 rounded-lg'>
						<h4 className='font-semibold mb-2'>Prediction Result:</h4>
						<div className='space-y-2'>
							<p>
								<strong>Class:</strong> {predictionResult.predicted_class}
							</p>
							<p>
								<strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(2)}%
							</p>
							<p>
								<strong>Interpretation:</strong> {predictionResult.interpretation}
							</p>
							<p>
								<strong>Recommendation:</strong> {predictionResult.recommendation}
							</p>
						</div>
					</div>
				)}
			</div>

			{/* Test Files */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<h3 className='text-lg font-semibold mb-4'>🧪 Test Files</h3>
				<p className='text-gray-600 mb-4'>
					Click on any test image below to quickly test the prediction system:
				</p>

				<div className='grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3'>
					{TEST_FILES.map((testFile, index) => (
						<button
							key={index}
							onClick={() => handleTestFileClick(testFile)}
							disabled={isUploading}
							className='group aspect-square relative overflow-hidden rounded-lg border hover:border-blue-500 transition-colors disabled:opacity-50'
						>
							<img
								src={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'}/api/test-images/${
									testFile.url
								}`}
								alt={testFile.name}
								className='w-full h-full object-cover group-hover:scale-105 transition-transform'
							/>
							<div className='absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white text-xs p-1'>
								<div className='font-medium'>{testFile.class}</div>
								<div className='truncate'>{testFile.name}</div>
							</div>
						</button>
					))}
				</div>
			</div>
		</div>
	);
};

export default PredictionTab;
