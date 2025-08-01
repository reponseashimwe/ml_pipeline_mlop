'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import toast from 'react-hot-toast';
import { Upload, Image as ImageIcon, Loader2 } from 'lucide-react';

interface ImageUploadProps {
	onPrediction?: (result: any) => void;
	isTrainingData?: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onPrediction, isTrainingData = false }) => {
	const [isLoading, setIsLoading] = useState(false);
	const [preview, setPreview] = useState<string | null>(null);

	const onDrop = useCallback(
		async (acceptedFiles: File[]) => {
			if (acceptedFiles.length === 0) return;

			const file = acceptedFiles[0];

			// Create preview
			const reader = new FileReader();
			reader.onload = () => {
				setPreview(reader.result as string);
			};
			reader.readAsDataURL(file);

			if (isTrainingData) {
				// Handle training data upload
				await uploadTrainingData(file);
			} else {
				// Handle prediction
				await predictImage(file);
			}
		},
		[isTrainingData, onPrediction]
	);

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: {
			'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp'],
		},
		multiple: false,
	});

	const predictImage = async (file: File) => {
		setIsLoading(true);
		try {
			const formData = new FormData();
			formData.append('image', file);

			const response = await axios.post('http://localhost:8000/predict/image', formData, {
				headers: {
					'Content-Type': 'multipart/form-data',
				},
			});

			if (onPrediction) {
				onPrediction(response.data);
			}

			toast.success('Prediction completed successfully!');
		} catch (error) {
			console.error('Prediction error:', error);
			toast.error('Failed to get prediction. Please try again.');
		} finally {
			setIsLoading(false);
		}
	};

	const uploadTrainingData = async (file: File) => {
		setIsLoading(true);
		try {
			const formData = new FormData();
			formData.append('images', file);

			const response = await axios.post('http://localhost:8000/upload/data', formData, {
				headers: {
					'Content-Type': 'multipart/form-data',
				},
			});

			toast.success('Training data uploaded successfully!');
		} catch (error) {
			console.error('Upload error:', error);
			toast.error('Failed to upload training data. Please try again.');
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className='space-y-4'>
			<div
				{...getRootProps()}
				className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
					isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
				}`}
			>
				<input {...getInputProps()} />

				{isLoading ? (
					<div className='flex flex-col items-center space-y-4'>
						<Loader2 className='w-12 h-12 text-blue-500 animate-spin' />
						<p className='text-gray-600'>
							{isTrainingData ? 'Uploading training data...' : 'Processing image...'}
						</p>
					</div>
				) : (
					<div className='flex flex-col items-center space-y-4'>
						{preview ? (
							<div className='relative'>
								<img src={preview} alt='Preview' className='w-32 h-32 object-cover rounded-lg border' />
								<button
									onClick={(e) => {
										e.stopPropagation();
										setPreview(null);
									}}
									className='absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm hover:bg-red-600'
								>
									×
								</button>
							</div>
						) : (
							<div className='flex flex-col items-center space-y-2'>
								<Upload className='w-12 h-12 text-gray-400' />
								<ImageIcon className='w-8 h-8 text-gray-400' />
							</div>
						)}

						<div>
							<p className='text-lg font-medium text-gray-900'>
								{isTrainingData ? 'Upload Training Image' : 'Upload Image for Prediction'}
							</p>
							<p className='text-sm text-gray-500 mt-1'>
								{isDragActive
									? 'Drop the image here...'
									: 'Drag & drop an image here, or click to select'}
							</p>
							<p className='text-xs text-gray-400 mt-2'>Supports: JPG, PNG, GIF, BMP</p>
						</div>
					</div>
				)}
			</div>

			{preview && !isLoading && (
				<div className='text-center'>
					<p className='text-sm text-gray-600'>
						{isTrainingData ? 'Image ready for training data upload' : 'Image ready for prediction'}
					</p>
				</div>
			)}
		</div>
	);
};

export default ImageUpload;
