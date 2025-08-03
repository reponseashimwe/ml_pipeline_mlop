'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import api from '../lib/api';

type ImageLabel = 'malnourished' | 'overnourished' | 'unlabeled';

interface PendingImage {
	id: string;
	file: File;
	preview: string;
	label: ImageLabel;
	name: string;
	size: number;
}

interface ExistingImage {
	id: string;
	name: string;
	label: ImageLabel;
	url: string;
	uploaded_at: string;
}

const UploadData: React.FC = () => {
	const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
	const [existingImages, setExistingImages] = useState<ExistingImage[]>([]);
	const [showHelpModal, setShowHelpModal] = useState(false);

	// Load existing uploaded images on component mount
	useEffect(() => {
		loadExistingUploadedImages();
	}, []);

	const loadExistingUploadedImages = async () => {
		try {
			console.log('🔄 Loading existing uploaded images...');
			const response = await api.getUploadedImages();
			console.log('📡 API Response:', response);

			if (response.success && response.images.length > 0) {
				// The backend now returns the correct structure
				const formattedImages: ExistingImage[] = response.images.map((img: any, index: number) => ({
					id: `existing-${index}`,
					name: img.name || `Image ${index + 1}`,
					label: img.label || 'unlabeled',
					url: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'}${img.url}`,
					uploaded_at: img.uploaded_at || new Date().toISOString(),
				}));

				console.log('📋 Formatted images:', formattedImages);
				setExistingImages(formattedImages);
				toast.success(`Loaded ${response.total_count} existing uploaded images`);
			} else {
				console.log('⚠️ No images found or API returned no success');
				setExistingImages([]);
			}
		} catch (error) {
			console.error('❌ Failed to load existing images:', error);
			toast.error('Failed to load existing images');
		}
	};

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

		// Add to existing images, don't replace
		setPendingImages((prev) => [...prev, ...newImages]);
		toast.success(`Added ${acceptedFiles.length} new images for labeling`);
	}, []);

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: {
			'image/*': ['.jpeg', '.jpg', '.png'],
		},
		multiple: true,
		maxSize: 10 * 1024 * 1024, // 10MB limit
	});

	// Label an image
	const labelImage = (imageId: string, label: ImageLabel) => {
		setPendingImages((prev) => prev.map((img) => (img.id === imageId ? { ...img, label } : img)));
		toast.success(`Image labeled as ${label}`, { duration: 1000 });
	};

	// Auto-label similar images
	const autoLabelSimilar = (imageId: string, label: ImageLabel) => {
		const sourceImage = pendingImages.find((img) => img.id === imageId);
		if (!sourceImage) return;

		let labeledCount = 0;
		setPendingImages((prev) =>
			prev.map((img) => {
				// Simple heuristic: similar file size might indicate similar images
				const sizeDiff = Math.abs(img.size - sourceImage.size) / sourceImage.size;
				if (img.id !== imageId && img.label === 'unlabeled' && sizeDiff < 0.3) {
					labeledCount++;
					return { ...img, label };
				}
				return img;
			})
		);

		if (labeledCount > 0) {
			toast.success(`Auto-labeled ${labeledCount} similar images as ${label}`);
		}
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

	// Clear all images
	const clearAllImages = () => {
		pendingImages.forEach((img) => URL.revokeObjectURL(img.preview));
		setPendingImages([]);
		toast.success('Cleared all images');
	};

	// Bulk labeling functions
	const labelAllAs = (label: ImageLabel) => {
		const unlabeledCount = pendingImages.filter((img) => img.label === 'unlabeled').length;
		setPendingImages((prev) => prev.map((img) => (img.label === 'unlabeled' ? { ...img, label } : img)));
		if (unlabeledCount > 0) {
			toast.success(`Labeled ${unlabeledCount} images as ${label}`);
		}
	};

	// Upload labeled images
	const uploadLabeledImages = async () => {
		const labeledImages = pendingImages.filter((img) => img.label !== 'unlabeled');

		if (labeledImages.length === 0) {
			toast.error('Please label at least one image before uploading');
			return;
		}

		try {
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

				// Wait a moment for files to be saved, then refresh the existing images list
				setTimeout(async () => {
					await loadExistingUploadedImages();
				}, 1000);

				toast.success(`Successfully uploaded ${labeledImages.length} labeled images`);
			} else {
				throw new Error(response.message);
			}
		} catch (error) {
			toast.error('Failed to upload labeled images');
		}
	};

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			pendingImages.forEach((img) => URL.revokeObjectURL(img.preview));
		};
	}, [pendingImages]);

	const labeledCount = pendingImages.filter((img) => img.label !== 'unlabeled').length;
	const unlabeledCount = pendingImages.filter((img) => img.label === 'unlabeled').length;
	const malnourishedCount = pendingImages.filter((img) => img.label === 'malnourished').length;
	const overnourishedCount = pendingImages.filter((img) => img.label === 'overnourished').length;

	return (
		<div className='space-y-6'>
			{/* Help Modal */}
			{showHelpModal && (
				<div
					className='fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50'
					onClick={(e) => {
						if (e.target === e.currentTarget) {
							setShowHelpModal(false);
						}
					}}
				>
					<div className='bg-white p-6 rounded-lg max-w-2xl max-h-96 overflow-y-auto'>
						<div className='flex justify-between items-center mb-4'>
							<h3 className='text-lg font-bold'>🏥 Image Classification Guide</h3>
							<button
								onClick={() => setShowHelpModal(false)}
								className='text-gray-500 hover:text-gray-700 text-xl font-bold'
							>
								×
							</button>
						</div>
						<div className='space-y-4 text-sm'>
							<div>
								<h4 className='font-medium text-red-600'>🔴 Malnourished Children</h4>
								<ul className='list-disc ml-4 space-y-1'>
									<li>Visible ribs and bones</li>
									<li>Sunken cheeks and eyes</li>
									<li>Thin arms and legs</li>
									<li>Protruding belly (kwashiorkor)</li>
									<li>Very low body weight for age</li>
								</ul>
							</div>
							<div>
								<h4 className='font-medium text-orange-600'>🟠 Overnourished Children</h4>
								<ul className='list-disc ml-4 space-y-1'>
									<li>Excess weight for age/height</li>
									<li>Round face and body</li>
									<li>Thick arms and legs</li>
									<li>Signs of childhood obesity</li>
									<li>Well-fed appearance</li>
								</ul>
							</div>
							<div className='bg-blue-50 p-3 rounded'>
								<p className='text-blue-800'>
									<strong>Tip:</strong> When in doubt, look at the overall body composition and
									compare to age-appropriate norms.
								</p>
							</div>
						</div>
						<button
							onClick={() => setShowHelpModal(false)}
							className='mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700'
						>
							Got it!
						</button>
					</div>
				</div>
			)}

			{/* Image Upload and Labeling Section */}
			<div className='bg-white p-6 rounded-lg shadow-md'>
				<div className='flex justify-between items-center mb-4'>
					<h3 className='text-lg font-semibold'>📚 Upload & Label Training Images</h3>
					<button
						onClick={() => setShowHelpModal(true)}
						className='px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200'
					>
						🏥 Classification Guide
					</button>
				</div>
				<p className='text-gray-600 mb-4 text-sm'>
					Upload images and label them for model retraining. Quality labeling is crucial for model
					performance.
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
						<div className='text-2xl mb-2'>📸</div>
						<p className='text-gray-600'>
							{isDragActive ? 'Drop images here...' : 'Drag & drop images here, or click to select'}
						</p>
						<p className='text-xs text-gray-500'>
							Supports: JPG, JPEG, PNG (Max 10MB each, Multiple files)
						</p>
						<p className='text-xs text-blue-500 font-medium'>
							💡 Each new selection adds to your existing images
						</p>
					</div>
				</div>

				{/* Statistics */}
				{pendingImages.length > 0 && (
					<div className='mt-4 p-4 bg-gray-50 rounded-lg'>
						<div className='grid grid-cols-2 md:grid-cols-4 gap-4 text-center'>
							<div>
								<div className='text-2xl font-bold text-gray-700'>{pendingImages.length}</div>
								<div className='text-xs text-gray-600'>Total Images</div>
							</div>
							<div>
								<div className='text-2xl font-bold text-red-600'>{malnourishedCount}</div>
								<div className='text-xs text-gray-600'>Malnourished</div>
							</div>
							<div>
								<div className='text-2xl font-bold text-orange-600'>{overnourishedCount}</div>
								<div className='text-xs text-gray-600'>Overnourished</div>
							</div>
							<div>
								<div className='text-2xl font-bold text-gray-500'>{unlabeledCount}</div>
								<div className='text-xs text-gray-600'>Unlabeled</div>
							</div>
						</div>
					</div>
				)}

				{/* Pending Images */}
				{pendingImages.length > 0 && (
					<div className='mt-6'>
						<div className='flex justify-between items-center mb-4'>
							<h4 className='font-semibold'>📋 Images for Labeling ({pendingImages.length})</h4>
							<div className='flex items-center space-x-2'>
								{unlabeledCount > 0 && (
									<>
										<button
											onClick={() => labelAllAs('malnourished')}
											className='px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200'
										>
											All → Malnourished
										</button>
										<button
											onClick={() => labelAllAs('overnourished')}
											className='px-2 py-1 text-xs bg-orange-100 text-orange-700 rounded hover:bg-orange-200'
										>
											All → Overnourished
										</button>
									</>
								)}
								<button
									onClick={clearAllImages}
									className='px-3 py-1 text-xs bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors'
								>
									🗑️ Clear All
								</button>
							</div>
						</div>

						<div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 max-h-96 overflow-y-auto'>
							{pendingImages.map((image) => (
								<div
									key={image.id}
									className={`border rounded-lg p-3 transition-all ${
										image.label === 'unlabeled'
											? 'bg-gray-50 border-gray-200'
											: image.label === 'malnourished'
											? 'bg-red-50 border-red-200'
											: 'bg-orange-50 border-orange-200'
									}`}
								>
									<div className='relative mb-3'>
										<img
											src={image.preview}
											alt={image.name}
											className='w-full h-32 object-cover rounded border'
										/>
										<button
											onClick={() => removeImage(image.id)}
											className='absolute top-1 right-1 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600 shadow-lg'
										>
											×
										</button>
										{image.label !== 'unlabeled' && (
											<div
												className={`absolute top-1 left-1 px-2 py-1 rounded text-xs font-bold text-white shadow-lg ${
													image.label === 'malnourished' ? 'bg-red-500' : 'bg-orange-500'
												}`}
											>
												{image.label === 'malnourished' ? '🔴' : '🟠'}
											</div>
										)}
									</div>

									<div className='space-y-2'>
										<p className='text-xs text-gray-600 truncate font-medium'>{image.name}</p>
										<p className='text-xs text-gray-500'>{(image.size / 1024).toFixed(1)} KB</p>

										{/* Labeling Buttons */}
										<div className='flex space-x-1'>
											<button
												onClick={() => labelImage(image.id, 'malnourished')}
												onDoubleClick={() => autoLabelSimilar(image.id, 'malnourished')}
												className={`flex-1 px-2 py-1 text-xs rounded font-medium transition-colors ${
													image.label === 'malnourished'
														? 'bg-red-500 text-white shadow-md'
														: 'bg-red-100 text-red-700 hover:bg-red-200'
												}`}
												title='Double-click to auto-label similar images'
											>
												🔴 Mal
											</button>
											<button
												onClick={() => labelImage(image.id, 'overnourished')}
												onDoubleClick={() => autoLabelSimilar(image.id, 'overnourished')}
												className={`flex-1 px-2 py-1 text-xs rounded font-medium transition-colors ${
													image.label === 'overnourished'
														? 'bg-orange-500 text-white shadow-md'
														: 'bg-orange-100 text-orange-700 hover:bg-orange-200'
												}`}
												title='Double-click to auto-label similar images'
											>
												🟠 Over
											</button>
										</div>

										{image.label !== 'unlabeled' && (
											<div
												className={`text-xs font-bold p-1 rounded text-center ${
													image.label === 'malnourished'
														? 'text-red-700 bg-red-100'
														: 'text-orange-700 bg-orange-100'
												}`}
											>
												✅ {image.label}
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
									className='px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium'
								>
									📤 Upload {labeledCount} Labeled Images
								</button>
							</div>
						)}
					</div>
				)}
			</div>

			{/* Existing Uploaded Images Section */}
			{existingImages.length > 0 && (
				<div className='bg-white p-6 rounded-lg shadow-md'>
					<div className='flex justify-between items-center mb-4'>
						<h3 className='text-lg font-semibold'>
							📁 Previously Uploaded Images ({existingImages.length})
						</h3>
						<button
							onClick={loadExistingUploadedImages}
							className='px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200'
						>
							🔄 Refresh
						</button>
					</div>
					<p className='text-gray-600 mb-4 text-sm'>
						These images have been uploaded and are ready for model retraining.
					</p>

					<div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 overflow-y-auto'>
						{existingImages.map((image) => (
							<div
								key={image.id}
								className={`border rounded-lg p-3 transition-all ${
									image.label === 'unlabeled'
										? 'bg-gray-50 border-gray-200'
										: image.label === 'malnourished'
										? 'bg-red-50 border-red-200'
										: 'bg-orange-50 border-orange-200'
								}`}
							>
								<div className='relative mb-3'>
									<img
										src={image.url}
										alt={image.name}
										className='w-full h-32 object-cover rounded border'
										onError={(e) => {
											// Fallback for broken images
											e.currentTarget.src =
												'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTI4IiBoZWlnaHQ9IjEyOCIgdmlld0JveD0iMCAwIDEyOCAxMjgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjgiIGhlaWdodD0iMTI4IiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik02NCAzMkM0Ny40MyAzMiAzNCA0NS40MyAzNCA2NEMzNCA4Mi41NyA0Ny40MyA5NiA2NCA5NkM4MC41NyA5NiA5NCA4Mi41NyA5NCA2NEM5NCA0NS40MyA4MC41NyAzMiA2NCAzMloiIGZpbGw9IiNEM0Q0RDYiLz4KPHBhdGggZD0iTTY0IDQwQzU2LjI3IDQwIDUwIDQ2LjI3IDUwIDU0QzUwIDYxLjczIDU2LjI3IDY4IDY0IDY4QzcxLjczIDY4IDc4IDYxLjczIDc4IDU0Qzc4IDQ2LjI3IDcxLjczIDQwIDY0IDQwWiIgZmlsbD0iI0QzRDRENiIvPgo8L3N2Zz4K';
										}}
									/>
									{image.label !== 'unlabeled' && (
										<div
											className={`absolute top-1 left-1 px-2 py-1 rounded text-xs font-bold text-white shadow-lg ${
												image.label === 'malnourished' ? 'bg-red-500' : 'bg-orange-500'
											}`}
										>
											{image.label === 'malnourished' ? '🔴' : '🟠'}
										</div>
									)}
								</div>

								<div className='space-y-2'>
									<p className='text-xs text-gray-600 truncate font-medium'>{image.name}</p>
									<p className='text-xs text-gray-500'>
										{new Date(image.uploaded_at).toLocaleDateString()}
									</p>

									{image.label !== 'unlabeled' && (
										<div
											className={`text-xs font-bold p-1 rounded text-center ${
												image.label === 'malnourished'
													? 'text-red-700 bg-red-100'
													: 'text-orange-700 bg-orange-100'
											}`}
										>
											✅ {image.label}
										</div>
									)}
								</div>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
};

export default UploadData;
