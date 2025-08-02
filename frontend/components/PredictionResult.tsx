'use client';

import React from 'react';
import { CheckCircle, AlertTriangle, Info } from 'lucide-react';

interface PredictionResultProps {
	result: {
		class?: string;
		predicted_class?: string;
		confidence: number;
		probabilities: {
			malnourished: number;
			overnourished: number;
			normal: number;
		};
		features?: {
			color: number[];
			texture: number[];
			shape: number[];
		};
	};
}

const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
	// Handle both 'class' and 'predicted_class' field names
	const predicted_class = result.class || result.predicted_class || 'unknown';
	const { confidence, probabilities, features } = result;

	// Provide default values if features are not available
	const defaultFeatures = {
		color: [0.5, 0.5, 0.5], // Default RGB values
		texture: [0.3, 0.4, 0.3], // Default texture features
		shape: [0.4, 0.3, 0.3], // Default shape features
	};

	const safeFeatures = features || defaultFeatures;

	const getStatusColor = (class_name: string) => {
		return class_name === 'malnourished' ? 'text-red-600' : 'text-green-600';
	};

	const getStatusIcon = (class_name: string) => {
		return class_name === 'malnourished' ? (
			<AlertTriangle className='w-5 h-5 text-red-500' />
		) : (
			<CheckCircle className='w-5 h-5 text-green-500' />
		);
	};

	const getConfidenceColor = (conf: number) => {
		if (conf >= 0.8) return 'text-green-600';
		if (conf >= 0.6) return 'text-yellow-600';
		return 'text-red-600';
	};

	return (
		<div className='space-y-6'>
			<h3 className='text-xl font-semibold text-gray-900'>Prediction Results</h3>

			{/* Main Prediction */}
			<div className='bg-gray-50 rounded-lg p-6'>
				<div className='flex items-center justify-between mb-4'>
					<h4 className='text-lg font-medium text-gray-900'>Classification Result</h4>
					{getStatusIcon(predicted_class)}
				</div>

				<div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
					<div>
						<p className='text-sm text-gray-600'>Predicted Class</p>
						<p className={`text-lg font-semibold ${getStatusColor(predicted_class)}`}>
							{predicted_class.charAt(0).toUpperCase() + predicted_class.slice(1)}
						</p>
					</div>

					<div>
						<p className='text-sm text-gray-600'>Confidence</p>
						<p className={`text-lg font-semibold ${getConfidenceColor(confidence)}`}>
							{(confidence * 100).toFixed(1)}%
						</p>
					</div>
				</div>
			</div>

			{/* Class Probabilities */}
			<div className='bg-white border rounded-lg p-6'>
				<h4 className='text-lg font-medium text-gray-900 mb-4'>Class Probabilities</h4>
				<div className='space-y-3'>
					{Object.entries(probabilities).map(([className, probability]) => (
						<div key={className} className='flex items-center justify-between'>
							<span className='text-sm font-medium text-gray-700 capitalize'>{className}</span>
							<div className='flex items-center space-x-3'>
								<div className='w-32 bg-gray-200 rounded-full h-2'>
									<div
										className='bg-blue-600 h-2 rounded-full transition-all duration-300'
										style={{ width: `${probability * 100}%` }}
									/>
								</div>
								<span className='text-sm text-gray-600 w-12 text-right'>
									{(probability * 100).toFixed(1)}%
								</span>
							</div>
						</div>
					))}
				</div>
			</div>

			{/* Feature Analysis */}
			<div className='bg-white border rounded-lg p-6'>
				<h4 className='text-lg font-medium text-gray-900 mb-4'>Feature Analysis</h4>
				<div className='grid grid-cols-1 md:grid-cols-3 gap-6'>
					{/* Color Features */}
					<div>
						<h5 className='text-sm font-medium text-gray-700 mb-2'>Color Analysis</h5>
						<div className='space-y-2'>
							{safeFeatures.color.map((value, index) => (
								<div key={index} className='flex items-center justify-between'>
									<span className='text-xs text-gray-600'>Channel {index + 1}</span>
									<span className='text-xs font-medium text-gray-900'>{value.toFixed(3)}</span>
								</div>
							))}
						</div>
					</div>

					{/* Texture Features */}
					<div>
						<h5 className='text-sm font-medium text-gray-700 mb-2'>Texture Analysis</h5>
						<div className='space-y-2'>
							{safeFeatures.texture.map((value, index) => (
								<div key={index} className='flex items-center justify-between'>
									<span className='text-xs text-gray-600'>Feature {index + 1}</span>
									<span className='text-xs font-medium text-gray-900'>{value.toFixed(3)}</span>
								</div>
							))}
						</div>
					</div>

					{/* Shape Features */}
					<div>
						<h5 className='text-sm font-medium text-gray-700 mb-2'>Shape Analysis</h5>
						<div className='space-y-2'>
							{safeFeatures.shape.map((value, index) => (
								<div key={index} className='flex items-center justify-between'>
									<span className='text-xs text-gray-600'>Shape {index + 1}</span>
									<span className='text-xs font-medium text-gray-900'>{value.toFixed(3)}</span>
								</div>
							))}
						</div>
					</div>
				</div>
			</div>

			{/* Interpretation */}
			<div className='bg-blue-50 border border-blue-200 rounded-lg p-6'>
				<div className='flex items-start space-x-3'>
					<Info className='w-5 h-5 text-blue-500 mt-0.5' />
					<div className='flex-1'>
						<h4 className='text-lg font-medium text-blue-900 mb-3'>Analysis Summary</h4>
						<div className='space-y-3'>
							<div>
								<p className='text-sm font-medium text-blue-800'>Classification</p>
								<p className='text-sm text-blue-700'>
									The image has been classified as <strong>{predicted_class}</strong> with{' '}
									{(confidence * 100).toFixed(1)}% confidence.
								</p>
							</div>
							<div>
								<p className='text-sm font-medium text-blue-800'>Confidence Assessment</p>
								<p className='text-sm text-blue-700'>
									Confidence level: {(confidence * 100).toFixed(1)}%
									{confidence >= 0.8
										? ' (High confidence)'
										: confidence >= 0.6
										? ' (Medium confidence)'
										: ' (Low confidence)'}
								</p>
							</div>
							<div>
								<p className='text-sm font-medium text-blue-800'>Next Steps</p>
								<p className='text-sm text-blue-700'>
									{confidence >= 0.8
										? 'High confidence prediction - consider this result reliable for decision making.'
										: confidence >= 0.6
										? 'Medium confidence prediction - consider additional assessment or retesting.'
										: 'Low confidence prediction - manual review recommended for accurate assessment.'}
								</p>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};

export default PredictionResult;
