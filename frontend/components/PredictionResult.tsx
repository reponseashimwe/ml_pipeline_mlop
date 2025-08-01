'use client';

import React from 'react';
import { CheckCircle, AlertTriangle, Info } from 'lucide-react';

interface PredictionResultProps {
	result: {
		predicted_class: string;
		confidence: number;
		probabilities: {
			[key: string]: number;
		};
		features: {
			color: number[];
			texture: number[];
			shape: number[];
		};
		interpretation: {
			risk_level: string;
			recommendation: string;
			confidence_interpretation: string;
		};
	};
}

const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
	const { predicted_class, confidence, probabilities, features, interpretation } = result;

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
							{features.color.map((value, index) => (
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
							{features.texture.map((value, index) => (
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
							{features.shape.map((value, index) => (
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
						<h4 className='text-lg font-medium text-blue-900 mb-3'>Interpretation</h4>
						<div className='space-y-3'>
							<div>
								<p className='text-sm font-medium text-blue-800'>Risk Level</p>
								<p className='text-sm text-blue-700'>{interpretation.risk_level}</p>
							</div>
							<div>
								<p className='text-sm font-medium text-blue-800'>Recommendation</p>
								<p className='text-sm text-blue-700'>{interpretation.recommendation}</p>
							</div>
							<div>
								<p className='text-sm font-medium text-blue-800'>Confidence Assessment</p>
								<p className='text-sm text-blue-700'>{interpretation.confidence_interpretation}</p>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};

export default PredictionResult;
