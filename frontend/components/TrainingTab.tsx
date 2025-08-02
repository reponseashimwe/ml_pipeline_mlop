'use client';

import React from 'react';
import RetrainingPanel from './RetrainingPanel';

const TrainingTab: React.FC = () => {
	return (
		<div className='space-y-6'>
			<div className='bg-blue-50 p-4 rounded-lg'>
				<h2 className='text-xl font-semibold text-blue-900 mb-2'>🔄 Model Retraining</h2>
				<p className='text-blue-800'>
					Upload labeled images and retrain the model to improve its performance. The retraining process
					includes real-time progress tracking and comprehensive logging.
				</p>
			</div>

			<RetrainingPanel />
		</div>
	);
};

export default TrainingTab;
