import React from 'react';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
	title: 'ML Pipeline - Malnutrition Detection',
	description: 'End-to-end ML pipeline for malnutrition detection using image data',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang='en'>
			<body className={inter.className}>
				{children}
				<Toaster position='top-right' />
			</body>
		</html>
	);
}
