import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react-native';
import App from '../App';
import { Camera } from 'expo-camera/legacy';
import * as tf from '@tensorflow/tfjs';

// Mock expo-camera
jest.mock('expo-camera/legacy', () => ({
  Camera: {
    requestCameraPermissionsAsync: jest.fn(),
  },
}));

// Mock TensorFlow
jest.mock('@tensorflow/tfjs', () => ({
  ready: jest.fn(() => Promise.resolve()),
  loadGraphModel: jest.fn(),
  ones: jest.fn(() => ({
    shape: [1, 3, 640, 640],
    dispose: jest.fn(),
  })),
  disposeVariables: jest.fn(),
}));

// Mock tfjs-react-native
jest.mock('@tensorflow/tfjs-react-native', () => ({}));

describe('App', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows loading state initially', () => {
    (Camera.requestCameraPermissionsAsync as jest.Mock).mockResolvedValue({ status: 'granted' });
    (tf.ready as jest.Mock).mockResolvedValue(undefined);
    (tf.loadGraphModel as jest.Mock).mockResolvedValue({
      inputs: [{ shape: [1, 3, 640, 640] }],
      executeAsync: jest.fn(),
      dispose: jest.fn(),
    });

    render(<App />);

    expect(screen.getByText(/Loading models.../)).toBeTruthy();
  });

  it('requests camera permission on mount', async () => {
    (Camera.requestCameraPermissionsAsync as jest.Mock).mockResolvedValue({ status: 'granted' });
    (tf.ready as jest.Mock).mockResolvedValue(undefined);
    (tf.loadGraphModel as jest.Mock).mockResolvedValue({
      inputs: [{ shape: [1, 3, 640, 640] }],
      executeAsync: jest.fn(),
      dispose: jest.fn(),
    });

    render(<App />);

    await waitFor(() => {
      expect(Camera.requestCameraPermissionsAsync).toHaveBeenCalledTimes(1);
    });
  });

  it('shows permission denied message when permission is rejected', async () => {
    (Camera.requestCameraPermissionsAsync as jest.Mock).mockResolvedValue({ status: 'denied' });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Camera permission not granted!')).toBeTruthy();
    });
  });

  it('displays error message when model loading fails', async () => {
    (Camera.requestCameraPermissionsAsync as jest.Mock).mockResolvedValue({ status: 'granted' });
    (tf.ready as jest.Mock).mockResolvedValue(undefined);
    (tf.loadGraphModel as jest.Mock).mockRejectedValue(new Error('Model load failed'));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Error:/)).toBeTruthy();
    });
  });
});
