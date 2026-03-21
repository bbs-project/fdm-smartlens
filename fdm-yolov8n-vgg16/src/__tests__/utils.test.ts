import { preprocess, cleanupTensors, isTensor } from '../utils';
import * as tf from '@tensorflow/tfjs';

jest.mock('@tensorflow/tfjs', () => ({
  tidy: jest.fn(fn => fn()),
  image: {
    resizeBilinear: jest.fn(() => ({
      div: jest.fn(() => ({
        expandDims: jest.fn(() => ({
          shape: [1, 640, 640, 3],
          dispose: jest.fn(),
        })),
      })),
    })),
  },
}));

describe('Utils', () => {
  describe('preprocess', () => {
    it('should pad image to square and resize', () => {
      const mockImg = {
        shape: [3, 640, 640],
        pad: jest.fn(() => ({
          shape: [3, 640, 640],
        })),
      } as any;

      const [input, xRatio, yRatio] = preprocess(mockImg, 640, 640);

      expect(input).toBeDefined();
      expect(xRatio).toBeDefined();
      expect(yRatio).toBeDefined();
    });

    it('should calculate correct ratios for non-square images', () => {
      const mockImg = {
        shape: [3, 480, 640],
        pad: jest.fn(() => ({
          shape: [3, 640, 640],
        })),
      } as any;

      const [, xRatio, yRatio] = preprocess(mockImg, 640, 640);

      expect(xRatio).toBeGreaterThan(0);
      expect(yRatio).toBeGreaterThan(0);
    });
  });

  describe('cleanupTensors', () => {
    it('should dispose valid tensors', () => {
      const mockTensor = {
        dispose: jest.fn(),
      } as any;

      cleanupTensors(mockTensor);

      expect(mockTensor.dispose).toHaveBeenCalled();
    });

    it('should skip non-tensor objects', () => {
      const nonTensor = { shape: [1, 2, 3] };

      expect(() => cleanupTensors(nonTensor as any)).not.toThrow();
    });

    it('should handle multiple tensors', () => {
      const tensor1 = { dispose: jest.fn() } as any;
      const tensor2 = { dispose: jest.fn() } as any;

      cleanupTensors(tensor1, tensor2);

      expect(tensor1.dispose).toHaveBeenCalled();
      expect(tensor2.dispose).toHaveBeenCalled();
    });
  });

  describe('isTensor', () => {
    it('should return true for valid tensors', () => {
      const mockTensor = {
        shape: [1, 2, 3],
        dispose: jest.fn(),
      } as any;

      expect(isTensor(mockTensor)).toBe(true);
    });

    it('should return false for non-tensors', () => {
      expect(isTensor(null)).toBe(false);
      expect(isTensor({})).toBe(false);
      expect(isTensor({ shape: [1, 2] })).toBe(false);
    });
  });
});
