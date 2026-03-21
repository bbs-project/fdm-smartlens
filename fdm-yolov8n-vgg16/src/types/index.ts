// Type definitions for FDM SmartLens

import * as tf from '@tensorflow/tfjs';

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionResult {
  numDetections: number;
  boxes: DetectionBox[];
  scores: number[];
  classes: number[];
}

export interface VggOutput {
  code: number;
  name: string;
  confidence: number;
}

export interface ModelConfig {
  threshold: number;
}

export interface LoadingState {
  loading: boolean;
  progress: number;
  error: string | null;
}

export interface CameraViewState {
  ctx: Expo2DContext | null;
  vggOutputs: VggOutput[];
  isDetecting: boolean;
}

export interface ColorPalette {
  palette: string[];
  n: number;
  get: (i: number) => string;
}

export interface PreprocessResult {
  input: tf.Tensor;
  xRatio: number;
  yRatio: number;
}

declare module 'expo-2d-context' {
  export default class Expo2DContext {
    constructor(gl: any);
    initializeText(): Promise<void>;
    flush(): void;
    clearRect(x: number, y: number, width: number, height: number): void;
    fillRect(x: number, y: number, width: number, height: number): void;
    strokeRect(x: number, y: number, width: number, height: number): void;
    fillText(text: string, x: number, y: number, maxWidth?: number): void;
    measureText(text: string): { width: number };
    font: string;
    textBaseline: string;
    fillStyle: string;
    strokeStyle: string;
    lineWidth: number;
    width: number;
    height: number;
  }
}
