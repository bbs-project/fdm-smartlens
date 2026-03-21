import * as tf from '@tensorflow/tfjs';

/**
 * 성능 측정 결과
 */
export interface PerformanceMetrics {
  inferenceTime: number;
  preprocessingTime: number;
  postprocessingTime: number;
  fps: number;
  memoryUsage?: number;
}

/**
 * 성능 모니터링 클래스
 */
export class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Map<string, number[]> = new Map();
  private frameCount: number = 0;
  private lastFpsUpdate: number = 0;
  private currentFps: number = 0;

  private constructor() {}

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  /**
   * 시간 측정 시작
   */
  startTimer(label: string): void {
    // @ts-ignore - performance.now() is available in React Native
    const startTime = performance.now();
    this.metrics.set(`start_${label}`, [startTime]);
  }

  /**
   * 시간 측정 종료 및 반환 (ms)
   */
  endTimer(label: string): number {
    // @ts-ignore - performance.now() is available in React Native
    const endTime = performance.now();
    const startTime = this.metrics.get(`start_${label}`)?.[0] || endTime;
    const duration = endTime - startTime;

    // Record metric
    if (!this.metrics.has(label)) {
      this.metrics.set(label, []);
    }
    const history = this.metrics.get(label)!;
    history.push(duration);

    // Keep only last 30 measurements
    if (history.length > 30) {
      history.shift();
    }

    return duration;
  }

  /**
   * 평균 추론 시간 조회 (ms)
   */
  getAverageTime(label: string): number {
    const history = this.metrics.get(label);
    if (!history || history.length === 0) return 0;
    return history.reduce((a, b) => a + b, 0) / history.length;
  }

  /**
   * FPS 업데이트
   */
  updateFps(): number {
    const now = Date.now();
    this.frameCount++;

    if (now - this.lastFpsUpdate >= 1000) {
      this.currentFps = this.frameCount;
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    return this.currentFps;
  }

  /**
   * 현재 FPS 조회
   */
  getFps(): number {
    return this.currentFps;
  }

  /**
   * 메모리 사용량 조회 (MB)
   */
  getMemoryUsage(): number | undefined {
    // @ts-ignore - React Native specific
    if (global.performance && global.performance.memory) {
      // @ts-ignore
      return global.performance.memory.usedJSHeapSize / (1024 * 1024);
    }
    return undefined;
  }

  /**
   * 성능 메트릭스 조회
   */
  getMetrics(label: string): PerformanceMetrics {
    return {
      inferenceTime: this.getAverageTime(label),
      preprocessingTime: this.getAverageTime(`${label}_preprocess`),
      postprocessingTime: this.getAverageTime(`${label}_postprocess`),
      fps: this.getFps(),
      memoryUsage: this.getMemoryUsage(),
    };
  }

  /**
   * 모든 메트릭스 로그 출력
   */
  logMetrics(): void {
    console.log('=== Performance Metrics ===');
    this.metrics.forEach((values, key) => {
      if (!key.startsWith('start_')) {
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        console.log(`${key}: ${avg.toFixed(2)}ms (min: ${min.toFixed(2)}ms, max: ${max.toFixed(2)}ms)`);
      }
    });
    console.log(`FPS: ${this.getFps()}`);
    console.log(`Memory: ${this.getMemoryUsage()?.toFixed(2) || 'N/A'} MB`);
    console.log('========================');
  }

  /**
   * 메트릭스 초기화
   */
  reset(): void {
    this.metrics.clear();
    this.frameCount = 0;
    this.lastFpsUpdate = 0;
    this.currentFps = 0;
  }
}

/**
 * 텐서 메모리 관리 유틸리티
 */
export class TensorMemoryManager {
  private static disposedTensors: WeakSet<tf.Tensor> = new WeakSet();

  /**
   * 텐서 자동 정리 (tidy wrapper)
   */
  static tidy<T>(fn: () => T): T {
    return tf.tidy(fn);
  }

  /**
   * 텐서 수동 정리
   */
  static dispose(...tensors: tf.Tensor[]): void {
    tensors.forEach(tensor => {
      if (!this.disposedTensors.has(tensor)) {
        tensor.dispose();
        this.disposedTensors.add(tensor);
      }
    });
  }

  /**
   * 텐서 배열 정리
   */
  static disposeArray(tensors: tf.Tensor[]): void {
    this.dispose(...tensors);
  }

  /**
   * 메모리 상태 조회
   */
  static getMemoryInfo(): tf.MemoryInfo {
    return tf.memory();
  }

  /**
   * 메모리 최적화 설정
   */
  static optimize(): void {
    // Dispose all tensors
    tf.disposeVariables();
    
    // Force garbage collection (if available)
    // @ts-ignore
    if (global.gc) {
      // @ts-ignore
      global.gc();
    }
  }
}

/**
 * GPU 가속 설정
 */
export class GpuAccelerator {
  /**
   * GPU 사용 가능 여부 확인
   */
  static async isGpuAvailable(): Promise<boolean> {
    try {
      const backend = tf.getBackend();
      return backend === 'webgl' || backend === 'rn-webgl';
    } catch {
      return false;
    }
  }

  /**
   * GPU 백엔드 설정
   */
  static async enableGpu(): Promise<boolean> {
    try {
      // React Native WebGL backend
      await tf.setBackend('rn-webgl');
      await tf.ready();
      console.log('[GPU] WebGL backend enabled');
      return true;
    } catch (error) {
      console.warn('[GPU] Failed to enable GPU, falling back to CPU');
      try {
        await tf.setBackend('cpu');
        await tf.ready();
        return false;
      } catch {
        return false;
      }
    }
  }

  /**
   * CPU 백엔드 설정
   */
  static async enableCpu(): Promise<void> {
    await tf.setBackend('cpu');
    await tf.ready();
    console.log('[CPU] CPU backend enabled');
  }

  /**
   * 현재 백엔드 조회
   */
  static getCurrentBackend(): string {
    return tf.getBackend();
  }

  /**
   * 백엔드 정보 조회
   */
  static getBackendInfo(): { backend: string; isGpu: boolean } {
    const backend = tf.getBackend();
    return {
      backend,
      isGpu: backend === 'webgl' || backend === 'rn-webgl',
    };
  }
}

export default {
  PerformanceMonitor,
  TensorMemoryManager,
  GpuAccelerator,
};
