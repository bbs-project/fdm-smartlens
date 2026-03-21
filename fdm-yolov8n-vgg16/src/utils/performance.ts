import ReactNativePerformance from 'react-native-performance';
import { Platform } from 'react-native';

/**
 * Performance Monitoring Utility for FDM SmartLens
 * 
 * Features:
 * - App startup time monitoring
 * - Model loading performance tracking
 * - Inference latency measurement
 * - Memory usage monitoring
 */

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: Map<string, number[]> = new Map();
  private isMonitoring: boolean = false;

  private constructor() {
    this.initialize();
  }

  public static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  private initialize() {
    // Register app launch marker
    ReactNativePerformance.markAppStart();

    // Enable FPS monitoring on iOS
    if (Platform.OS === 'ios') {
      ReactNativePerformance.startFPSMonitor({
        sampleInterval: 1000,
        lowFPSWarningThreshold: 50,
        lowFPSSampleCount: 3,
      });
    }
  }

  /**
   * Start tracking a metric
   */
  public startMetric(name: string): void {
    ReactNativePerformance.mark(`${name}_start`);
  }

  /**
   * End tracking a metric and record the duration
   */
  public endMetric(name: string): number | null {
    const markName = `${name}_start`;
    const duration = ReactNativePerformance.measure(name, markName).duration;
    
    // Store metric for analysis
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);

    // Log slow operations
    if (duration > 1000) {
      console.warn(`[Performance] Slow operation detected: ${name} took ${duration.toFixed(2)}ms`);
    }

    return duration;
  }

  /**
   * Track model loading performance
   */
  public trackModelLoading(modelName: string, duration: number): void {
    console.log(`[Performance] ${modelName} loaded in ${duration.toFixed(2)}ms`);
    
    // Send to analytics service if configured
    this.sendToAnalytics('model_load', {
      model: modelName,
      duration,
      timestamp: Date.now(),
    });
  }

  /**
   * Track inference latency
   */
  public trackInference(modelName: string, latency: number, imageSize: string): void {
    const key = `inference_${modelName}`;
    
    if (!this.metrics.has(key)) {
      this.metrics.set(key, []);
    }
    this.metrics.get(key)!.push(latency);

    console.log(`[Performance] ${modelName} inference: ${latency.toFixed(2)}ms (${imageSize})`);
  }

  /**
   * Get average latency for a metric
   */
  public getAverageLatency(name: string): number {
    const values = this.metrics.get(name) || [];
    if (values.length === 0) return 0;
    
    const sum = values.reduce((a, b) => a + b, 0);
    return sum / values.length;
  }

  /**
   * Get performance report
   */
  public getReport(): Record<string, any> {
    const report: Record<string, any> = {};
    
    this.metrics.forEach((values, key) => {
      if (values.length > 0) {
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        report[key] = {
          count: values.length,
          avg: avg.toFixed(2),
          min: min.toFixed(2),
          max: max.toFixed(2),
          unit: 'ms',
        };
      }
    });

    return report;
  }

  /**
   * Start continuous monitoring
   */
  public startMonitoring(): void {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    
    // Monitor memory usage every 30 seconds
    setInterval(() => {
      this.checkMemoryUsage();
    }, 30000);
  }

  /**
   * Check memory usage
   */
  private checkMemoryUsage(): void {
    // @ts-ignore - performance.memory is not in types but available in React Native
    if (performance.memory) {
      // @ts-ignore
      const { usedJSHeapSize, totalJSHeapSize } = performance.memory;
      const usagePercent = (usedJSHeapSize / totalJSHeapSize) * 100;
      
      console.log(`[Performance] Memory usage: ${usagePercent.toFixed(2)}%`);
      
      if (usagePercent > 80) {
        console.warn('[Performance] High memory usage detected! Consider cleanup.');
      }
    }
  }

  /**
   * Send metrics to analytics service
   */
  private sendToAnalytics(event: string, data: Record<string, any>): void {
    // Implement your analytics integration here
    // Example: Firebase Analytics, Mixpanel, etc.
    console.log(`[Analytics] ${event}:`, data);
  }

  /**
   * Clear all metrics
   */
  public clearMetrics(): void {
    this.metrics.clear();
  }
}

// Export singleton instance
export const performanceMonitor = PerformanceMonitor.getInstance();

// Export HOC for component performance monitoring
export function withPerformanceMonitoring<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  componentName: string
) {
  return function WithPerformanceMonitoring(props: P) {
    React.useEffect(() => {
      performanceMonitor.startMetric(`${componentName}_mount`);
      
      return () => {
        const duration = performanceMonitor.endMetric(`${componentName}_mount`);
        if (duration && duration > 500) {
          console.warn(`[Performance] ${componentName} mount took ${duration.toFixed(2)}ms`);
        }
      };
    }, []);

    return <WrappedComponent {...props} />;
  };
}

export default performanceMonitor;
