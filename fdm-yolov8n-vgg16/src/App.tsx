import React, { useEffect, useState, useCallback } from 'react';
import { Text, TouchableOpacity, View, ActivityIndicator } from 'react-native';
import { Camera } from 'expo-camera/legacy';
import { StatusBar } from 'expo-status-bar';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { yoloModelURI, vggModelURI } from './modelHandler';
import CameraView from './CameraView';
import { LoadingState, ModelConfig } from './types';
import { performanceMonitor } from './utils/performance';

// Start app launch performance tracking
performanceMonitor.startMetric('app_launch');

interface TensorInfo {
  shape: number[];
}

const App: React.FC = () => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [type, setType] = useState<'front' | 'back'>('back');
  const [yoloModel, setYoloModel] = useState<tf.GraphModel | null>(null);
  const [vggModel, setVggModel] = useState<tf.GraphModel | null>(null);
  const [loading, setLoading] = useState<LoadingState>({ loading: true, progress: 0, error: null });
  const [inputTensor, setInputTensor] = useState<number[]>([]);

  const configurations: ModelConfig = { threshold: 0.25 };

  const loadModels = useCallback(async () => {
    try {
      setLoading({ loading: true, progress: 0, error: null });

      // Request camera permission
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');

      if (status !== 'granted') {
        setLoading({ loading: false, progress: 1, error: 'Camera permission denied' });
        return;
      }

      // Wait for TensorFlow to be ready
      await tf.ready();
      console.log('[App] TensorFlow ready');
      performanceMonitor.endMetric('app_launch');

      // Load YOLOv8 model
      console.log('[App] Loading YOLOv8 model...');
      performanceMonitor.startMetric('yolo_load');
      const yolov8 = await tf.loadGraphModel(yoloModelURI, {
        onProgress: (fraction: number) => {
          setLoading(prev => ({ ...prev, loading: true, progress: fraction * 0.5 }));
        },
      });
      const yoloDuration = performanceMonitor.endMetric('yolo_load');
      if (yoloDuration) performanceMonitor.trackModelLoading('YOLOv8', yoloDuration);

      // Load VGG16 model
      console.log('[App] Loading VGG16 model...');
      performanceMonitor.startMetric('vgg_load');
      const vgg16 = await tf.loadGraphModel(vggModelURI, {
        onProgress: (fraction: number) => {
          setLoading(prev => ({ ...prev, loading: true, progress: 0.5 + fraction * 0.5 }));
        },
      });
      const vggDuration = performanceMonitor.endMetric('vgg_load');
      if (vggDuration) performanceMonitor.trackModelLoading('VGG16', vggDuration);

      // Warm up YOLO model
      console.log('[App] Warming up models...');
      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      await yolov8.executeAsync(dummyInput);
      tf.dispose(dummyInput);

      // Update state
      setInputTensor(yolov8.inputs[0].shape);
      setYoloModel(yolov8);
      setVggModel(vgg16);
      setLoading({ loading: false, progress: 1, error: null });
      
      console.log('[App] Models loaded successfully');
    } catch (error) {
      console.error('[App] Model loading error:', error);
      setLoading({ 
        loading: false, 
        progress: 1, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  }, []);

  useEffect(() => {
    loadModels();
    
    // Cleanup on unmount
    return () => {
      if (yoloModel) yoloModel.dispose();
      if (vggModel) vggModel.dispose();
      tf.disposeVariables();
    };
  }, [loadModels]);

  const handleFlipCamera = () => {
    setType(prev => prev === 'back' ? 'front' : 'back');
  };

  // Render loading state
  if (loading.loading) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' }}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={{ fontSize: 16, marginTop: 16 }}>
          Loading models... {(loading.progress * 100).toFixed(0)}%
        </Text>
        {loading.error && (
          <Text style={{ color: '#ff0000', marginTop: 8 }}>Error: {loading.error}</Text>
        )}
      </View>
    );
  }

  // Render permission denied state
  if (hasPermission === false) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' }}>
        <Text style={{ fontSize: 16 }}>Camera permission not granted!</Text>
        <TouchableOpacity 
          style={{ marginTop: 16, backgroundColor: '#3b82f6', paddingHorizontal: 24, paddingVertical: 12, borderRadius: 8 }}
          onPress={loadModels}
        >
          <Text style={{ color: '#fff', fontWeight: '600' }}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Render main camera view
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' }}>
      <View style={{ flex: 1, width: '100%', height: '100%' }}>
        <View style={{ flex: 1, width: '100%', height: '100%', alignItems: 'center', justifyContent: 'center' }}>
          <CameraView
            type={type}
            yoloModel={yoloModel}
            vggModel={vggModel}
            inputTensorSize={inputTensor}
            config={configurations}
          >
            <View style={{ 
              position: 'absolute', 
              left: 0, 
              top: 0, 
              width: '100%', 
              height: '100%', 
              justifyContent: 'flex-end', 
              alignItems: 'center', 
              backgroundColor: 'transparent',
              zIndex: 20 
            }}>
              <TouchableOpacity
                style={{ 
                  flexDirection: 'row', 
                  alignItems: 'center', 
                  backgroundColor: 'transparent', 
                  borderWidth: 2, 
                  borderColor: '#fff', 
                  padding: 12, 
                  marginBottom: 40, 
                  borderRadius: 8 
                }}
                onPress={handleFlipCamera}
              >
                <MaterialCommunityIcons name="camera-flip" size={30} color="#fff" style={{ marginHorizontal: 8 }} />
                <Text style={{ marginHorizontal: 8, color: '#fff', fontSize: 16, fontWeight: '600' }}>Flip Camera</Text>
              </TouchableOpacity>
            </View>
          </CameraView>
        </View>
      </View>
      <StatusBar style="auto" />
    </View>
  );
};

export default App;
