import React, { useEffect, useState, useCallback } from "react";
import { Text, TouchableOpacity, View, ActivityIndicator } from "react-native";
import { Camera } from "expo-camera/legacy";
import { StatusBar } from "expo-status-bar";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { yoloModelURI, vggModelURI } from "./modelHandler";
import CameraView from "./CameraView";

const App = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState("back");
  const [yoloModel, setYoloModel] = useState(null);
  const [vggModel, setVggModel] = useState(null);
  const [loading, setLoading] = useState({ loading: true, progress: 0, error: null });
  const [inputTensor, setInputTensor] = useState([]);

  // Model configuration
  const configurations = { threshold: 0.25 };

  const loadModels = useCallback(async () => {
    try {
      // Request camera permission
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");

      if (status !== "granted") {
        setLoading({ loading: false, progress: 1, error: "Camera permission denied" });
        return;
      }

      // Wait for TensorFlow to be ready
      await tf.ready();
      console.log("[App] TensorFlow ready");

      // Load YOLOv8 model
      console.log("[App] Loading YOLOv8 model...");
      const yolov8 = await tf.loadGraphModel(yoloModelURI, {
        onProgress: (fraction) => {
          setLoading(prev => ({ ...prev, loading: true, progress: fraction * 0.5 }));
        },
      });

      // Load VGG16 model
      console.log("[App] Loading VGG16 model...");
      const vgg16 = await tf.loadGraphModel(vggModelURI, {
        onProgress: (fraction) => {
          setLoading(prev => ({ ...prev, loading: true, progress: 0.5 + fraction * 0.5 }));
        },
      });

      // Warm up YOLO model
      console.log("[App] Warming up models...");
      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      await yolov8.executeAsync(dummyInput);
      tf.dispose(dummyInput);

      // Update state
      setInputTensor(yolov8.inputs[0].shape);
      setYoloModel(yolov8);
      setVggModel(vgg16);
      setLoading({ loading: false, progress: 1, error: null });
      
      console.log("[App] Models loaded successfully");
    } catch (error) {
      console.error("[App] Model loading error:", error);
      setLoading({ loading: false, progress: 1, error: error.message });
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

  // Render loading state
  if (loading.loading) {
    return (
      <View className="flex-1 items-center justify-center bg-white">
        <ActivityIndicator size="large" color="#0000ff" />
        <Text className="text-lg mt-4">
          Loading models... {(loading.progress * 100).toFixed(0)}%
        </Text>
        {loading.error && (
          <Text className="text-red-500 mt-2">Error: {loading.error}</Text>
        )}
      </View>
    );
  }

  // Render permission denied state
  if (hasPermission === false) {
    return (
      <View className="flex-1 items-center justify-center bg-white">
        <Text className="text-lg">Camera permission not granted!</Text>
        <TouchableOpacity 
          className="mt-4 bg-blue-500 px-6 py-3 rounded-lg"
          onPress={loadModels}
        >
          <Text className="text-white font-semibold">Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Render main camera view
  return (
    <View className="flex-1 items-center justify-center bg-white">
      <View className="flex-1 w-full h-full">
        <View className="flex-1 w-full h-full items-center justify-center">
          <CameraView
            type={type}
            yoloModel={yoloModel}
            vggModel={vggModel}
            inputTensorSize={inputTensor}
            config={configurations}
          >
            <View className="absolute left-0 top-0 w-full h-full flex justify-end items-center bg-transparent z-20">
              <TouchableOpacity
                className="flex flex-row items-center bg-transparent border-2 border-white p-3 mb-10 rounded-lg"
                onPress={() => setType((current) => (current === "back" ? "front" : "back"))}
              >
                <MaterialCommunityIcons
                  className="mx-2" name="camera-flip" size={30} color="white"
                />
                <Text className="mx-2 text-white text-lg font-semibold">Flip Camera</Text>
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
