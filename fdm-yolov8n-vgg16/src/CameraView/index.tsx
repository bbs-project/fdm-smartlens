// React-related imports
import { useState, useEffect, useRef, useCallback } from "react";
import { StyleSheet, Text, View } from "react-native";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import * as tf from "@tensorflow/tfjs";
import { LogBox } from 'react-native';

// Only ignore benign warnings, NOT memory warnings
LogBox.ignoreLogs(['This model execution did not contain any nodes with control flow or dynamic output shapes']);

// Expo-related imports
import { Camera, CameraType } from "expo-camera";
import { GLView } from "expo-gl";
import Expo2DContext from "expo-2d-context";

// Local imports
import { preprocess } from "../utils/preprocess";
import { detectYoloBoxes } from "../utils/detectBox";
import { detectVggBoxes } from "../utils/detectBox";
import { renderBoxes } from "../utils/renderBox";

const TensorCamera = cameraWithTensors(Camera);

const CameraView = ({ type, yoloModel, vggModel, inputTensorSize: inputShape, config, children }) => {
  const [ctx, setCTX] = useState(null);
  const [vggOutputs, setVggOutputs] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const animationFrameRef = useRef(null);
  const imagesRef = useRef(null);
  const cameraType = { back: CameraType.back, front: CameraType.front };

  // Cleanup function to cancel animation frame
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Detect an image from camera frame
  const detectFrame = useCallback(async () => {
    if (!ctx || !yoloModel || !vggModel || !imagesRef.current) return;

    tf.engine().startScope();

    try {
      const image = imagesRef.current.next().value;
      if (!image) {
        tf.engine().endScope();
        return;
      }

      // Transpose image from [640, 640, 3] to [1, 3, 640, 640]
      const transposedTensor = image.transpose([2, 0, 1]);
      const [modelHeight, modelWidth] = inputShape.slice(1, 3);
      const [input, xRatio, yRatio] = preprocess(transposedTensor, modelWidth, modelHeight);

      // YOLO Detection
      const [numDetections, boxesData, scoresData, classesData] = await detectYoloBoxes(yoloModel, [input]);

      if (numDetections > 0) {
        const [vggBoxesData, vggClasses, vggKlasses, vggScores, vOutputs] = await detectVggBoxes(vggModel, image);
        
        setVggOutputs(vOutputs);
        ctx.clearRect(0, 0, ctx.width, ctx.height);

        renderBoxes(ctx, config.threshold, numDetections, boxesData, scoresData, classesData, [xRatio, yRatio]);
        ctx.flush();

        // Cleanup tensors
        tf.dispose([vggBoxesData, vggClasses, vggKlasses, vggScores, vOutputs, input]);
        tf.dispose([numDetections, boxesData, scoresData, classesData]);
      } else {
        tf.dispose([input]);
      }
    } catch (error) {
      console.error("[CameraView] Detection error:", error);
    } finally {
      tf.engine().endScope();
      // Schedule next frame with throttling (500ms)
      setTimeout(() => {
        animationFrameRef.current = requestAnimationFrame(detectFrame);
      }, 500);
    }
  }, [ctx, yoloModel, vggModel, inputShape, config.threshold]);

  // Executed for every new frame from camera
  const cameraStream = useCallback((images) => {
    imagesRef.current = images;
    if (isDetecting) return;
    setIsDetecting(true);
    
    // Start the detection loop
    animationFrameRef.current = requestAnimationFrame(detectFrame);
  }, [detectFrame, isDetecting]);

  return (
    <>
      {ctx && (
        <TensorCamera
          style={{ width: "100%", height: "100%", zIndex: 0 }}
          type={cameraType[type]}
          cameraTextureHeight={inputShape[2]}
          cameraTextureWidth={inputShape[3]}
          resizeHeight={inputShape[2]}
          resizeWidth={inputShape[3]}
          resizeDepth={inputShape[1]}
          onReady={cameraStream}
          autorender={true}
        />
      )}
      {/* Create a 2D canvas for rendering bounding boxes  */}
      <View style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", zIndex: 10 }}>
        <GLView
          style={{ width: "100%", height: "100%" }}
          onContextCreate={async (gl) => {
            const ctx2d = new Expo2DContext(gl);
            await ctx2d.initializeText();
            setCTX(ctx2d);
          }}
        />
        {renderVgg(vggOutputs)}
      </View>
      {children}
    </>
  );
};

const renderVgg = (vggOutputs) => {
  return (
    <View style={styles.vggContainer}>
      {vggOutputs.map((vggOutput, index) => (
        <Text key={index}>질병명: {vggOutput.name}({vggOutput.code})</Text>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  vggContainer: {
    position: 'absolute',
    top: 30,
    left: '50%',
    transform: [{ translateX: -185 }],
    width: 370,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});

export default CameraView;
