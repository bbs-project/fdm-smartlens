import { useState, useEffect } from "react";
import { View } from "react-native";
import { Camera, CameraType } from "expo-camera";
import { GLView } from "expo-gl";
import Expo2DContext from "expo-2d-context";
import * as tf from "@tensorflow/tfjs";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { preprocess } from "../utils/preprocess";
import { renderBoxes } from "../utils/renderBox";

const TensorCamera = cameraWithTensors(Camera);

const CameraView = ({ type, model, inputTensorSize, config, children }) => {
  const [ctx, setCTX] = useState(null);
  const typesMapper = { back: CameraType.back, front: CameraType.front };

  const cameraStream = (images) => {
    const detectFrame = async () => {
      tf.engine().startScope();
      const imageTensor = images.next().value;
      if (imageTensor) {
        console.log("Image tensor shape:", imageTensor.shape);
        const [input, xRatio, yRatio] = preprocess(
          imageTensor,
          inputTensorSize[2],
          inputTensorSize[1]
        );

        await model.executeAsync(input).then((res) => {
          const [boxes, scores, classes] = res.slice(0, 3);
          const boxesData = boxes.dataSync();
          const scoresData = scores.dataSync();
          const classesData = classes.dataSync();

          renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
          tf.dispose([res, input]);
        });
      }
      requestAnimationFrame(detectFrame);
      tf.engine().endScope();
    };

    detectFrame();
  };

  return (
    <>
      {ctx && (
        <TensorCamera
          style={{ width: "100%", height: "100%", zIndex: 0 }}
          type={typesMapper[type]}
          cameraTextureHeight={inputTensorSize[1]}
          cameraTextureWidth={inputTensorSize[2]}
          resizeHeight={inputTensorSize[1]}
          resizeWidth={inputTensorSize[2]}
          resizeDepth={inputTensorSize[3]}
          onReady={cameraStream}
          autorender={true}
        />
      )}
      <View style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", zIndex: 10 }}>
        <GLView
          style={{ width: "100%", height: "100%" }}
          onContextCreate={async (gl) => {
            const ctx2d = new Expo2DContext(gl);
            await ctx2d.initializeText();
            setCTX(ctx2d);
          }}
        />
      </View>
      {children}
    </>
  );
};

export default CameraView;
