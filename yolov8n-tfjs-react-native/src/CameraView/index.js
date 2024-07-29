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
  // console.log("CameraView enabled: type=", type, ", inputTensorSize=", inputTensorSize, ", config=", config);
  const [ctx, setCTX] = useState(null);
  const typesMapper = { back: CameraType.back, front: CameraType.front };

  // executed for every new frame from camera
  const cameraStream = (images) => {
    const detectFrame = async () => {
      // console.log("Detected a frame from camera.");
      //console.log("images: ", images);
      //console.log("typeof images: ", typeof images);
      let methods = Object.getOwnPropertyNames(images).filter(
        (property) => typeof images[property] === "function"
      );
      // console.log("Methods of images:", methods);

      // console.log("images.next: ", images.next);
      // console.log("images.next(): ", images.next());

//      let methods1 = Object.getOwnPropertyNames(images.next()).filter(
//        (property) => typeof images.next()[property] === "function"
//      );
      //console.log("Methods of images.next():", methods1);

      //console.log("images.next().value: ", images.next().value);
      //console.log("typeof images.next().value: ", typeof images.next().value);

      tf.engine().startScope();

      // const imageTensor = images.next().value;
      const imageTensorObj = images.next();
      const imageTensor = imageTensorObj.value;

      if (imageTensor) {
        // console.log("Image tensor shape:", imageTensor.shape); // [640, 640, 3]
        // const originalTensor = tf.tensor3d( /* your image data */, [640, 640, 3]);
        const transposedTensor = imageTensor.transpose([2, 0, 1]); // Transpose dimensions to [3, 640, 640]
        // console.log("Transposed tensor shape:", transposedTensor.shape)
        const [input, xRatio, yRatio] = preprocess(
          //imageTensor,
          //inputTensorSize[2],
          //inputTensorSize[1]
          transposedTensor,
          inputTensorSize[1],
          inputTensorSize[2]
        );

        // console.log("input:", input)

//        await model.executeAsync(input).then((res) => {
//          const [boxes, scores, classes] = res.slice(0, 3);
//          const boxesData = boxes.dataSync();
//          const scoresData = scores.dataSync();
//          const classesData = classes.dataSync();
//
//          renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
//          tf.dispose([res, input]);
//        });

          const res = model.execute(input)
          // console.log("model.execute() result:", res)
          const numDetections = res.shape[1];
          const boxes = res.slice([0, 0, 0], [1, numDetections, 4]).squeeze(); // Extract bounding boxes
          const scores = res.slice([0, 0, 4], [1, numDetections, 1]).squeeze(); // Extract scores
          const classes = res.slice([0, 0, 5], [1, numDetections, 1]).squeeze(); // Extract classes

          //const [boxes, scores, classes] = res.slice(0, 3);
          const boxesData = boxes.dataSync();
          const scoresData = scores.dataSync();
          const classesData = classes.dataSync();
          renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
          tf.dispose([res, input]);

      } else {
        console.log("No image tensor found.");
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
//          cameraTextureHeight={inputTensorSize[1]}
//          cameraTextureWidth={inputTensorSize[2]}
//          resizeHeight={inputTensorSize[1]}
//          resizeWidth={inputTensorSize[2]}
//          resizeDepth={inputTensorSize[3]}

          cameraTextureHeight={inputTensorSize[2]}
          cameraTextureWidth={inputTensorSize[3]}
          resizeHeight={inputTensorSize[2]}
          resizeWidth={inputTensorSize[3]}
          resizeDepth={inputTensorSize[1]}

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
