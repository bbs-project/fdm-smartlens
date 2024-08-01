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

const CameraView = ({ type, model, modelVgg16, inputTensorSize, config, children }) => {
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

      // imageTensor: [640,640,3]
      if (imageTensor) {
        // Change inputTensor to [1,3,640,640] from [640,640,3]
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

        await model.executeAsync(input).then((res) => {
          console.log("[YOLOv8] model.executeAsync():", res)
//          const [boxes, scores, classes] = res.slice(0, 3);
          const numDetections = res.shape[1];
          const boxes = res.slice([0, 0, 0], [1, numDetections, 4]).squeeze(); // Extract bounding boxes
          const scores = res.slice([0, 0, 4], [1, numDetections, 1]).squeeze(); // Extract scores
          const classes = res.slice([0, 0, 5], [1, numDetections, 1]).squeeze(); // Extract classes

          const boxesData = boxes.dataSync();
          const scoresData = scores.dataSync();
          const classesData = classes.dataSync();

          renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
          tf.dispose([res, input]);
        });

        // 1. Add a batch dimension (making it [1, 640, 640, 3])
        const batchedTensor = tf.expandDims(imageTensor, 0);

        // 2. Resize the image to 112x112
        const resizedTensorVGG16 = tf.image.resizeBilinear(batchedTensor, [112, 112]);
        // console.log("[VGG16] resized shape:", resizedTensorVGG16.shape); // Output: [1, 112, 112, 3]

//        await modelVgg16.executeAsync(resizedTensorVGG16).then((res) => {
//          console.log("[VGG16] modelVgg16.executeAsync() result:", res);
//
//          // -----------------------------------
//          // Process the result here.
//
//          //evaluateModel().then(log => {
//          //  // Do something with the evaluation log
//          //});
//          // -----------------------------------
//          tf.dispose([res, resizedTensorVGG16]);
//        });


          // Run predict() using trained VGG16 model
          const pred = await modelVgg16.predict(resizedTensorVGG16).array(); // Get predictions as an array
          if (pred) {
            console.log("[VGG16] model.predict():", pred);

            const result = pred.map(idx =>
              idx.map(value => Math.round(value)) // Round and convert to integers
            );

            // 최종 분류 라벨 7가지는 다음과 같다.
            // (disease1,disease2,disease6,disease8,disease11,disease13,disease19)
            // * 01 : 바이러스성출혈성패혈증
            // * 02 : 림포시스티스병
            // * 06 : 여윔병
            // * 08 : 스쿠티카병
            // * 11 : 연쇄구균증
            // * 13 : 비브리오병
            // * 19 : 에드워드병
            const classes = [1, 2, 6, 8, 11, 13, 19];

            // 'result' is an array of arrays containing binary predictions (0 or 1)
            // and 'classes' is an array of corresponding class labels

            // Convert prediction results to codes
            const resultCode = result.map(value => {
              const code = [];
              for (let i = 0; i < value.length; i++) {
                if (value[i] === 1) {
                  code.push(classes[i]);
                }
              }
              return code;
            });

            // Convert prediction codes to strings (for the first prediction only)
            const resultStr = [];
            for (let i = 0; i < resultCode[0].length; i++) {
              const value = resultCode[0][i]; // Accessing the first prediction
              switch (value) {
                case 1:
                  resultStr.push("바이러스성출혈성패혈증");
                  break;
                case 2:
                  resultStr.push("림포시스티스병");
                  break;
                case 6:
                  resultStr.push("여윔병");
                  break;
                case 8:
                  resultStr.push("스쿠티카병");
                  break;
                case 11:
                  resultStr.push("연쇄구균증");
                  break;
                case 13:
                  resultStr.push("비브리오병");
                  break;
                case 19:
                  resultStr.push("에드워드병");
                  break;
                default:
                  break; // Handle cases where the value doesn't match any class
              }
            }

            // resultCode: Array of code arrays
            // resultStr: Array of disease names for the first prediction
            console.log("    - resultCode:", resultCode, ", resultStr:", resultStr);

            tf.dispose([pred, resizedTensorVGG16]);
          }

//          const res = model.execute(input)
//          // console.log("model.execute() result:", res)
//          const numDetections = res.shape[1];
//          const boxes = res.slice([0, 0, 0], [1, numDetections, 4]).squeeze(); // Extract bounding boxes
//          const scores = res.slice([0, 0, 4], [1, numDetections, 1]).squeeze(); // Extract scores
//          const classes = res.slice([0, 0, 5], [1, numDetections, 1]).squeeze(); // Extract classes
//
//          //const [boxes, scores, classes] = res.slice(0, 3);
//          const boxesData = boxes.dataSync();
//          const scoresData = scores.dataSync();
//          const classesData = classes.dataSync();
//          renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
//          tf.dispose([res, input]);

      } else {
        // console.log("No image tensor found.");
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
