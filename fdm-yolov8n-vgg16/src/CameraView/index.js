// React-related imports
import { useState, useEffect } from "react";
import { StyleSheet, Text, View } from "react-native";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import * as tf from "@tensorflow/tfjs";
//import { YellowBox } from 'react-native';
import { LogBox } from 'react-native';

// YellowBox.ignoreWarnings(['This model execution did not contain any nodes with control flow or dynamic output shapes']);
LogBox.ignoreLogs(['This model execution did not contain any nodes with control flow or dynamic output shapes','High memory usage in GPU',]);
// Expo-related imports
import { Camera, CameraType } from "expo-camera";
import { GLView } from "expo-gl";
import Expo2DContext from "expo-2d-context";

// Local imports
import { preprocess } from "../utils/preprocess";
import { renderBoxes } from "../utils/renderBox";
import { renderYoloBoxes } from "../utils/renderBox";
import labels from "../utils/labels.json";

// Number of classes predicted by the YOLOv8 model
const numClass = labels.length;
const TensorCamera = cameraWithTensors(Camera);

const CameraView = ({ type, yoloModel, vggModel, inputTensorSize: inputShape, config, children }) => {
  // console.log("CameraView enabled: type=", type, ", inputTensorSize=", inputTensorSize, ", config=", config);
  
  const [ctx, setCTX] = useState(null);
  const [vggOutputs, setVggOutputs] = useState([]);
  const typesMapper = { back: CameraType.back, front: CameraType.front };

  // Throttling function
  const throttle = (func, limit) => {
    let inThrottle;
    return function() {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  };

  // Executed for every new frame from camera
  const cameraStream = (images) => {

    const detectFrame = async () => {
      // console.log("Detected a frame from camera.");
      
      // inputShape: [1, 3, 640, 640]
      // console.log("inputShape:", inputShape);

      tf.engine().startScope();

      const imageTensor = images.next().value; // tf.Tensor3D
      // imageTensor = tf.Tensor3D ([640, 640, 3]);
      if (imageTensor) {        

        // Transpose imageTensor from [640, 640, 3] to [1, 3, 640, 640] = [batchSize, channels, height, width]
        const transposedTensor = imageTensor.transpose([2, 0, 1]); 
        // inputShape = [batchSize, height, width, channels]
        // but transposedTensor has [3, 640, 640]
        const [modelHeight, modelWidth] = inputShape.slice(1, 3); // get model height(1) and width(2)
        const [input, xRatio, yRatio] = preprocess(transposedTensor, modelWidth, modelHeight); // preprocess image
 
        // -----------------------------------
        const { boxesData, scoresData, classesData } = await detectYoloBoxes([input]);
        const { vggBoxesData, vggClasses, vggKlasses, vggScores } = await detectVggBoxes();
        // -----------------------------------
        
        // let xrate = ctx.width / 640; // modelWidth
        // let yrate = ctx.height / 640; // modelHeight

        ctx.clearRect(0, 0, ctx.width, ctx.height); // clean canvas, 캔버스를 초기화

        // renderYoloBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
        renderYoloBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
        // VggBoxes are rendered in the View

        ctx.flush(); // 화면에 실제로 반영되도록 함
         
        tf.dispose([input]);
        tf.dispose([boxesData, scoresData, classesData]);
        tf.dispose([vggBoxesData, vggClasses, vggKlasses, vggScores]);  

      } else {
        // console.log("No image tensor found.");
      }
    
      // console.log("Requesting animation frame.");
      throttledDetectFrame = throttle(detectFrame, 100); // Adjust the limit as needed (e.g., 100ms)

      // requestAnimationFrame(detectFrame);
      requestAnimationFrame(throttledDetectFrame);

      tf.engine().endScope();

      // Detect boxes using YOLOv8 model
      async function detectYoloBoxes(params) {
        const [input] = params;

       
        // // "shape": [1, 9, 8400] => [1, 8400, 9]
        //   const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
        //   const boxes = tf.tidy(() => {
        //     const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
        //     const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
        //     const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
        //     const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
        //     return tf
        //       .concat(
        //         [
        //           y1,
        //           x1,
        //           tf.add(y1, h), //y2
        //           tf.add(x1, w), //x2
        //         ],
        //         2
        //       )
        //       .squeeze();
        //   }); // process boxes [y1, x1, y2, x2]
        //   console.log("boxes.shape:", boxes.shape);

        //   const [scores, classes] = tf.tidy(() => {
        //     // class scores
        //     const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
        //     // console.log("rawScores:", rawScores);
        //     // print rawScores values
        //     rawScores.data().then(data => {
        //       console.log("rawScores data.shape:", data.shape);
        //     });
        //     return [rawScores.max(1), rawScores.argMax(1)];
        //   }); // get max scores and classes index

        //   // const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes
        //   const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes          
        //   console.log("nms.shape:", nms.shape);

        //   const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
        //   const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
        //   const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

        //   renderBoxes(ctx, boxes_data, scores_data, classes_data, [xRatio, yRatio]); // render boxes
        //   tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory

        //   //renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
        //   //tf.dispose([res, input]);
        // //});


        const yoloRes = await yoloModel.executeAsync(input); 
        // const yoloRes = yoloModel.predict(input);
        // const yoloRes = yoloModel.execute(input);
        yoloRes.dataSync();

  
        // ---------------------------------------------------------------------------------------
        // o dataId: An identifier for the tensor's data
        // o dtype: The data type of the tensor elements, which is "float32" (single-precision floating-point numbers)
        // o id: A unique identifier for the tensor
        // o isDisposedInternal: Indicates whether the tensor has been disposed of internally (false here)
        // o kept: Whether the tensor is being kept in memory for future use (false here)
        // o rankType: The rank (number of dimensions) of the tensor, which is "3" (a 3D tensor)
        // o scopeId: Related to TensorFlow.js's internal memory management.
        // o shape: The dimensions of the tensor: [1, 9, 8400]
        //   - 1: Batch size (processing a single image)
        //   - 9: Number of anchor boxes per grid cell (YOLOv8 uses 9 anchors)
        //   - 8400: Number of grid cells (e.g., a grid of 70x120 cells)
        // o size: The total number of elements in the tensor (75600 = 1 * 9 * 8400).
        // o strides: Information about how to access elements in the tensor's memory layout.
        //            Understanding the output The tensor likely contains the raw output from the YOLOv8 model after processing an image.
        //   - Bounding box coordinates: For each detected object.
        //   - Objectness scores: Confidence levels that an object is present in a bounding box.
        //   - Class probabilities: Probabilities for each object class the model is trained to detect. Further processing You would typically need to perform post-processing on this raw output tensor to:
        //   - Filter out low-confidence detections: Remove bounding boxes with low objectness scores.
        //   - Apply non-maximum suppression (NMS): Eliminate overlapping bounding boxes for the same object.
        //   - Extract relevant information: Get the final bounding box coordinates, class labels, and confidence scores for the detected objects. By understanding the structure and contents of this output tensor, you can effectively process the results of your YOLOv8 model and use them for object detection in your application.

        // Check the prediction results
        if (Array.isArray(yoloRes)) {
          yoloRes.forEach((item, i) => {
            item.data().then(data => {
              // console.log(`[YOLO] ${i}th data:`, data);
              console.log("[YOLO]", i, "th data: ", data);
            });
          });
        } else {
          // If res is not an array, it's a tensor
          yoloRes.data().then(data => {            
            // console.log('[YOLO] single data:', data);
            // print number of elements in data
            // console.log("data.length:", data.length);
          });
        }
 
        const maxNumber = 1;
        const iouThreshold = 0.45;
        const scoreThreshold = 0.5;

        // -----------------------------------------------------------------------------------
        // Version 1
        // -----------------------------------------------------------------------------------

        // transpose result [b, det, n] => [b, n, det]
        // [1, 9, 8400] -> [1, 8400, 9]
        const transRes = yoloRes.transpose([0, 2, 1]); 

        // Assuming reshapedOutput is a tensor of shape [1, 8400, 9]
        // Convert tensor to array and get the first element
        const reshapedOutput = transRes.arraySync()[0]; 
        
        let boxes = [];
        let scores = [];
        let classes = [];
        let klasses = [];
          // Process each prediction
        for (let i = 0; i < reshapedOutput.length; i++) {
            const prediction = reshapedOutput[i];
            // console.log("[YOLOv8] prediction: ", i, " : ", prediction);
            // Extract bounding box coordinates (assuming they are the first 4 values)
            const x1 = prediction[0];
            const x2 = prediction[1];
            const y1 = prediction[2];
            const y2 = prediction[3];
            boxes.push([x1, y1, x2, y2]);

            // Find the class with the highest confidence (assuming class confidences are the last 5 values)
            let maxConfidence = 0;
            let maxClassIndex = -1;
            for (let j = 4; j < prediction.length; j++) {
                if (prediction[j] > maxConfidence) {
                    maxConfidence = prediction[j];
                    maxClassIndex = j - 4; // Adjust index to get class index
                }
            }

            scores.push(maxConfidence);
            classes.push(maxClassIndex);
            
            // Get class name using the index
            const className = labels[maxClassIndex];

            // ## 증상명
            // * 01 : 출혈 (Bleeding)
            // * 02 : 궤양 (Corrosion)
            // * 03 : 부식 (Tumor)
            // * 04 : 종양 (Ulcer)
            // * 05 : 안구증상 (eyesSymptom)

            klasses.push(className);

            // Now you have the class name, confidence, and bounding box for this prediction
            // You can use this information as needed in your application
            // console.log(`${i} - Class: ${className}, Confidence: ${maxConfidence}, Box: [${x}, ${y}, ${width}, ${height}]`);
        }

        const boxesTensor = tf.tensor2d(boxes);
        const scoresTensor = tf.tensor1d(scores);
        const classesTensor = tf.tensor1d(classes);

        const nms = await tf.image.nonMaxSuppressionAsync(boxesTensor, scoresTensor, maxNumber, iouThreshold, scoreThreshold); // NMS to filter boxes
        const boxesData = boxesTensor.gather(nms, 0).dataSync(); // indexing boxes by nms index
        const scoresData = scoresTensor.gather(nms, 0).dataSync(); // indexing scores by nms index
        const classesData = classesTensor.gather(nms, 0).dataSync(); // indexing classes by nms index

        
        // -----------------------------------------------------------------------------------
        // Version 2
        // -----------------------------------------------------------------------------------
        // const boxes = tf.tidy(() => {
        //   const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
        //   const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
        //   const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
        //   const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
        //   //console.log("- w:", w.dataSync(), ", h:", h.dataSync(), ", x1:", x1.dataSync(), ", y1:", y1.dataSync()); 
        //   return tf
        //     .concat(
        //       [
        //         y1,
        //         x1,
        //         tf.add(y1, h), //y2
        //         tf.add(x1, w), //x2
        //       ],
        //       2
        //     )
        //     .squeeze();
        // }); // process boxes [y1, x1, y2, x2]

        // const [scores, classes] = tf.tidy(() => {
        //   // class scores
        //   const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
        //   return [rawScores.max(1), rawScores.argMax(1)];
        // }); // get max scores and classes index
      
        // // Convert boxes, scores, and classes to tensors
      
        // const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, maxNumber, iouThreshold, scoreThreshold); // NMS to filter boxes
      
        // const boxesData = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
        // const scoresData = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
        // const classesData = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

        // -----------------------------------------------------------------------------------
        // End of Detection
        // -----------------------------------------------------------------------------------

        tf.dispose([yoloRes, transRes, nms]);

        return { boxesData, scoresData, classesData };
      }

      // Detect prediction boxes using VGG16 model
      async function detectVggBoxes() {
        // 1. Add a batch dimension (making it [1, 640, 640, 3]) to the image
        const batchedTensor = tf.expandDims(imageTensor, 0);

        // 2. Resize the image to 112x112
        const vggInput = tf.image.resizeBilinear(batchedTensor, [112, 112]);
       
        const vggBoxesData = [];
        const vggClasses = []; // class codes
        const vggKlasses = []; // class names
        const vggScores = []; // class scores

        await vggModel.executeAsync(vggInput).then((vggRes) => {
          // await vggModel.predict(vggInput).then((res) => {
          // Check the prediction results
          if (Array.isArray(vggRes)) {
            vggRes.forEach((item, i) => {
              item.data().then(data => {
                // console.log(`[VGG16] ${i}th data:`, data);
                // console.log("[VGG16]", i, "th data: ", data);
              });
            });
          } else {
            // If res is not an array, it's a tensor
            vggRes.data().then(data => {
              // '[VGG16] data:', { '0': 0, '1': 0, '2': 1, '3': 0, '4': 0, '5': 1, '6': 0 }
              // console.log("[VGG16] data:", data);
            });
          }

          // result:, [ [ 0, 0, 1, 0, 0, 1, 0 ] ]
          const pred = vggRes.arraySync(); // Get predictions as an array          


          // console.log("[VGG16] model.predict() result:", pred);
          // const pred = await vggModel.predict(resizedTensorVGG16).array(); // Get predictions as an array
          if (pred) {
            // console.log("[VGG16] model.predict():", pred);
            const result = pred.map(idx => idx.map(value => Math.round(value)) // Round and convert to integers
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
            // const resultCode = result.map(value => {
            const resultCodes = result.map(value => {
              const code = [];
              for (let i = 0; i < value.length; i++) {
                if (value[i] === 1) {
                  code.push(classes[i]);
                }
              }
              return code;
            });
            // push each of the result codes to the vggClasses array
            resultCodes[0].forEach(code => {
              //console.log("code:", code);
              vggClasses.push(code);
              vggScores.push(1.0); // Dummy score
            });
            //vggClasses.push(resultCodes);
            // Convert prediction codes to strings (for the first prediction only)
            const outputs = [];
            for (let i = 0; i < vggClasses.length; i++) {
              const value = vggClasses[i]; // Accessing the first prediction

              // let klass readable string
              let klass = "";
              // const klass= "";
              switch (value) {
                case 1:
                  klass = "바이러스성출혈성패혈증";
                  break;
                case 2:
                  klass = "림포시스티스병";
                  break;
                case 6:
                  klass = "여윔병";
                  break;
                case 8:
                  klass = "스쿠티카병";
                  break;
                case 11:
                  klass = "연쇄구균증";
                  break;
                case 13:
                  klass = "비브리오병";
                  break;
                case 19:
                  klass = "에드워드병";
                  break;
                default:
                  break; // Handle cases where the value doesn't match any class
              }
              vggKlasses.push(klass);
              outputs.push({ code: value, name: klass });
              setVggOutputs(outputs);

              // vggClasses[{code: 6, name: "여윔병"}, {code: 13, name: "비브리오병"}];
            }

            // resultCode: Array of code arrays
            // resultStr: Array of disease names for the first prediction
            //console.log("    - resultCode:", resultCode, ", resultStr:", resultStr);
            const numRes = vggClasses.length;
            // create array of bounding box data for the predictions
            // const vggBoxesData = Array(numRes);
            // const vggBoxesData = []; // Dummy bounding box data
            for (let i = 0; i < numRes; i++) {
              // create vggBoxesData for the prediction and push it to the array
              vggBoxesData.push([0, 0, 100, 200 * (i + 1)]);
            }

            // for (let i = 0; i < numRes; i++) {              
            //   console.log("[VGG]", i, ": ", vggOutputs[i]);
            // }      
            // 'vggBoxesData:', [ [ 0, 0, 10, 20 ], [ 0, 0, 10, 40 ] ]
            // 'resultCode:', [ [ 6, 13 ] ]
            // 'resultStr:', [ '스쿠티카병', '비브리오병' ]
            // console.log("[VGG16] boxesData:", vggBoxesData, ", scores:", vggScores, ", classes:", vggClasses, ", klasses:", vggKlasses); 
            // renderBoxes(ctx, config.threshold, vggBoxesData, resultCode, resultStr, [xRatio, yRatio]);
            // tf.dispose([pred, vggInput]);    
          }

          tf.dispose([vggRes, vggInput]);
        }).catch((error) => {
          console.log("Error in executing model:", error);
        });
        return { vggBoxesData, vggClasses, vggKlasses, vggScores };
      }
    }
   
    // console.log("Requesting animation frame.");
    // Wrap detectFrame with the throttling function
    // detectFrame();
    let throttledDetectFrame = throttle(detectFrame, 500); // Adjust the limit as needed (e.g., 100ms)

    // Start the detection loop
    requestAnimationFrame(throttledDetectFrame);
  };

  return (
    <>
      {ctx && (
        <TensorCamera
          style={{ width: "100%", height: "100%", zIndex: 0 }}
          type={typesMapper[type]}
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
  // containerPortrait: {
  //   position: 'relative',
  //   width: CAM_PREVIEW_WIDTH,
  //   height: CAM_PREVIEW_HEIGHT,
  //   marginTop: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  // },
  // containerLandscape: {
  //   position: 'relative',
  //   width: CAM_PREVIEW_HEIGHT,
  //   height: CAM_PREVIEW_WIDTH,
  //   marginLeft: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  // },
  // loadingMsg: {
  //   position: 'absolute',
  //   width: '100%',
  //   height: '100%',
  //   alignItems: 'center',
  //   justifyContent: 'center',
  // },
  // camera: {
  //   width: '100%',
  //   height: '100%',
  //   zIndex: 1,
  // },
  // svg: {
  //   width: '100%',
  //   height: '100%',
  //   position: 'absolute',
  //   zIndex: 30,
  // },

  vggContainer: {
    position: 'absolute',
    top: 30,
    left: 10,
    width: 370,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  // cameraTypeSwitcher: {
  //   position: 'absolute',
  //   top: 10,
  //   right: 10,
  //   width: 180,
  //   alignItems: 'center',
  //   backgroundColor: 'rgba(255, 255, 255, .7)',
  //   borderRadius: 2,
  //   padding: 8,
  //   zIndex: 20,
  // },
});


export default CameraView;
