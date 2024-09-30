import { useState, useEffect } from "react";
import { View } from "react-native";
import { Camera, CameraType } from "expo-camera";
import { GLView } from "expo-gl";
import Expo2DContext from "expo-2d-context";
import * as tf from "@tensorflow/tfjs";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { preprocess } from "../utils/preprocess";
import { renderBoxes } from "../utils/renderBox";
import { renderYoloBoxes } from "../utils/renderBox";
import { renderVggBoxes } from "../utils/renderBox";
import labels from "../utils/labels.json";

// Number of classes predicted by the YOLOv8 model
const numClass = labels.length;
const TensorCamera = cameraWithTensors(Camera);

const CameraView = ({ type, yoloModel, vggModel, inputTensorSize, config, children }) => {
  console.log("CameraView enabled: type=", type, ", inputTensorSize=", inputTensorSize, ", config=", config);
  
  const [ctx, setCTX] = useState(null);
  const typesMapper = { back: CameraType.back, front: CameraType.front };

  // Executed for every new frame from camera
  const cameraStream = (images) => {
    const detectFrame = async () => {
      console.log("Detected a frame from camera.");
      //console.log("images: ", images);

      tf.engine().startScope();

      // const imageTensor = images.next().value;
      const imageTensorObj = images.next();
      const imageTensor = imageTensorObj.value;

      // Change inputTensor to [1,3,640,640] from [640,640,3]
      // imageTensor: [640,640,3]
      if (imageTensor) {        
        // 'Image tensor\'s shape:', [ 640, 640, 3 ]
        console.log("Image tensor's shape:", imageTensor.shape); // [640, 640, 3]

        // const originalTensor = tf.tensor3d( /* your image data */, [640, 640, 3]);
        // Transpose dimensions to [3, 640, 640]
        const transposedTensor = imageTensor.transpose([2, 0, 1]); 
        // console.log("Transposed tensor shape:", transposedTensor.shape) // [3, 640, 640]

        // const [input, xRatio, yRatio] = preprocess(
        //   //imageTensor,
        //   //inputTensorSize[2],
        //   //inputTensorSize[1]
        //   transposedTensor,
        //   inputTensorSize[1],
        //   inputTensorSize[2]
        // );
        const [modelWidth, modelHeight] = inputTensorSize.slice(1, 3); // get model width and height
        //const [modelWidth, modelHeight] = yoloModel.inputShape.slice(1, 3); // get model width and height
        const [input, xRatio, yRatio] = preprocess(transposedTensor, modelWidth, modelHeight); // preprocess image

        // -----------------------------------
        // Begin of YOLO model prediction
        // -----------------------------------

        // // const res = yoloModel.execute(input);
        // // console.log("res:", res);

        // await yoloModel.executeAsync(input).then((res) => {

        // const res = yoloModel.execute(input);
        // const res = await yoloModel.executeAsync(input);

        // // console.log("[YOLO] res:", res);
        // // if res has output, then process it
        // res.data().then(data => {
        //   // console.log("res data:", data);
        // });

        // //});
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

        // 'input:', { kept: false, isDisposedInternal: false, shape: [ 1, 3, 640, 640 ], dtype: 'float32', size: 1228800,
        //             strides: [ 1228800, 409600, 640 ], dataId: { id: 38724 }, id: 33303, rankType: '4', scopeId: 67914 }
        // console.log("input:", input)        

        const res = await yoloModel.executeAsync(input);
        // await yoloModel.executeAsync(input).then((res) => {
          // "res" contains {"dataId": {"id": 1216}, "dtype": "float32", "id": 884,
          //                 "isDisposedInternal": false, "kept": false, "rankType": "3",
          //                 "scopeId": 0, "shape": [1, 9, 8400], "size": 75600,
          //                 "strides": [75600, 8400]}
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
          if (Array.isArray(res)) {
            res.forEach((item, i) => {
              item.data().then(data => {
                // console.log(`[YOLO] ${i}th data:`, data);
                console.log("[YOLO]", i, "th data: ", data);
              });
            });
          } else {
            // If res is not an array, it's a tensor
            res.data().then(data => {
              // '[YOLO] single data:', { '0': 21.381683349609375,
              //                          '1': 43.794063568115234,
              //                          '2': 39.883846282958984,
              // ...
              //                          '57': 462.33428955078125,
              //                          '58': 470.3690185546875,
              // ...
              //                          '148': 548.0645751953125,
              //                          '149': 555
              // ...
              // console.log('[YOLO] single data:', data);
              // print number of elements in data
              // console.log("data.length:", data.length);
            });
          }

          // 'res': { kept: false, isDisposedInternal: false, shape: [ 1, 9, 8400 ], dtype: 'float32', size: 75600,
          //          strides: [ 75600, 8400 ], dataId: { id: 39230 }, id: 33752, rankType: '3', scopeId: 67914 }
          console.log("[YOLOv8] model.executeAsync():", res);
                    
          // const [boxes, scores, classes] = res.slice(0, 3);
          // shape: [1, 9, 8400]
          // const rankType = res.rankType;
          // const shape0 = res.shape[0]; // Batch size
          // const shape1 = res.shape[1]; // Numer of attributes per detection
          // const shape2 = res.shape[2]; // Number of detections         

          
          // // transpose result [b, det, n] => [b, n, det]
          // [1, 9, 8400] -> [1, 8400, 9]
          const transRes = res.transpose([0, 2, 1]); 
          const boxes = tf.tidy(() => {
            const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
            const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
            const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
            const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
            //console.log("- w:", w.dataSync(), ", h:", h.dataSync(), ", x1:", x1.dataSync(), ", y1:", y1.dataSync()); 
            return tf
              .concat(
                [
                  y1,
                  x1,
                  tf.add(y1, h), //y2
                  tf.add(x1, w), //x2
                ],
                2
              )
              .squeeze();
          }); // process boxes [y1, x1, y2, x2]


          const [scores, classes] = tf.tidy(() => {
            // class scores
            const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
            return [rawScores.max(1), rawScores.argMax(1)];
          }); // get max scores and classes index
        
          const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 5, 0.45, 0.5); // NMS to filter boxes
          // const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes
        
          const boxesData = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
          const scoresData = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
          const classesData = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

          // Extract boxes, scores, classes from res
          // res = res.reshape([shape0, shape1, shape2, 3, -1]);
          // res = res.arraySync();
         
          //const numDetections = res.shape[1];
          // const numDetections = shape2;
          // const boxes = res.slice([0, 0, 0], [1, numDetections, 4]).squeeze(); // Extract bounding boxes
          // const scores = res.slice([0, 0, 4], [1, numDetections, 1]).squeeze(); // Extract scores
          // const classes = res.slice([0, 0, 5], [1, numDetections, 1]).squeeze(); // Extract classes
          // const boxes = res.slice([0, 0, 0], [1, 4, numDetections]).squeeze(); // Extract bounding boxes
          // const scores = res.slice([0, 4, 0], [1, 1, numDetections]).squeeze(); // Extract scores
          // const classes = res.slice([0, 5, 0], [1, 1, numDetections]).squeeze(); // Extract classes

          //const [boxes, scores, classes] = res.slice(0, 3);
          // const tensorData = res.array();
          //const reshapedData = tensorData.reshape([1, 8400, 9]);
          // const boxes = tf.tensor1d(tensorData.slice(0, 8400*4));
          // const scores = tf.tensor1d(tensorData.slice(8400*4, 8400*5));
          // const classes = tf.tensor1d(tensorData.slice(8400*5, 8400*6));

          // let boxes = [];
          // for (let index=0; index<8400; index++) {
          //     const [class_id, prob] = [...Array(80).keys()]
          //         .map(col => [col, tensorData[8400*(col + 4)  +index]])
          //         .reduce((accum, item) => item[1]>accum[1] ? item : accum,[0,0]);
          //     if (prob < 0.5) {
          //         continue;
          //     }
          //     const label = yolo_classes[class_id];
          //     const xc = output[index];
          //     const yc = output[8400+index];
          //     const w = output[2*8400+index];
          //     const h = output[3*8400+index];
          //     const x1 = (xc-w/2)/640*img_width;
          //     const y1 = (yc-h/2)/640*img_height;
          //     const x2 = (xc+w/2)/640*img_width;
          //     const y2 = (yc+h/2)/640*img_height;
          //     boxes.push([x1,y1,x2,y2,label,prob]);
          // }
      
          // boxes = boxes.sort((box1,box2) => box2[5]-box1[5])
          // const result = [];
          // while (boxes.length > 0) {
          //     result.push(boxes[0]);
          //     boxes = boxes.filter(box => iou(boxes[0],box)<0.7);
          // }
          // // return result;
          // console.log("result:", result);

          // boxes[0]

          // 'boxes:', { kept: false, isDisposedInternal: false, shape: [ 9, 4 ], dtype: 'float32', size: 36, 
          //             strides: [ 4 ], dataId: { id: 38656 }, id: 33256, rankType: '2', scopeId: 66885 }
          //console.log("boxes:", boxes);

          // 'scores:', { kept: false, isDisposedInternal: false, shape: [ 9 ], dtype: 'float32', size: 9,
          //              strides: [], dataId: { id: 38658 }, id: 33258, rankType: '1', scopeId: 66885 }
          //console.log("scores:", scores);

          // 'classes:', { kept: false, isDisposedInternal: false, shape: [ 9 ], dtype: 'float32', size: 9,
          //               strides: [], dataId: { id: 38660 }, id: 33260, rankType: '1', scopeId: 66885 }
          //console.log("classes:", classes);

          // const boxesData = boxes.dataSync();
          // const scoresData = scores.dataSync();
          // const classesData = classes.dataSync();

          // 'boxesData:', { '0': 21.2984619140625,
          //                 '1': 43.83866882324219,
          //                 '2': 39.98420715332031,
          //                 '3': 44.731388092041016,
          //                 '4': 16.39701271057129,
          //                 '5': 17.825220108032227,
          //                 '6': 14.30429458618164,
          //                 '7': 15.371467590332031,
          //                 '8': 43.18511962890625,
          //                 '9': 89.7205810546875,
          // ...
          

          // 'scoresData:', { '0': 50.121177673339844,
          //                  '1': 11.156957626342773,
          //                  '2': 108.65730285644531,
          //                  '3': 23.23900604248047,
          //                  '4': 0.000024720056899241172,
          //                  '5': 0.000030405779398279265,
          //                  '6': 0.0000036065116546524223,
          //                  '7': 0.0000501406830153428,
          //                  '8': 4.7644746814512473e-7 }
         

          // 'classesData:', { '0': 53.163238525390625,
          //                   '1': 9.674726486206055,
          //                   '2': 134.7602996826172,
          //                   '3': 20.17319107055664,
          //                   '4': 0.00002437926195852924,
          //                   '5': 0.000035004726669285446,
          //                   '6': 0.0000024664916509209434,
          //                   '7': 0.00007081677176756784,
          //                   '8': 5.959183795312128e-7 }

          console.log("[YOLOv8] boxesData:", boxesData, ", scoresData:", scoresData, ", classesData:", classesData);

          //print length of boxesData, scoresData, classesData
          // console.log("boxesData.length:", boxesData.length);
          // console.log("scoresData.length:", scoresData.length);
          // console.log("classesData.length:", classesData.length);

          // renderBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
          // tf.dispose([res, input, transRes, boxes, scores, classes, nms]);
        // }).catch((error) => {
        //   console.log("Error in executing model:", error);
        // });
        
        // -----------------------------------
        // End of YOLO model prediction
        // -----------------------------------

        // -----------------------------------
        // Begin of VGG16 model prediction
        // -----------------------------------

        // 1. Add a batch dimension (making it [1, 640, 640, 3])
        const batchedTensor = tf.expandDims(imageTensor, 0);

        // 2. Resize the image to 112x112
        const vggInput = tf.image.resizeBilinear(batchedTensor, [112, 112]);
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

        // This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.

        // Run predict() using trained VGG16 model
        const vggBoxesData = [];
        const vggClasses = []; // class codes
        const vggKlasses = []; // class names
        const vggScores = []; // class scores
       
        await vggModel.executeAsync(vggInput).then((res) => {
        // await vggModel.predict(vggInput).then((res) => {

          // Check the prediction results
          if (Array.isArray(res)) {
            res.forEach((item, i) => {
              item.data().then(data => {
                // console.log(`[VGG16] ${i}th data:`, data);
                // console.log("[VGG16]", i, "th data: ", data);
              });
            });
          } else {
            // If res is not an array, it's a tensor
            res.data().then(data => {
              // '[VGG16] data:', { '0': 0, '1': 0, '2': 1, '3': 0, '4': 0, '5': 1, '6': 0 }
              // console.log("[VGG16] data:", data);
            });
          }

          // result:, [ [ 0, 0, 1, 0, 0, 1, 0 ] ]
          const pred = res.arraySync(); // Get predictions as an array          
          // console.log("[VGG16] model.predict() result:", pred);
          // const pred = await vggModel.predict(resizedTensorVGG16).array(); // Get predictions as an array
          if (pred) {
            // console.log("[VGG16] model.predict():", pred);

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
            // const resultStr = [];
            for (let i = 0; i < vggClasses.length; i++) {
              const value = vggClasses[i]; // Accessing the first prediction
              switch (value) {
                case 1:
                  vggKlasses.push("바이러스성출혈성패혈증");
                  break;
                case 2:
                  vggKlasses.push("림포시스티스병");
                  break;
                case 6:
                  vggKlasses.push("여윔병");
                  break;
                case 8:
                  vggKlasses.push("스쿠티카병");
                  break;
                case 11:
                  vggKlasses.push("연쇄구균증");
                  break;
                case 13:
                  vggKlasses.push("비브리오병");
                  break;
                case 19:
                  vggKlasses.push("에드워드병");
                  break;
                default:
                  break; // Handle cases where the value doesn't match any class
              }
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
              vggBoxesData.push([0, 0, 100, 200*(i+1)])
            }            

            // 'vggBoxesData:', [ [ 0, 0, 10, 20 ], [ 0, 0, 10, 40 ] ]
            // 'resultCode:', [ [ 6, 13 ] ]
            // 'resultStr:', [ '스쿠티카병', '비브리오병' ]
            console.log("[VGG16] boxesData:", vggBoxesData, ", scores:", vggScores, ", classes:", vggClasses, ", klasses:", vggKlasses); 

            // renderBoxes(ctx, config.threshold, vggBoxesData, resultCode, resultStr, [xRatio, yRatio]);
            // tf.dispose([pred, vggInput]);    
          }

          tf.dispose([res, vggInput]);
        }).catch((error) => {
          console.log("Error in executing model:", error);
        });
        // -----------------------------------
        // End of VGG16 model prediction
        // -----------------------------------
        
        ctx.clearRect(0, 0, ctx.width, ctx.height); // clean canvas, 캔버스를 초기화
        renderYoloBoxes(ctx, config.threshold, boxesData, scoresData, classesData, [xRatio, yRatio]);
        renderVggBoxes(ctx, config.threshold, vggBoxesData, vggScores, vggClasses, vggKlasses, [xRatio, yRatio]);
        ctx.flush(); // 화면에 실제로 반영되도록 함

        tf.dispose([res, input, transRes, boxes, scores, classes, nms]);
        tf.dispose([vggBoxesData, vggClasses, vggKlasses, vggScores]);  

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
    }

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
