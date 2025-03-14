import * as tf from "@tensorflow/tfjs";
import labels from "../utils/labels.json";
// Number of classes predicted by the YOLOv8 model
const numClass = labels.length;


// Detect boxes using YOLOv8 model
export async function detectYoloBoxes(yoloModel, params) {
  const [input] = params;  

  const res = await yoloModel.executeAsync(input); // output shape: [1, 9, 8400]
  // const res = yoloModel.predict(input);
  // const res = await yoloModel.executeAsync({ 'Identity': input }); // output shape: [1, 3, 640, 640]
  // console.log("[YOLO] res:", res);

  // res shpae is [1, 3, 640, 640]
  // extract boxes, scores, and classes from the model output
  // Assuming res has shape [1, 3, 640, 640]
  //const [batch, channels, height, width] = res.shape;
  //console.log("[YOLO] res.shape:", res.shape);


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
      // console.log('[YOLO] single data:', data);
      // print number of elements in data
      // console.log("data.length:", data.length);
    });
  }

  // transpose result from [1, 9, 8400] -> [1, 8400, 9] 
  let tx_res;
  if (Array.isArray(res)) {
    tx_res = res[0].transpose([0, 2, 1]);
  } else {
    tx_res = res.transpose([0, 2, 1]);
  }  
    
  const boxes1 = tf.tidy(() => {
    const x1 = tx_res.slice([0, 0, 0], [-1, -1, 1]); // get x1
    const y1 = tx_res.slice([0, 0, 1], [-1, -1, 1]); // get y1
    const width = tx_res.slice([0, 0, 2], [-1, -1, 1]); // get width
    const height = tx_res.slice([0, 0, 3], [-1, -1, 1]); // get height

    // for (let i = 0; i < width.length; i++) {
    //   console.log("- width:", width[i], ", height:", height[i], ", x1:", x1[i], ", y1:", y1[i], ", x_center:", x_center[i], ", y_center:", y_center[i]);
    // }

    // [x1, y1, x2, y2] = [x1, y1, x1 + width, y1 + height]
    // axis=2: concatenate along the last axis
    return tf.concat([x1, y1, tf.add(x1, width), tf.add(y1, height), ], 2).squeeze();
  });
  // console.log("boxes1.shape:", boxes1.shape);
  // console.log("boxes1:", boxes1.dataSync());

  // const rawScores = tx_res.slice([0, 0, 4], [-1, -1, 1]).squeeze(0);
  // const class_probs = tx_res.slice([0, 0, 5], [-1, -1, numClass]).squeeze();
  // const max_class_prob = class_probs.max(1);
  // const scores1 = tf.mul(confidence.transpose([0, -1]).squeeze(), max_class_prob);
  // const classes1 = class_probs.argMax(1);

  const rawClasses = tx_res.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0);
  const scores1 = tf.max(rawClasses, 1);
  const classes1 = tf.argMax(rawClasses, 1);
  // for (let i = 4; i < rawClasses.length; i++) {
  //   if (rawClasses[i] > scores1[0]) {
  //     scores1[0] = rawClasses[i];
  //     classes1[0] = i - 4; // Adjust index to get class index
  //   }
  // } 

  // const [scores1, classes1] = tf.tidy(() => {
  //   // class scores
  //   const rawScores = tx_res.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
  //   return [rawScores.max(1), rawScores.argMax(1)];
  // }); // get max scores and classes index

  const maxNumber = 3; // Maximum number of boxes to detect : 1
  const iouThreshold = 0.45; // Intersection over Union threshold : 0.45
  const scoreThreshold = 0.7; // Score threshold : 0.5
  const nms1 = await tf.image.nonMaxSuppressionAsync(boxes1, scores1, maxNumber, iouThreshold, scoreThreshold); // NMS to filter boxes

  const boxes_data = boxes1.gather(nms1, 0).dataSync(); // indexing boxes by nms index
  const scores_data = scores1.gather(nms1, 0).dataSync(); // indexing scores by nms index
  const classes_data = classes1.gather(nms1, 0).dataSync(); // indexing classes by nms index

  const num_detections = scores_data.length;
  //console.log("num_detections:", scores_data.length); // number of detections, e.g., 2
  //console.log("classes_data:", classes_data); // class index of the detections, e.g., { '0': 3, '1': 3 }
  //console.log("scores_data:", scores_data);

  // boxes_data: [x1, y1, x2, y2]
  // e.g 2 boxes found
  //  { '0': 276.0263671875, '1': 415.5894775390625, '2': 658.13037109375, '3': 575.891845703125,
  //    '4': -3.3436279296875, '5': 416.6234130859375, '6': 339.51361083984375, '7': 587.1741333007812 }
  // console.log("boxes_data:", boxes_data);

  //   // ## 증상명
  //   // * 01 : 출혈 (Bleeding)
  //   // * 02 : 궤양 (Corrosion)
  //   // * 03 : 부식 (Tumor)
  //   // * 04 : 종양 (Ulcer)
  //   // * 05 : 안구증상 (eyesSymptom)
  //   const klass = labels[classes_data[i]];
  //   //klasses.push(klass);
  // }


  // -----------------------------------------------------------------------------------
  // End of Detection
  // -----------------------------------------------------------------------------------

  tf.dispose([res, tx_res, boxes1, scores1, classes1, nms1]);

  return [ num_detections, boxes_data, scores_data, classes_data ];   
}

// Detect prediction boxes using VGG16 model
// - image's shape : [640, 640, 3]
export async function detectVggBoxes(vggModel, image) {
  // 1. Add a batch dimension to the image : making it [1, 640, 640, 3]
  const batchTensor = tf.expandDims(image, 0);

  // 2. Resize the image to 112x112, and name it as input with shape: [1, 112, 112, 3]
  const input = tf.image.resizeBilinear(batchTensor, [112, 112]);
  
  const boxesData = [];
  const classes = []; // class codes
  const klasses = []; // class names
  const scores = []; // class scores
  const outputs = []; // { code: value, name: klass }

  await vggModel.executeAsync(input).then((res) => {
    
    // Check the prediction results
    if (Array.isArray(res)) {
      res.forEach((item, i) => {
        item.data().then(data => {
          // console.log(`[VGG16] ${i}th data:`, data);
          // console.log("[VGG16]", i, "th data: ", data);
        });
      });
    } else {
      // If rs is not an array, it's a tensor
      res.data().then(data => {
        // [VGG] data: [0, 0, 8.73132399714649e-11, 0, 0, 1.0819256897390123e-25, 0]
        // console.log("[VGG] data:", data);
      });
    }

    // res: [ [ 0, 0, 1, 0, 0, 1, 0 ] ]
    const pred = res.arraySync(); // Get predictions as an array          

    if (pred) {
      // console.log("[VGG16] model.predict():", pred);
      // 'result' is an array of arrays containing binary predictions (0 or 1)
      const result = pred.map(idx => 
        idx.map(value => Math.round(value)) // Round and convert to integers
      );
      // console.log("[VGG] result:", result);
      
      // check if result has any value with 1
      const hasValue = result.some(value => value.includes(1));
      if (!hasValue) {
        // console.log("[VGG] No disease detected");
        return [ boxesData, classes, klasses, scores, outputs ];
      } 

      // 'classes' is an array of corresponding class labels
      // (disease1, disease2, disease6, disease8, disease11, disease13,disease19)
      // * 01 : 바이러스성출혈성패혈증
      // * 02 : 림포시스티스병
      // * 06 : 여윔병
      // * 08 : 스쿠티카병
      // * 11 : 연쇄구균증
      // * 13 : 비브리오병
      // * 19 : 에드워드병
      const classes = [1, 2, 6, 8, 11, 13, 19];
            
      // Convert prediction results to codes
      // const resultCode = result.map(value => {
      const resultCodes = result.map(value => {
        // console.log("value.length:", value.length, ", ", value);
        const codes = [];
        for (let i = 0; i < value.length; i++) {
          if (value[i] === 1) {
            const code = classes[i];
            codes.push(code);
            classes.push(code); 
            const klass = getVggClassName(code);
            klasses.push(klass); 
            scores.push(1.0); // Dummy score
            outputs.push({ code: code, name: klass });            
          }
        }

        // print outputs
        console.log("[VGG] ", outputs);
        
        return codes;
      });
      // console.log("numCodes", resultCodes.length, ", ", resultCodes);

      // push each of the result codes to the classes array
      // resultCodes[0].forEach(code => {
      //   //console.log("code:", code);
      //   classes.push(code);
      //   const klass = getVggClassName(code);
      //   klasses.push(klass);        
      //   scores.push(1.0); // Dummy score
      //   outputs.push({ code: code, name: klass });
      // });

      // vggClasses.push(resultCodes);
      // Convert prediction codes to strings (for the first prediction only)
      // const outputs = [];
      // console.log(classes.length);

      // resultCode: Array of code arrays
      // resultStr: Array of disease names for the first prediction
      //console.log("    - resultCode:", resultCode, ", resultStr:", resultStr);
      const numRes = classes.length;
      // create array of bounding box data for the predictions
      // const vggBoxesData = Array(numRes);
      // const vggBoxesData = []; // Dummy bounding box data
      for (let i = 0; i < numRes; i++) {
        // create vggBoxesData for the prediction and push it to the array
        boxesData.push([0, 0, 100, 200 * (i + 1)]);
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

    tf.dispose([res, input]);
  }).catch((error) => {
    console.log("Error in executing model:", error);
  });

  return [ boxesData, classes, klasses, scores, outputs ];
}

/**
 * VGG 모델의 클래스 코드를 한글 질병명으로 변환
 * @param {number} code - 질병 코드 (1, 2, 6, 8, 11, 13, 19)
 * @returns {string} - 해당하는 한글 질병명
 */
function getVggClassName(code) {
  switch (code) {
    case 1:
      return "바이러스성출혈성패혈증";
    case 2:
      return "림포시스티스병";
    case 6:
      return "여윔병";
    case 8:
      return "스쿠티카병";
    case 11:
      return "연쇄구균증";
    case 13:
      return "비브리오병";
    case 19:
      return "에드워드병";
    default:
      return ""; // 매칭되지 않는 코드의 경우 빈 문자열 반환
  }
}