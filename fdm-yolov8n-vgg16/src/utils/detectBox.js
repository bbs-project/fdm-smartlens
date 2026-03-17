import * as tf from "@tensorflow/tfjs";
import labels from "../utils/labels.json";
import vggLabels from "../utils/vgglabels.json";

const numClass = labels.length;
const VGG_CLASSES = [1, 2, 6, 8, 11, 13, 19]; // VGG16 이 예측하는 7 개 질병 클래스

/**
 * VGG 모델의 클래스 코드를 한글 질병명으로 변환
 */
function getVggClassName(code) {
  const classNameMap = {
    1: "바이러스성출혈성패혈증",
    2: "림포시스티스병",
    6: "여윔병",
    8: "스쿠티카병",
    11: "연쇄구균증",
    13: "비브리오병",
    19: "에드워드병"
  };
  return classNameMap[code] || `Unknown(${code})`;
}

/**
 * Detect boxes using YOLOv8 model
 */
export async function detectYoloBoxes(yoloModel, params) {
  const [input] = params;

  try {
    const res = await yoloModel.executeAsync(input);
    
    // transpose result from [1, 9, 8400] -> [1, 8400, 9]
    const tx_res = Array.isArray(res) ? res[0].transpose([0, 2, 1]) : res.transpose([0, 2, 1]);

    // Extract bounding boxes [x1, y1, width, height]
    const boxes1 = tf.tidy(() => {
      const x1 = tx_res.slice([0, 0, 0], [-1, -1, 1]);
      const y1 = tx_res.slice([0, 0, 1], [-1, -1, 1]);
      const width = tx_res.slice([0, 0, 2], [-1, -1, 1]);
      const height = tx_res.slice([0, 0, 3], [-1, -1, 1]);

      // Convert to [x1, y1, x2, y2]
      return tf.concat([x1, y1, tf.add(x1, width), tf.add(y1, height)], 2).squeeze();
    });

    // Extract class probabilities and get max
    const rawClasses = tx_res.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0);
    const scores1 = tf.max(rawClasses, 1);
    const classes1 = tf.argMax(rawClasses, 1);

    // Non-maximum suppression
    const maxNumber = 3;
    const iouThreshold = 0.45;
    const scoreThreshold = 0.7;
    const nms1 = await tf.image.nonMaxSuppressionAsync(boxes1, scores1, maxNumber, iouThreshold, scoreThreshold);

    // Gather results
    const boxes_data = boxes1.gather(nms1, 0).dataSync();
    const scores_data = scores1.gather(nms1, 0).dataSync();
    const classes_data = classes1.gather(nms1, 0).dataSync();
    const num_detections = scores_data.length;

    // Cleanup
    tf.dispose([res, tx_res, boxes1, scores1, classes1, nms1]);

    return [num_detections, boxes_data, scores_data, classes_data];
  } catch (error) {
    console.error("[YOLO] Detection error:", error);
    return [0, [], [], []];
  }
}

/**
 * Detect prediction boxes using VGG16 model
 */
export async function detectVggBoxes(vggModel, image) {
  const boxesData = [];
  const classes = [];
  const klasses = [];
  const scores = [];
  const outputs = [];

  try {
    // 1. Add batch dimension: [640, 640, 3] -> [1, 640, 640, 3]
    const batchTensor = tf.expandDims(image, 0);
    
    // 2. Resize to 112x112 (VGG16 input size)
    const input = tf.image.resizeBilinear(batchTensor, [112, 112]);

    const res = await vggModel.executeAsync(input);
    
    // Get predictions as array
    const pred = res.arraySync();
    
    if (pred && pred.length > 0) {
      // Round predictions to binary (0 or 1)
      const result = pred[0].map(value => Math.round(value));
      
      // Check if any disease detected
      const detectedIndices = result.map((value, index) => value === 1 ? index : -1).filter(i => i !== -1);
      
      if (detectedIndices.length > 0) {
        // Get softmax probabilities for confidence scores
        const softmaxRes = tf.softmax(res);
        const probs = softmaxRes.arraySync()[0];
        
        detectedIndices.forEach((idx) => {
          const code = VGG_CLASSES[idx];
          if (code) {
            const klass = getVggClassName(code);
            const confidence = probs[idx] || 0.0;
            
            classes.push(code);
            klasses.push(klass);
            scores.push(confidence);
            outputs.push({ code, name: klass, confidence });
            
            // Create dummy bounding box (VGG doesn't provide location)
            boxesData.push([0, 0, 100, 200 * (classes.length)]);
          }
        });
        
        tf.dispose([softmaxRes]);
        
        console.log("[VGG] Detections:", outputs);
      }
    }

    tf.dispose([res, input, batchTensor]);
  } catch (error) {
    console.error("[VGG] Detection error:", error);
  }

  return [boxesData, classes, klasses, scores, outputs];
}
