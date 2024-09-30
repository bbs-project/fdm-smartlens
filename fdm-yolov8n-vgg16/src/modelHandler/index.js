import { bundleResourceIO } from "@tensorflow/tfjs-react-native";

// 모델 경로 설정: yolov5n-320
// inputTensorSize=[1, 320, 320, 3]
//const modelJson = require("../../assets/model/yolov5n-320/model.json");
//const modelWeights = [
//  require("../../assets/model/yolov5n-320/group1-shard1of2.bin"),
//  require("../../assets/model/yolov5n-320/group1-shard2of2.bin"),
//];

// 모델 경로 설정: yolov8n
// inputTensorSize=[1, 3, 640, 640]
//const modelJson = require("../../assets/model/yolov8n/model.json");
//const modelWeights = [
//  require("../../assets/model/yolov8n/group1-shard1of4.bin"),
//  require("../../assets/model/yolov8n/group1-shard2of4.bin"),
//  require("../../assets/model/yolov8n/group1-shard3of4.bin"),
//  require("../../assets/model/yolov8n/group1-shard4of4.bin"),
//];

//console.log(__dirname)
//console.log("Current working directory:", process.cwd());

// Set mode path: fdm-yolov8n
const yoloModelJson = require("../../assets/model/fdm-yolov8n/model.json");
const yoloModelWeights = [
  require("../../assets/model/fdm-yolov8n/group1-shard1of3.bin"),
  require("../../assets/model/fdm-yolov8n/group1-shard2of3.bin"),
  require("../../assets/model/fdm-yolov8n/group1-shard3of3.bin"),
];


// Create model URI: fdm-yolov8n
export const yoloModelURI = bundleResourceIO(yoloModelJson, yoloModelWeights);

// Set model path: fdm-vgg16
const vggModelJson = require("../../assets/model/fdm-vgg16/model.json");
const vggModelWeights = [
  require("../../assets/model/fdm-vgg16/group1-shard1of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard2of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard3of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard4of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard5of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard6of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard7of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard8of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard9of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard10of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard11of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard12of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard13of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard14of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard15of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard16of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard17of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard18of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard19of20.bin"),
  require("../../assets/model/fdm-vgg16/group1-shard20of20.bin"),
];


// 모델 URI 생성: fdm-vgg16
export const vggModelURI = bundleResourceIO(vggModelJson, vggModelWeights);