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

// 모델 경로 설정: fdm-yolov8n
const modelJson = require("../../assets/model/fdm-yolov8n/model.json");
const modelWeights = [
  require("../../assets/model/fdm-yolov8n/group1-shard1of3.bin"),
  require("../../assets/model/fdm-yolov8n/group1-shard2of3.bin"),
  require("../../assets/model/fdm-yolov8n/group1-shard3of3.bin"),
];


// 모델 URI 생성: fdm-yolov8n
export const modelURI = bundleResourceIO(modelJson, modelWeights);

// 모델 경로 설정: fdm-vgg16
const modelJsonVgg16 = require("../../assets/model/fdm-vgg16/model.json");
const modelWeightsVgg16 = [
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
export const modelURIVgg16 = bundleResourceIO(modelJsonVgg16, modelWeightsVgg16);