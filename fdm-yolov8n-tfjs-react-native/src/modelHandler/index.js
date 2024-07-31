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


// 모델 URI 생성
export const modelURI = bundleResourceIO(modelJson, modelWeights);