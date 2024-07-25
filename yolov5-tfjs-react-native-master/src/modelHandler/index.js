import { bundleResourceIO } from "@tensorflow/tfjs-react-native";

// 모델 경로 설정
const modelJson = require("../../assets/model/yolov5n-320/model.json");
const modelWeights = [
  require("../../assets/model/yolov5n-320/group1-shard1of2.bin"),
  require("../../assets/model/yolov5n-320/group1-shard2of2.bin"),
];

// 모델 URI 생성
export const modelURI = bundleResourceIO(modelJson, modelWeights);