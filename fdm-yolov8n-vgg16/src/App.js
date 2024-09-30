import React, { useEffect, useState } from "react";
import { Text, TouchableOpacity, View } from "react-native";
import { Camera } from "expo-camera";
import { StatusBar } from "expo-status-bar";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { yoloModelURI } from "./modelHandler";
import { vggModelURI } from "./modelHandler";
import CameraView from "./CameraView";

const App = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState("back");
  const [yoloModel, setYoloModel] = useState(null);
  const [vggModel, setVggModel] = useState(null);
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [inputTensor, setInputTensor] = useState([]);

  // model configuration
  const configurations = { threshold: 0.25 };

  useEffect(() => {
    (async () => {
      try  {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === "granted"); // 카메라 승인 상태 허용으로 변경
      } catch (error) {
        console.log(error)
      }

      // Load the models if tensorflow is ready
      tf.ready().then(async () => {
        // Load YOLOv8 model
        const yolov8 = await tf.loadGraphModel(yoloModelURI, {
          onProgress: (fractions) => {
            // Set loading progress
            setLoading({ loading: true, progress: fractions });
          },
        }); 

        // Load VGG16 model
        const vgg16 = await tf.loadGraphModel(vggModelURI, { 
          onProgress: (fractions) => {
            // Set loading progress
            setLoading({ loading: true, progress: fractions }); 
          },
        }); 

        // Warming up the model to reduce the latency of the first inference
        // tf. ones -> 모든 요소가 1인 텐서 생성, 모델의 첫 번째 입력 텐서의 모양 반환(dummy)
        // dummyInput's shape: [1, 3, 640, 640]
        const dummyInput = tf.ones(yolov8.inputs[0].shape);
        console.log("dummyInput's shape:", yolov8.inputs[0].shape)
        
        // Execute the model with dummy input tensor
        // await yolov8.executeAsync(dummyInput);
        // yolov8.execute(dummyInput);

        // Release the tensor from memory
        tf.dispose(dummyInput);

        // set state(모델의 성공상태 Load)
        setInputTensor(yolov8.inputs[0].shape); // 실제로 Load된 모델의 inputTensor 상태 업데이트
        setYoloModel(yolov8); // 실제로 Load 된 모델의 상태 업데이트
        setVggModel(vgg16);
        setLoading({ loading: false, progress: 1 }); // loading이 끝났으므로 loading 상태를 false + progress의 상태를 완료된 상태인 1로 지정
      });
    })();
  }, []);

  return ( // App에 실제로 보여줄 화면 컴포넌트들
    <View className="flex-1 items-center justify-center bg-white">
      {hasPermission ? ( // 카메라 권환 확인
        <>
          {loading.loading ? ( // loading 중이면 (loading 상태가 true) 출력
            <Text className="text-lg">Loading model... {(loading.progress * 100).toFixed(2)}%</Text>
          ) : ( // loading 상태가 false 출력
            <View className="flex-1 w-full h-full">
              <View className="flex-1 w-full h-full items-center justify-center">
                <CameraView // 각각의 속성 값을 전달
                  type={type}
                  yoloModel={yoloModel}
                  vggModel={vggModel}
                  inputTensorSize={inputTensor}
                  config={configurations}
                >
                  <View className="absolute left-0 top-0 w-full h-full flex justify-end items-center bg-transparent z-20">
                    <TouchableOpacity // 카메라 전환 버튼 버튼을 누르면 type의 상태에 따라 front or back으로 변경
                      className="flex flex-row items-center bg-transparent border-2 border-white p-3 mb-10 rounded-lg"
                      onPress={() => setType((current) => (current === "back" ? "front" : "back"))} 
                    >
                      <MaterialCommunityIcons
                        className="mx-2" name="camera-flip" size={30} color="white"
                      />
                      <Text className="mx-2 text-white text-lg font-semibold">Flip Camera</Text>
                    </TouchableOpacity>
                  </View>
                </CameraView>
              </View>
            </View>
          )}
        </>
      ) : (
        <View>
          <Text className="text-lg">Permission not granted!</Text>
        </View>
      )}
      <StatusBar style="auto" />
    </View>
  );
};

export default App;
