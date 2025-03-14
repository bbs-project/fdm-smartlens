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
import { detectYoloBoxes } from "../utils/detectBox";
import { detectVggBoxes } from "../utils/detectBox";
import { renderBoxes } from "../utils/renderBox";

// import { cleanupTensors } from "../utils/cleanupTensors";

const TensorCamera = cameraWithTensors(Camera);

const CameraView = ({ type, yoloModel, vggModel, inputTensorSize: inputShape, config, children }) => {
  // console.log("CameraView enabled: type=", type, ", inputTensorSize=", inputTensorSize, ", config=", config);
  
  const [ctx, setCTX] = useState(null);
  const [vggOutputs, setVggOutputs] = useState([]);
  const cameraType = { back: CameraType.back, front: CameraType.front };

  // Throttling function : detectFrame 함수가 limit 간격(ms)으로만 실행되도록 제한
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

    // Detect an image from camera frame
    const detectFrame = async () => {
      // console.log("Detected a frame from camera.");
      
      // inputShape: [1, 3, 640, 640]
      // console.log("inputShape:", inputShape);

      tf.engine().startScope();

      const image = images.next().value; // tf.Tensor3D ([640, 640, 3])
      if (image) {        
        // Transpose image from [640, 640, 3] to [1, 3, 640, 640] = [batchSize, channels, height, width]
        const transposedTensor = image.transpose([2, 0, 1]); 
        // inputShape = [batchSize, height, width, channels]
        // but transposedTensor has [3, 640, 640]
        const [modelHeight, modelWidth] = inputShape.slice(1, 3); // get model height(1) and width(2)
        const [input, xRatio, yRatio] = preprocess(transposedTensor, modelWidth, modelHeight); // preprocess image
 
        // -----------------------------------        
        const [numDetections, boxesData, scoresData, classesData] = await detectYoloBoxes(yoloModel,[input]);       
        // console.log("[YOLO] numDetections:", numDetections);

        if (numDetections > 0) {
          const [vggBoxesData, vggClasses, vggKlasses, vggScores, vOutputs] = await detectVggBoxes(vggModel, image);
          // vggBoxesData, vggClasses, vggKlasses, vggScores are not used in this version
          // console.log("vggBoxesData: ", vggBoxesData);
          // console.log("vggClasses: ", vggClasses);
          // console.log("vggKlasses: ", vggKlasses);
          // console.log("vggScores: ", vggScores);
          // console.log("vOutputs: ", vOutputs);
  
          ctx.clearRect(0, 0, ctx.width, ctx.height); // clean canvas, 캔버스를 초기화

          // This makes the vggOutputs available in the View
          setVggOutputs(vOutputs);
          // -----------------------------------
          
          tf.dispose([vggBoxesData, vggClasses, vggKlasses, vggScores, vggOutputs]);  

          // let xrate = ctx.width / 640; // modelWidth
          // let yrate = ctx.height / 640; // modelHeight          

          // render YoloBoxes
          renderBoxes(ctx, config.threshold, numDetections, boxesData, scoresData, classesData, [xRatio, yRatio]);
          // vggBoxes are rendered in the View
        }

        ctx.flush(); // 화면에 실제로 반영되도록 함
        
        // clear 
        tf.dispose([input]);
        tf.dispose([numDetections, boxesData, scoresData, classesData]);
        // tf.dispose([vggBoxesData, vggClasses, vggKlasses, vggScores, vggOutputs]);  

      } else {
        // console.log("No image tensor found.");
      }
    
      // // console.log("Requesting animation frame.");
      setTimeout(() => {
        requestAnimationFrame(detectFrame);
      }, 1000); // Adjust the limit as needed (e.g., 5000ms)

      tf.engine().endScope();      
    }
   
    // console.log("Requesting animation frame.");
    // Wrap detectFrame with the throttling function
    // detectFrame();
    // let throttledDetectFrame = throttle(detectFrame, 5000); // Adjust the limit as needed (e.g., 100ms)

    // Start the detection loop
    // console.log("Requesting animation frame.");
    requestAnimationFrame(detectFrame);
  };

  return (
    <>
      {ctx && (
        <TensorCamera
          style={{ width: "100%", height: "100%", zIndex: 0 }}
          type={cameraType[type]}
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
    left: '50%',
    transform: [{ translateX: -185 }],
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
