import { useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { preprocess } from "../utils/preprocess";
import { detectYoloBoxes } from "../utils/detectBox";
import { renderBoxes } from "../utils/renderBox";

const CameraView = ({ type, yoloModel, vggModel, inputTensorSize, config, children }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameRef = useRef(null);
  const [modelHeight, modelWidth] = inputTensorSize.slice(1, 3);
  const typesMapper = { back: "environment", front: "user" };

  /**
   * Function to detect every frame loaded from webcam in video tag.
   */
  const detectFrame = async () => {
    tf.engine().startScope();
    const rawInput = tf.browser.fromPixels(videoRef.current);
    const [input, xRatio, yRatio] = preprocess(rawInput, modelWidth, modelHeight);
    const ctx = canvasRef.current.getContext("2d");

    const [numDetections, boxesData, scoresData, classesData] = await detectYoloBoxes(yoloModel, [input]);

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (numDetections > 0) {
      renderBoxes(ctx, config.threshold, numDetections, boxesData, scoresData, classesData, [xRatio, yRatio]);
    }

    tf.dispose([input, rawInput]);

    frameRef.current = requestAnimationFrame(detectFrame);
    tf.engine().endScope();
  };

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: typesMapper[type],
          },
        })
        .then((stream) => {
          if (videoRef.current.srcObject) {
            cancelAnimationFrame(frameRef.current);
            frameRef.current = null;
            videoRef.current.srcObject.getTracks().forEach((track) => {
              track.stop();
            });
            videoRef.current.srcObject = null;
          }

          videoRef.current.srcObject = stream;
        });
    } else alert("Can't open Webcam!");
  }, [type]);

  return (
    <>
      <div className="relative">
        <video
          className="w-full max-w-screen-md max-h-fit rounded-lg"
          autoPlay
          muted
          onPlay={() => detectFrame()}
          ref={videoRef}
        />
        <canvas
          className="absolute top-0 left-0 w-full h-full"
          width={modelWidth}
          height={modelHeight}
          ref={canvasRef}
        />
        {children}
      </div>
    </>
  );
};

export default CameraView;
