import * as tf from "@tensorflow/tfjs";

/** 각각 파라미터에 대한 설명
 * Preprocess image / frame before forwarded into the model
 * @param {tf.Tensor} img
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
export const preprocess = (img, modelWidth, modelHeight) => {
  // 패딩된 이미지의 비율을 저장하는 데 사용, 경계 상자를 원래 이미지의 크기를 변환할 때 필요
  let xRatio, yRatio; // ratios for boxes 

  // .tidy()는 내부에서 생성된 모든 텐서를 자동으로 메모리에서 해제한다.(메모리 누수 방지)
  const input = tf.tidy(() => { 
    // img is already a tensor, 이미지가 텐서로 이미 변환되어 있음
    // const img2 = tf.browser.fromPixels(img); // convert image to tensor, 이미지를 텐서로 변환
    // padding image to square => [n, m] to [n, n], n > m
    const [h, w] = img.shape.slice(0, 2); // get source width and height, slice(0,2) -> 텐서의 첫 두 차원을 가져오는데 이는 height와 width 이다.
    const maxSize = Math.max(w, h); // get max size, width 와 height 중 더 큰 size를 계산
    const imgPadded = img.pad([ // 이미지에 패딩 추가, 입력 이미지를 정사각형으로 패딩하여 모델의 입력 크기에 맞춤(이미지를 정사각형화하여 모델에 입력할 수 있도록)
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio, 예측 결과를 원래 이미지 크기에 맞추기 위해 사용
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame, 패딩된 이미지를 modelWidth, modelHeight에 맞게 resize
      .div(255.0) // normalize, 정규화
      .expandDims(0); // add batch, 텐서의 형태를 [batchSize, height, width, channels]로 변환
  });

  return [input, xRatio, yRatio];
};
