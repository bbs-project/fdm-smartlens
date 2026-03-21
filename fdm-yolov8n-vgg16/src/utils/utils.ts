import * as tf from "@tensorflow/tfjs";

/**
 * Preprocess image/frame before forwarding into the model
 * @param {tf.Tensor} img - Input image tensor
 * @param {number} modelWidth - Target model width
 * @param {number} modelHeight - Target model height
 * @returns {[tf.Tensor, number, number]} - Input tensor, xRatio, yRatio
 */
export const preprocess = (img, modelWidth, modelHeight) => {
  let xRatio, yRatio;

  const input = tf.tidy(() => {
    const [h, w] = img.shape.slice(0, 2);
    const maxSize = Math.max(w, h);
    
    // Pad image to square
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    // Calculate ratios for box scaling
    xRatio = maxSize / w;
    yRatio = maxSize / w;

    // Resize, normalize, and add batch dimension
    return tf.image
      .resizeBilinear(imgPadded, [modelHeight, modelWidth])
      .div(255.0)
      .expandDims(0);
  });

  return [input, xRatio, yRatio];
};

/**
 * Cleanup tensors to prevent memory leaks
 * @param {Array} tensors - Array of tensors to dispose
 */
export const cleanupTensors = (...tensors) => {
  tensors.forEach(tensor => {
    if (tensor && typeof tensor.dispose === 'function') {
      tensor.dispose();
    }
  });
};

/**
 * Check if object is a tensor
 * @param {any} obj - Object to check
 * @returns {boolean}
 */
export const isTensor = (obj) => {
  return obj && typeof obj.shape === 'object' && typeof obj.dispose === 'function';
};
