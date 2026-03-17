import labels from "./labels.json";
import vgglabels from "./vgglabels.json";
import { Colors } from "../utils/utils";

/**
 * Render prediction boxes
 * @param {Expo2DContext} ctx - Expo 2D context
 * @param {number} threshold - Score threshold
 * @param {number} num_detections - Number of detections
 * @param {Array} boxes_data - Bounding boxes array
 * @param {Array} scores_data - Scores array
 * @param {Array} classes_data - Classes array
 * @param {Array} ratios - Box ratios [xRatio, yRatio]
 * @param {boolean} flipX - Whether to flip horizontally
 */
export const renderBoxes = async (
  ctx,
  threshold,
  num_detections,
  boxes_data,
  scores_data,
  classes_data,
  ratios,
  flipX = true
) => {
  if (!ctx || !ctx.flush) {
    console.error("[renderBoxes] ctx is not an Expo 2D context");
    return;
  }

  const [xratio, yratio] = ratios;
  const font = `${Math.max(Math.round(Math.max(ctx.width, ctx.height) / 40), 14)}pt sans-serif`;
  ctx.font = font;
  ctx.textBaseline = "top";

  const colors = new Colors();

  for (let i = 0; i < num_detections; ++i) {
    if (scores_data[i] > threshold) {
      const code = classes_data[i];
      const klass = labels[code] || `Unknown(${code})`;
      const color = colors.get(code);
      const score = (scores_data[i] * 100).toFixed(1);

      let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
      let width = x2 - x1;
      let height = y2 - y1;

      // Scale coordinates
      y1 = y1 * 2;
      height = height * 2;

      // Draw bounding box
      const color3 = colors.get(3); // Orange Red
      ctx.strokeStyle = color3;
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, width, height);

      console.log("[YOLO] Box:", { x: x1, y: y1, width, height, klass, score });

      // Draw label background
      ctx.fillStyle = color;
      const textWidth = ctx.measureText(`${klass} (${score}%)`).width;
      const textHeight = parseInt(font, 10);
      const yText = y1 - (textHeight + 2);
      
      ctx.fillRect(x1 - 1, yText < 0 ? 0 : yText, textWidth + 2, textHeight + 2);

      // Draw label text
      ctx.fillStyle = "#000000";
      ctx.fillText(`${klass} (${score}%)`, x1 - 1, yText < 0 ? 0 : yText, width);
    }
  }
};

/**
 * Render VGG16 disease detection results
 */
export const renderVggBoxes = (ctx, vggOutputs, yOffset = 30) => {
  if (!ctx || !ctx.flush || !vggOutputs || vggOutputs.length === 0) return;

  const font = "16pt sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(0, 0, 0, 0.8)";

  vggOutputs.forEach((output, index) => {
    const text = `${output.name} (${(output.confidence * 100).toFixed(1)}%)`;
    const x = 10;
    const y = yOffset + (index * 25);
    ctx.fillText(text, x, y);
  });
};
