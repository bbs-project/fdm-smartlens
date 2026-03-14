/**
 * loadModel for web.
 * Load model via static url. Copying model to static folder via copy-webpack-plugin
 * see webpack.config.js
 */
export const yoloModelURI = `${window.location.origin}/model/fdm-yolov8n/model.json`;
export const vggModelURI = `${window.location.origin}/model/fdm-vgg16/model.json`;
