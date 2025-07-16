package kr.re.etri.fdm.smartlens

object Constants {
    // Models and labels for YOLOv8 model
    // const val MODEL_PATH = "fdm_yolov8n.tflite" // non-quantized model
    const val MODEL_YOLOV8_PATH = "fdm_yolov8n_float16.tflite" // float16 quantized model
    const val LABELS_YOLOV8_PATH = "labels_yolov8.txt"

    // Models and labels for VGG16 model
    // const val MODEL_VGG16_PATH = "fdm_vgg16.tflite" // non-quantized model
    // const val MODEL_VGG16_PATH = "fdm_vgg16_dynrange.tflite" // dynamic range quantized model
    const val MODEL_VGG16_PATH = "fdm_vgg16_float16.tflite" // float16 quantized model
    const val LABELS_VGG16_PATH = "labels_vgg16.txt"

    const val DEFAULT_TEXT_SIZE = 50F
    const val BOX_STROKE_WIDTH = 8F
}
