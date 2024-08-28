package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener,
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        val compatList = CompatibilityList()

        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun detect(frame: Bitmap) {
        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, 224, 224, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE).apply { load(resizedBitmap) }
        val processedImage = imageProcessor.process(tensorImage)

        val output = TensorBuffer.createFixedSize(intArrayOf(1, 1000), OUTPUT_IMAGE_TYPE)
        interpreter.run(processedImage.buffer, output.buffer)

        val confidences = output.floatArray
        val maxIdx = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val maxConfidence = confidences[maxIdx]

        val detectedClassName = labels.getOrNull(maxIdx) ?: "Unknown"
        val boundingBox = BoundingBox(
            x1 = 0f, y1 = 0f, x2 = 1f, y2 = 1f,
            cx = 0.5f, cy = 0.5f, w = 1f, h = 1f,
            cnf = maxConfidence, cls = maxIdx, clsName = detectedClassName
        )

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (maxConfidence > CONFIDENCE_THRESHOLD) {
            detectorListener.onDetect(listOf(boundingBox), inferenceTime)
        } else {
            detectorListener.onEmptyDetect()
        }
    }

    fun close() {
        interpreter.close()
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
    }
}
