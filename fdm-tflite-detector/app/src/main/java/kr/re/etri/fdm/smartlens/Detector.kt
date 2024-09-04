package kr.re.etri.fdm.smartlens

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
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder


class Detector(
    // 클래스 변수
    private val context: Context, // Android 컨텍스트
    private val modelYolov8Path: String, // YOLOv8 모델 파일 경로
    private val labelYolov8Path: String, // YOLOv8 레이블 파일 경로
    private val modelVgg16Path: String, // YOLOv8 모델 파일 경로
    private val labelVgg16Path: String, // YOLOv8 레이블 파일 경로
    private val detectorListener: DetectorListener, // 감지 결과를 받을 리스너
) {

    private var interpreter_Y: Interpreter // TensorFlow Lite 'Interpreter' 객체 저장
    private var interpreter_V: Interpreter // TensorFlow Lite 'Interpreter' 객체 저장
    private var labels_Y = mutableListOf<String>() // 레이블 저장할 'labels'
    private var labels_V = mutableListOf<String>() // 레이블 저장할 'labels'

    private var tensorWidth = 0 // 텐서 너비
    private var tensorHeight = 0 // 텐서 높이
    private var numChannel = 0 // 채널 수
    private var numElements = 0 // 요소 수

    private val imageProcessor = ImageProcessor.Builder() // 입력 이미지의 전처리 담당 객체
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION)) // 정규화
        .add(CastOp(INPUT_IMAGE_TYPE)) // 타입 변환
        .build()

    // VGG16: Input tensor shape: 1 x 112 x 112 x 3
    private val imageProcessor_V = ImageProcessor.Builder() // 입력 이미지의 전처리 담당 객체
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION)) // 정규화
        .add(CastOp(INPUT_IMAGE_TYPE)) // 타입 변환
        .add(ResizeOp(112, 112, ResizeOp.ResizeMethod.BILINEAR)) // 이미지 크기 조정
        //.add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)) // 이미지 크기 조정
        .build()

    init { // 초기화 블록
        val compatList = CompatibilityList() // 장치의 gpu 호환성 체크, 지원되는 경우 gpu 딜리게이트 추가

        val options = Interpreter.Options().apply{
            if(compatList.isDelegateSupportedOnThisDevice){
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(4)
            }
        }

        val model_Y = FileUtil.loadMappedFile(context, modelYolov8Path) // YOLOv8 model load
        val model_V = FileUtil.loadMappedFile(context, modelVgg16Path) // VGG16 model load
        // Interpreter는 input 과 output 의 형태를 알려주는 역할(임의 해석)
        interpreter_Y = Interpreter(model_Y, options)
        // Interpreter는 input 과 output 의 형태를 알려주는 역할(임의 해석)
        interpreter_V = Interpreter(model_V, options)

        val inputShape = interpreter_Y.getInputTensor(0)?.shape() // input tensor shape
        val outputShape = interpreter_Y.getOutputTensor(0)?.shape() // output tensor shape

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numChannel = outputShape[1]
            numElements = outputShape[2]
        }

        // Load labels for YOLOv8
        try {
            val inputStream: InputStream = context.assets.open(labelYolov8Path)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels_Y.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }

        // Load labels for VGG16
        try {
            val inputStream: InputStream = context.assets.open(labelVgg16Path)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels_V.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    // Define what to do when the App restarts
    fun restart(isGpu: Boolean) {
        interpreter_Y.close()
        interpreter_V.close()

        val options = if (isGpu) {
            val compatList = CompatibilityList()
            Interpreter.Options().apply{
                if(compatList.isDelegateSupportedOnThisDevice){
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    this.setNumThreads(4)
                }
            }
        } else {
            Interpreter.Options().apply{
                this.setNumThreads(4)
            }
        }

        // Load models and creates interpreter objects for the loaded models
        val model_Y = FileUtil.loadMappedFile(context, modelYolov8Path)
        interpreter_Y = Interpreter(model_Y, options)

        val model_V = FileUtil.loadMappedFile(context, modelVgg16Path)
        interpreter_V = Interpreter(model_V, options)
    }

    fun close() {
        interpreter_Y.close()
        interpreter_V.close()
    }

    fun detect(frame: Bitmap) {
        // 입력 유효성 검사( 초기화 확인 )
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        // 현재 시간 기록( 추론이 얼마나 걸리는지 측정 )
        var inferenceTime = SystemClock.uptimeMillis()

        // 입력 이미지를 모델이 요구하는 크기로 조정
        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        // 이미지를 모델의 입력 형식으로 변환하고, 전처리 수행
        val tensorImage = TensorImage(INPUT_IMAGE_TYPE) // 입력 이미지 타입 지정한 텐서 이미지 생성
        tensorImage.load(resizedBitmap) // 조정된 이미지를 'tensorImage' 객체에 로드
        // 이미지 정규화하고 타입을 변환(CastOp), 학습된 모델이 기대하는 형식으로 이미지 준비
        val processedImage = imageProcessor.process(tensorImage)
        // 전처리된 이미지 데이터를 버퍼로 가져옴
        val imageBuffer = processedImage.buffer

        // 모델에 전처리된 이미지를 입력으로 주고 추론 수행
        // 모델 출력 형식과 크기에 맞는 고정된 크기의 텐서 버퍼 생성( 3차원 배열, 1 x numChannel x num Elements )
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        // 전처리된 이미지 버퍼를 모델에 입력으로 제공, 모델 예측 결과를 output.buffer에 저장
        interpreter_Y.run(imageBuffer, output.buffer)

        // 추론 결과로부터 가장 신뢰도 높은 객체를 선택하고, 추론 시간을 측정
        val bestBoxes  = bestBox(output.floatArray) // 모델 예측 결과를 기반으로 신뢰도가 높은 검출 박스를 선택

        // 감지된 객체가 없는 경우 onEmptyDetect() 호출
        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        // Print the size of the output tensor
        //println("YOLO: Input tensor capacity: ${processedImage.buffer.capacity()}, size: ${processedImage.buffer.limit()}")
        //println("YOLO: Output tensor capacity: ${output.buffer.capacity()}")
        val inputTensor_Y = interpreter_Y.getInputTensor(0)
        val shape_Y = inputTensor_Y.shape()
        // println("YOLO: Input tensor shape: ${shape_Y[0]} x ${shape_Y[1]} x ${shape_Y[2]} x ${shape_Y[3]}")

        // -----------------------------------------------------------------------------------------
        // If we detected something from this image, run VGG16 interpreter for this image also
        // Resize the frame to 112x112
        val resizedBitmap_V = Bitmap.createScaledBitmap(frame, 112, 112, false)
        // Create a TensorImage from the resized bitmap
        val tensorImage_V = TensorImage(INPUT_IMAGE_TYPE).apply { load(resizedBitmap_V) }
        // Preprocess the tensor image using the imageProcessor
        val processedImage_V = imageProcessor_V.process(tensorImage_V)
        // Create a TensorBuffer with the same shape as the output tensor
        val output_V = TensorBuffer.createFixedSize(intArrayOf(1, 7), OUTPUT_IMAGE_TYPE)
        // val output_V = TensorBuffer.createFixedSize(intArrayOf(1, 112, 112, 3), DataType.UINT8)
        // val output_V = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)

        //println("VGG16: Input tensor capacity: ${processedImage_V.buffer.capacity()}, size: ${processedImage.buffer.limit()}")
        //println("VGG16: Output tensor capacity: ${output_V.buffer.capacity()}")
        val inputTensor_V = interpreter_V.getInputTensor(0)
        val shape_V = inputTensor_V.shape()
        println("VGG16: Input tensor shape: ${shape_V[0]} x ${shape_V[1]} x ${shape_V[2]} x ${shape_V[3]}")

        //interpreter_V.run(processedImage_V.buffer, output_V.buffer)
        interpreter_V.run(convertBitmapToByteBuffer(frame), output_V.buffer)

        val confidences = output_V.floatArray
        // println("VGG16: confidences: ${confidences.size}")
        print("VGG16: confidences=[ ")
        for (confidence in confidences) {
            print("$confidence ")
        }
        println("]")

        val maxIdx = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val maxConfidence = confidences[maxIdx]
        val detectedClassName = labels_V.getOrNull(maxIdx) ?: "Unknown"
        println("VGG16: maxIdx: $maxIdx, maxConfidence: $maxConfidence, detectedClassName: $detectedClassName")

        val boundingBox_V = BoundingBox(
            x1 = 0f, y1 = 0f, x2 = 1f, y2 = 1f,
            cx = 0.5f, cy = 0.5f, w = 1f, h = 1f,
            cnf = maxConfidence, cls = maxIdx, clsName = detectedClassName
        )

        val allBoxes: MutableList<BoundingBox> = ArrayList()
        if (bestBoxes != null) {
            allBoxes.addAll(bestBoxes)
        }
        allBoxes.add(boundingBox_V)
        // -----------------------------------------------------------------------------------------

        // 현재 시간에서 시작 시간을 빼서 추론 소요 시간 계산
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        // 감지된 객체가 있는 경우 감지된 박스 리스트와 추론 시간을 전달
        // detectorListener.onDetect(bestBoxes, inferenceTime)
        detectorListener.onDetect(allBoxes, inferenceTime)
    }

    private fun convertBitmapToByteBuffer(bp: Bitmap): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(java.lang.Float.BYTES * 112 * 112 * 3)
        // val imgData = ByteBuffer.allocateDirect(224 * 224 * 3)
        imgData.order(ByteOrder.nativeOrder())
        val bitmap = Bitmap.createScaledBitmap(bp, 112, 112, true)
        val intValues = IntArray(112 * 112)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Convert the image to floating point.
        var pixel = 0

        for (i in 0..111) {
            for (j in 0..111) {
                val `val` = intValues[pixel++]
                imgData.putFloat(((`val` shr 16) and 0xFF) / 255f)
                imgData.putFloat(((`val` shr 8) and 0xFF) / 255f)
                imgData.putFloat((`val` and 0xFF) / 255f)
            }
        }
        return imgData
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>() // 검출 객체 정보 저장 리스트 초기화

        // 모델 출력 배열에서 각 객체의 정보를 읽고 신뢰도 계산
        for (c in 0 until numElements) { // numElements까지 반복하여 객체 정보 순회
            var maxConf = CONFIDENCE_THRESHOLD // 현재 객체의 최대 신뢰도 저장 변수 ( 초기값은 미리 설정된 임계값 )
            var maxIdx = -1 // 최대 신뢰도 갖는 클래스의 인덱스 저장
            var j = 4 // cx, cy, w, h 총 4개
            var arrayIdx = c + numElements * j // 클래스 신뢰도 정보를 가리키는 인덱스 계산
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) { // 클래스 신뢰도가 더 높으면 저장
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            // 신뢰도가 높은 객체 필터링
            if (maxConf > CONFIDENCE_THRESHOLD) {// 신뢰도 낮은 객체 제외
                val clsName = labels_Y[maxIdx] // 해당 객체의 클래스 이름 가져옴
                val cx = array[c] // 0, 객체 중심 좌표 x
                val cy = array[c + numElements] // 1, 객체 중심 좌표 y
                val w = array[c + numElements * 2] // 너비
                val h = array[c + numElements * 3] // 높이
                // (x1, y1) , (x2, y2)는 객체의 좌상단과 우하단 좌표를 계산함
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        // 검출된 결과에 대해서 비최대 억제법(NMS)를 적용하여 중복 박스를 제거한다.
        val selectedBoxes = applyNMS(boundingBoxes)
        return selectedBoxes
    }

    // 비최대 억제법(Non-Maximum Suppression)을 적용하여 검출된 객체의 중복 제거, 신뢰도 높은 객체만 남김
    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList() // 검출된 박스들을 신뢰도에 따라 내림차순으로 정렬( 신뢰도가 높은 박스가 리스트 앞쪽에 오도록 정렬 )
        val selectedBoxes = mutableListOf<BoundingBox>() // NMS 적용 후 남겨진 최종적으로 선택된 박스를 저장할 리스트 초기화

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first() // first는 현재 가장 신뢰도가 높은 박스를 나타냄, 이는 sortedBoxes의 첫 번째 요소가 됨
            selectedBoxes.add(first) // 가장 신뢰도 높은 박스를 선택된 리스트 추가
            sortedBoxes.remove(first) // 선택된 박스를 후보 리스트에서 제거

            // 중복 박스 제거
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) { // sortedBoxes 순회
                val nextBox = iterator.next() // 현재 비교 대상이 되는 박스
                val iou = calculateIoU(first, nextBox) // 두 박스 간의 IoU를 계산하는 함수
                if (iou >= IOU_THRESHOLD) { // IOU_THRESHOLD를 넘으면 박스가 겹친다는 의미 -> remove를 통해 제거
                    iterator.remove()
                }
            }
        }

        return selectedBoxes // 중복된 박스 제거, 신뢰도 높은 박스만 return
    }

    // IoU 계산 ( IoU는 두 박스가 얼마나 겹치는지 측정 지표, 0~1 값을 가짐)
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        // 두 박스가 겹치는 부분의 좌상단 및 우하단 좌표 계산
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        // 두 박스 간의 겹치는 영역의 면적 계산
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1) // x2 - x1 -> 너비 , y2 - y1 -> 높이
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        // IoU 계산 및 반환
        return intersectionArea / (box1Area + box2Area - intersectionArea)
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
        private const val IOU_THRESHOLD = 0.5F
    }
}