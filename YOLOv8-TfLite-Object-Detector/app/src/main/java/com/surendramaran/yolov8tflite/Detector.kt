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
    // 클래스 변수
    private val context: Context, // Android 컨텍스트
    private val modelPath: String, // 모델 파일 경로
    private val labelPath: String, // 레이블 파일 경로
    private val detectorListener: DetectorListener, // 감지 결과를 받을 리스너
) {

    private var interpreter: Interpreter // TensorFlow Lite 'Interpreter' 객체 저장
    private var labels = mutableListOf<String>() // 레이블 저장할 'labels'

    private var tensorWidth = 0 // 텐서 너비
    private var tensorHeight = 0 // 텐서 높이
    private var numChannel = 0 // 채널 수
    private var numElements = 0 // 요소 수

    private val imageProcessor = ImageProcessor.Builder() // 입력 이미지의 전처리 담당 객체
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION)) // 정규화
        .add(CastOp(INPUT_IMAGE_TYPE)) // 타입 변환
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

        val model = FileUtil.loadMappedFile(context, modelPath) // model load
        interpreter = Interpreter(model, options) // Interpreter는 input 과 output 의 형태를 알려주는 역할(임의 해석)

        val inputShape = interpreter.getInputTensor(0)?.shape() // input tensor shape
        val outputShape = interpreter.getOutputTensor(0)?.shape() // output tensor shape

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

    fun restart(isGpu: Boolean) {
        interpreter.close()

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

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun close() {
        interpreter.close()
    }

    fun detect(frame: Bitmap) {
        // 입력 유효성 검사( 초기화 확인 )
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis() // 현재 시간 기록( 추론이 얼마나 걸리는지 측정 )

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false) // 입력 이미지를 모델이 요구하는 크기로 조정

        // 이미지를 모델의 입력 형식으로 변환하고, 전처리 수행
        val tensorImage = TensorImage(INPUT_IMAGE_TYPE) // 입력 이미지 타입 지정한 텐서 이미지 생성
        tensorImage.load(resizedBitmap) // 조정된 이미지를 'tensorImage' 객체에 로드
        val processedImage = imageProcessor.process(tensorImage) // 이미지 정규화하고 타입을 변환(CastOp), 학습된 모델이 기대하는 형식으로 이미지 준비
        val imageBuffer = processedImage.buffer // 전처리된 이미지 데이터를 버퍼로 가져옴

        // 모델에 전처리된 이미지를 입력으로 주고 추론 수행
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE) // 모델 출력 형식과 크기에 맞는 고정된 크기의 텐서 버퍼 생성( 3차원 배열, 1 x numChannel x num Elements )
        interpreter.run(imageBuffer, output.buffer) // 전처리된 이미지 버퍼를 모델에 입력으로 제공, 모델 예측 결과를 output.buffer에 저장

        // 추론 결과로부터 가장 신뢰도 높은 객체를 선택하고, 추론 시간을 측정
        val bestBoxes = bestBox(output.floatArray) // 모델 예측 결과를 기반으로 신뢰도가 높은 검출 박스를 선택
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime // 현재 시간에서 시작 시간을 빼서 추론 소요 시간 계산

        if (bestBoxes == null) { // 감지된 객체가 없는 경우 onEmptyDetect() 호출
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime) // 감지된 객체가 있는 경우 감지된 박스 리스트와 추론 시간을 전달
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
                val clsName = labels[maxIdx] // 해당 객체의 클래스 이름 가져옴
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

        return applyNMS(boundingBoxes) // 검출된 결과에 대해서 비최대 억제법(NMS)를 적용하여 중복 박스를 제거한 후 반환
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