package kr.re.etri.fdm.smartlens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kr.re.etri.fdm.smartlens.Constants.LABELS_YOLOV8_PATH
import kr.re.etri.fdm.smartlens.Constants.MODEL_YOLOV8_PATH
import kr.re.etri.fdm.smartlens.Constants.LABELS_VGG16_PATH
import kr.re.etri.fdm.smartlens.Constants.MODEL_VGG16_PATH
import kr.re.etri.fdm.smartlens.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding // 데이터 바인딩 객체
    private val isFrontCamera = false

    // 카메라 미리 보기 이미지 분석을 위한 컴포넌트
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null // 객체 탐지 모델

    private lateinit var cameraExecutor: ExecutorService // 비동기 작업 처리를 위한 'ExecutorService'

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater) // 데이터 바인딩을 사용해 UI 설정
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor() // 카메라 관련 작업 처리 단일 스레드 실행

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_YOLOV8_PATH, LABELS_YOLOV8_PATH, MODEL_VGG16_PATH, LABELS_VGG16_PATH, this) // 모델과 레이블 파일 로드
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()
    }

    // UI 요소에 이벤트 리스너 설정
    private fun bindListeners() {
        binding.apply {
            isGpu.setOnCheckedChangeListener { buttonView, isChecked -> // GPU 사용 여부
                cameraExecutor.submit {
                    detector?.restart(isGpu = isChecked)
                }
                if (isChecked) {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.orange))
                } else {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.gray))
                }
            }
        }
    }

    // 카메라 초기화
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this) // 카메라 provider을 비동기로 가져옴
        cameraProviderFuture.addListener({ // 카메라 provider가 준비되면 bindCameraUseCases()를 호출하여 카메라 사용 설정
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases() // 카메라 미리보기, 이미지 분석 설정
        }, ContextCompat.getMainExecutor(this))
    }

    // 카메라 프리뷰와 이미지 분석 설정, 모델을 사용하여 실시간 객체 감지를 수행
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder() // 카메라의 비율과 회전 지정
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder() // 이미지 분석 기능 설정, 가장 최신의 이미지만 처리하도록 'STRATEGY_KEEP_ONLY_LATEST' 사용
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy -> // 분석 실행마다 비트맵 생성, 필요 시 회전 또는 좌우 반전 적용
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            detector?.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll() // 기존 바인딩된 모든 사용 사례 해제( 리소스 확보 )

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    // 권한 체크 및 요청
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() { // 감지된 객체가 없을 때 호출, 오버레이 지우는 역할
        runOnUiThread {
            binding.overlay.clear()
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) { // 감지된 객체가 있을 때 호출, 감지된 박스를 오버레이 표시, 추론 시간 업데이트
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }
}
