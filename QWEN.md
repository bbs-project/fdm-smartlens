# FDM SmartLens (fdm-smartlens)

## Project Overview

**FDM SmartLens** is a mobile AI application for detecting diseases in flounder (넙치) fish. The project provides real-time object detection and disease classification using camera input, implemented across multiple platforms:

- **Native Android App** (Kotlin) - Using TensorFlow Lite with YOLOv8 and VGG16 models
- **React Native Apps** (JavaScript/TypeScript) - Using TensorFlow.js with YOLOv8 and VGG16 models

The system performs two-stage detection:
1. **YOLOv8** - Object detection to locate fish and symptoms (Bleeding, Corrosion, Tumor, Ulcer, EyesSymptom)
2. **VGG16** - Image classification to identify specific diseases (7 disease classes)

### Detection Pipeline

```
Camera Frame → YOLOv8 (symptom detection) → VGG16 (disease classification) → Result
                    [640x640]                    [112x112]
```

### YOLOv8 Output Classes (Symptoms)
- Bleeding (출혈)
- Corrosion (부식/침식)
- Tumor (종양)
- Ulcer (궤양)
- EyesSymptom (눈 증상)

### VGG16 Output Classes (Diseases)

| Index | Korean | English |
|-------|--------|---------|
| 1 | 바이러스성출혈성패혈증 | Viral Hemorrhagic Septicemia |
| 2 | 림포시스티스병 | Lymphocystis Disease |
| 6 | 여윔병 | Streptococcosis |
| 8 | 스쿠티카병 | Scuticociliatosis |
| 11 | 연쇄구균증 | Streptococcosis |
| 13 | 비브리오병 | Vibriosis |
| 19 | 에드워드병 | Edwardsiellosis |

## Project Structure

```
fdm-smartlens/
├── fdm-tflite-detector/           # Native Android app (Kotlin + TFLite)
│   ├── app/src/main/
│   │   ├── java/kr/re/etri/fdm/smartlens/
│   │   │   ├── MainActivity.kt    # Main activity with CameraX integration
│   │   │   ├── Detector.kt        # TFLite inference engine (YOLOv8 + VGG16)
│   │   │   ├── BoundingBox.kt     # Bounding box data class
│   │   │   ├── OverlayView.kt     # Custom view for drawing detection results
│   │   │   └── Constants.kt       # Model paths and configuration
│   │   ├── assets/                # TFLite models and label files
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
│
├── fdm-yolov8n-vgg16/             # React Native app (TypeScript + TFJS)
│   ├── src/
│   │   ├── App.tsx                # Main app component
│   │   ├── CameraView/            # Camera component with tensor support
│   │   ├── modelHandler/          # Model loading configuration
│   │   └── utils/                 # Detection utilities
│   │       ├── detectBox.ts       # YOLO and VGG detection logic
│   │       ├── preprocess.js      # Image preprocessing
│   │       ├── renderBox.ts       # Box rendering utilities
│   │       └── labels.json        # Class labels
│   └── assets/model/              # TensorFlow.js models
│       ├── fdm-yolov8n/           # YOLOv8n TFJS model (3 shards)
│       └── fdm-vgg16/             # VGG16 TFJS model (20 shards)
│
├── YOLOv8-TfLite-Object-Detector/ # Base YOLOv8 Android detector (reference)
├── VGG16-TfLite-Object-Detector/  # Base VGG16 Android detector (reference)
├── yolov8n-tfjs-react-native/     # React Native app with YOLOv8
├── yolov5-tfjs-react-native-master/ # React Native app with YOLOv5 (reference)
├── smartlens-data/                # Training dataset organized by disease class
└── 안드로이드 스마트 렌즈 구현 ReadMe/ # Implementation documentation (Korean)
```

## Building and Running

### Native Android App (fdm-tflite-detector)

**Prerequisites:**
- Android Studio (Arctic Fox or later)
- JDK 17
- Android SDK (API 26+)

**Build Commands:**
```bash
cd fdm-tflite-detector

# Debug build
./gradlew assembleDebug

# Release build
./gradlew assembleRelease

# Install on connected device
./gradlew installDebug

# Run tests
./gradlew test
```

**Model Files:** Located in `fdm-tflite-detector/app/src/main/assets/`
- `fdm_yolov8n_float16.tflite` - YOLOv8n detection model (float16 quantized)
- `fdm_vgg16_float16.tflite` - VGG16 classification model (float16 quantized)
- `labels_yolov8.txt` - YOLOv8 labels (5 symptom classes)
- `labels_vgg16.txt` - VGG16 labels (7 disease classes)

**Key Configuration (Constants.kt):**
```kotlin
const val MODEL_YOLOV8_PATH = "fdm_yolov8n_float16.tflite"
const val LABELS_YOLOV8_PATH = "labels_yolov8.txt"
const val MODEL_VGG16_PATH = "fdm_vgg16_float16.tflite"
const val LABELS_VGG16_PATH = "labels_vgg16.txt"
const val CONFIDENCE_THRESHOLD = 0.3F
const val IOU_THRESHOLD = 0.5F  // NMS threshold
```

### React Native Apps (fdm-yolov8n-vgg16)

**Prerequisites:**
- Node.js 18+
- Yarn
- Expo CLI
- Android Studio / Xcode for native builds

**Setup and Run:**
```bash
cd fdm-yolov8n-vgg16

# Install dependencies
yarn install

# Start Expo development server
yarn start

# Run on Android
yarn android

# Run on iOS
yarn ios

# Run on Web
yarn web

# Build APK
cd android && ./gradlew assembleRelease
```

**Test Commands:**
```bash
# Unit tests
yarn test

# E2E tests (Android)
yarn test:e2e:android

# E2E tests (iOS)
yarn test:e2e:ios

# Type check
yarn typecheck
```

**Model Files:** Located in `fdm-yolov8n-vgg16/assets/model/`
- `fdm-yolov8n/` - YOLOv8n TensorFlow.js model (3 weight shards)
- `fdm-vgg16/` - VGG16 TensorFlow.js model (20 weight shards)

**Model Configuration (src/modelHandler/index.js):**
```javascript
// YOLOv8n: input shape [1, 3, 640, 640]
const yoloModelJson = require("../../assets/model/fdm-yolov8n/model.json");
const yoloModelWeights = [
  require("../../assets/model/fdm-yolov8n/group1-shard1of3.bin"),
  require("../../assets/model/fdm-yolov8n/group1-shard2of3.bin"),
  require("../../assets/model/fdm-yolov8n/group1-shard3of3.bin"),
];
export const yoloModelURI = bundleResourceIO(yoloModelJson, yoloModelWeights);

// VGG16: input shape [1, 112, 112, 3]
const vggModelJson = require("../../assets/model/fdm-vgg16/model.json");
const vggModelWeights = [ /* 20 shards */ ];
export const vggModelURI = bundleResourceIO(vggModelJson, vggModelWeights);
```

## Key Technologies

### Android (Native)
- **Language:** Kotlin
- **ML Framework:** TensorFlow Lite 2.16.1
  - `tensorflow-lite-support` - Image preprocessing utilities
  - `tensorflow-lite-gpu-delegate-plugin` - GPU acceleration
- **Camera:** CameraX 1.4.0-alpha04
  - `ImageAnalysis` - Real-time frame processing
  - `Preview` - Camera preview display
- **UI:** ViewBinding, Material Design, Custom OverlayView
- **Target SDK:** 34, Min SDK: 26

### React Native
- **Framework:** Expo SDK 52, React Native 0.76.3
- **ML Framework:** TensorFlow.js 4.22.0
  - `@tensorflow/tfjs-react-native` - RN backend (rn-webgl)
  - `@tensorflow/tfjs-backend-webgl` - WebGL backend for web
- **Camera:** expo-camera with `cameraWithTensors` HOC
- **Styling:** NativeWind (Tailwind CSS for RN)
- **Testing:** Jest (unit), Detox (E2E)
- **Language:** TypeScript (fdm-yolov8n-vgg16)

### ML Models

| Model | Input Shape | Output Shape | Purpose |
|-------|-------------|--------------|---------|
| YOLOv8n | [1, 3, 640, 640] | [1, 9, 8400] | Symptom detection (5 classes) |
| VGG16 | [1, 112, 112, 3] | [1, 7] | Disease classification (7 classes) |

**YOLOv8 Output Format:**
- 4 bounding box coordinates (cx, cy, w, h)
- 5 class probabilities (symptoms)

**VGG16 Output:**
- 7 disease class probabilities (sigmoid activation)

## Development Conventions

### Code Style
- **Kotlin:** Standard Kotlin conventions
  - Files: `src/main/java/kr/re/etri/fdm/smartlens/`
  - Data classes for bounding boxes, interfaces for callbacks
- **JavaScript/TypeScript:** ESLint configured, TypeScript for type safety
  - Functional components with hooks (useState, useEffect, useCallback)
  - Tensor memory management with `tf.tidy()` and `tf.dispose()`
- **Naming:** CamelCase for variables/functions, PascalCase for components/classes

### Architecture Patterns
- **Android:** 
  - `MainActivity` - CameraX lifecycle management, UI binding
  - `Detector` - TFLite interpreter management, inference pipeline
  - `DetectorListener` - Callback interface for detection results
  - GPU delegate with fallback to CPU (4 threads)
- **React Native:**
  - Component-based architecture with `CameraView` component
  - `modelHandler` abstraction for model loading
  - Real-time detection using `requestAnimationFrame`
  - Tensor memory cleanup in `finally` blocks

### Detection Pipeline

**Android (Detector.kt):**
1. Preprocess image (resize, normalize to [-1, 1])
2. Run YOLOv8 inference → Get symptom bounding boxes
3. Apply NMS (IoU threshold: 0.5) to remove duplicates
4. Run VGG16 inference on full frame → Get disease classification
5. Return combined results with inference times

**React Native (detectBox.ts):**
1. Preprocess: transpose [H, W, C] → [1, C, H, W]
2. Run YOLOv8 with `executeAsync()`
3. Extract boxes, scores, classes from output tensor
4. Apply `tf.image.nonMaxSuppressionAsync` (IoU: 0.45, score: 0.7)
5. Run VGG16 on resized image (112x112)
6. Convert class indices to Korean disease names

### Testing Practices
- Unit tests with Jest (React Native)
- E2E tests with Detox (React Native)
- Manual testing on physical devices recommended for camera functionality

### Model Integration
When updating ML models:

**Android (TFLite):**
1. Place `.tflite` model in `app/src/main/assets/`
2. Update `Constants.kt` with new model path
3. Update label file if classes changed
4. Adjust input/output tensor shapes in `Detector.kt` if needed

**React Native (TFJS):**
1. Export model to TensorFlow.js format
2. Place model files in `assets/model/`
3. Update `src/modelHandler/index.js` (Android) or `index.web.js` (Web)
4. Update `src/utils/labels.json` and `vgglabels.json` for custom labels
5. Adjust `inputTensorSize` in App component if model input changes

## Dataset Structure

Training data is organized in `smartlens-data/` by disease class:

```
smartlens-data/
├── 0-정상/                          # Healthy flounder images
├── 1-바이러스성출혈성패혈증/        # Viral Hemorrhagic Septicemia
├── 2-림포시스티스병/               # Lymphocystis Disease
├── 6-여윔병/                        # Streptococcosis
├── 8-스쿠티카병/                    # Scuticociliatosis
├── 11-연쇄구균증/                   # Streptococcosis
├── 13-비브리오병/                   # Vibriosis
└── 19-에드워드병/                   # Edwardsiellosis
```

Image naming convention (example):
```
F03_U01_O2292_D2022-09-20_L310_W0275_S2_R01_B02_I00000017.JPG
```

## Additional Resources

- **Implementation Guide:** `안드로이드 스마트 렌즈 구현 ReadMe/`
  - `requirements.txt` - Python dependencies for model training
  - `스마트 렌즈 환경 구성 참고 사이트 모음집.txt` - Reference links for implementation
  - `안드로이드 스마트 렌즈 ReadMe.pptx` - Presentation slides

- **Reference Links:**
  - TensorFlow Lite mobile AI projects: https://smilegate.ai/2021/03/19/awesome-tensorflow-lite/
  - YOLOv5 React Native: https://github.com/Hyuto/yolov5-tfjs-react-native
  - YOLOv8 Android (Kotlin): https://github.com/surendramaran/YOLOv8-TfLite-Object-Detector
  - Ultralytics YOLO issues: https://github.com/ultralytics/ultralytics/issues

- **Security Policy:** `SECURITY.md`

## Contact & Support

- **Repository:** https://github.com/bbs-project/fdm-lens
- **Issues:** https://github.com/bbs-project/fdm-lens/issues
