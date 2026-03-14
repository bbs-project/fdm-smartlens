# fdm-yolov8n-vgg16

## Project Overview

React Native/Expo 기반 멀티플랫폼 어류 질병 탐지 앱.
YOLOv8 Nano로 증상(출혈/궤양/부식/종양/안구증상)을 검출하고, VGG16으로 7종 질병을 분류한다.

- **패키지명**: kr.etri.bbs.fdm_lens
- **플랫폼**: Android (검증), Web (검증), iOS (미검증)

## Tech Stack

- React Native 0.69.9 + Expo 46
- TensorFlow.js + tfjs-react-native
- NativeWind (Tailwind CSS for RN)
- Expo Camera, Expo GL, Expo 2D Context

## Commands

```bash
yarn              # 의존성 설치
yarn start        # Expo Go 실행
yarn android      # Android 실행
yarn web          # Web 실행
yarn ios          # iOS 실행
```

### Android APK 빌드

```bash
cd android
./gradlew assembleRelease
# 출력: android/app/build/outputs/apk/release/app-release.apk
```

## Project Structure

```
src/
  App.js                    # 진입점, 모델 로딩, 카메라 권한
  CameraView/
    index.js                # Native: TensorCamera + 탐지 루프
    index.web.js            # Web: HTML5 Video + Canvas
  modelHandler/
    index.js                # Native: bundleResourceIO 모델 로딩
    index.web.js            # Web: static URL 모델 로딩
  utils/
    detectBox.js            # YOLOv8 검출 + VGG16 분류 로직
    preprocess.js           # 이미지 전처리 (패딩, 리사이즈, 정규화)
    renderBox/
      index.js              # Native: Expo2DContext 바운딩 박스 렌더링
      index.web.js          # Web: Canvas2D 바운딩 박스 렌더링
    labels.json             # YOLO 클래스 레이블 (5종 증상)
    vgglabels.json          # VGG16 클래스 레이블 (7종 질병)
    utils.js                # 색상 팔레트 유틸리티
assets/model/
  fdm-yolov8n/              # YOLOv8 Nano 모델 (3 shards)
  fdm-vgg16/                # VGG16 모델 (20 shards)
```

## Models

### YOLOv8 Nano (증상 검출)
- **입력**: [1, 3, 640, 640]
- **출력**: [1, 9, 8400] -> transpose -> [1, 8400, 9]
- **클래스**: Bleeding, Corrosion, Tumor, Ulcer, EyesSymptom
- **NMS**: IoU 0.45, Score 0.7, 최대 3개 검출

### VGG16 (질병 분류)
- **입력**: [1, 112, 112, 3]
- **출력**: 7개 이진 예측
- **질병 코드 매핑**:
  - 1: 바이러스성출혈성패혈증
  - 2: 림포시스티스병
  - 6: 여윔병
  - 8: 스쿠티카병
  - 11: 연쇄구균증
  - 13: 비브리오병
  - 19: 에드워드병

## Architecture Notes

- Native에서 플랫폼별 파일 분기: `*.web.js` (Web), `*.js` (Native)
- Metro 설정에서 `.bin` 확장자를 에셋으로 등록 (모델 웨이트 번들링)
- Webpack에서 CopyWebpackPlugin으로 모델 파일을 static 서빙
- `tf.engine().startScope()`/`endScope()`로 프레임별 텐서 메모리 관리

## Known Limitations

- Web 버전에서 VGG16 분류 미통합 (YOLO 검출만 동작)
- 프레임 처리 간격이 1초로 고정 (실시간성 제한)
- iOS 미검증
- Expo 46 기반 (구버전)
