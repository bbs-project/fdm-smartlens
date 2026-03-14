import labels from "../labels.json"; // YOLO 결과를 출력하기 위해 클래스 레이블이 정의된 JSON 파일을 가져옴
import vgglabels from "../vgglabels.json"; // YOLO 결과를 출력하기 위해 클래스 레이블이 정의된 JSON 파일을 가져옴
import { Colors } from "../utils";

/**
 * Render prediction boxes
 * @param {Expo2DContext} ctx Expo context, Expo 2D 컨텍스트 객체로 해당 객체를 사용하여 화면에 그림을 그린다.
 * @param {number} threshold threshold number, 점수의 임계값으로 해당 threshold 이상의 점수만 렌더링
 * @param {Array} boxes_data boxes array, 예측 경계 상자 배열
 * @param {Array} scores_data scores array, 예측 점수 배열
 * @param {Array} classes_data class array, 예측 클래스 배열
 * @param {Array[Number]} ratios boxes ratio [xRatio, yRatio], 경계 상자 비율 나타내는 배열
 */
export const renderBoxes = async ( // 예측 경계 상자를 렌더링하는 역할
  ctx,
  threshold,
  num_detections,
  boxes_data,
  scores_data,
  classes_data,
  ratios,
  flipX = true // x축을 뒤집을지 여부
) => {
  // console.log("[render input] threshold, boxes_data, scores_data, classes_data, ratios, flipX: ", threshold, boxes_data, scores_data, classes_data, ratios, flipX);
  // console.log("[YOLO]=[", boxes_data, scores_data, classes_data, ratios, "]");
  // check if ctx is expo 2d context
  if (!ctx || !ctx.flush) {
    console.error("[renderYoloBoxes] ctx is not an Expo 2D context");
    return;
  }
  
  let xratio = ratios[0];
  let yratio = ratios[1];

  // ctx.clearRect(0, 0, ctx.width, ctx.height); // clean canvas, 캔버스를 초기화

  // font configs
  const font = `${Math.max(Math.round(Math.max(ctx.width, ctx.height) / 40), 14)}pt sans-serif`; // 텍스트 그릴 때 사용 폰트, 캔버스 너비와 높이 중 더 큰 값 기준으로 폰트 크기 설정(최소 폰트 크기 14pt)
  ctx.font = font;
  ctx.textBaseline = "top"; // text 베이스라인 top 설정

  const colors = new Colors(); // 색상 유틸리티 객체 초기화

  // const color = colors.get(5);
  // ctx.fillStyle = Colors.hexToRgba(color, 0.2); // 경계 상자의 반투명 배경색 설정   
  // ctx.fillRect(100, 100, 100, 100);
  // ctx.fillRect(466.36212158203125, 380.9473876953125, 143.83380126953125, 159.82557678222656);
 
  // const color2 = colors.get(1);
  // ctx.fillStyle = Colors.hexToRgba(color2, 0.2); // 경계 상자의 반투명 배경색 설정   
  // ctx.fillRect(466.36212158203125-100, 380.9473876953125-100, 143.83380126953125, 159.82557678222656);

  // const color3 = colors.get(9);
  // ctx.fillStyle = Colors.hexToRgba(color2, 0.2); // 경계 상자의 반투명 배경색 설정   
  // // ctx.fillRect(242.59141159057617, 269.8766098022461, 544.6524181365967, 373.113315582275);
  // ctx.fillRect(242, 269.8766098022461 + 373.113315582275, 544.6524181365967, 373.113315582275);

  // Draw labels
  // ctx.fillStyle = "#ffffff";
  // ctx.fillStyle = "#000000";
  // ctx.fillText("눈의 질병", 300, 400);// 레이블을 화면에 그림

  for (let i = 0; i < num_detections; ++i) { // 실제로 예측 경계 상자 화면에 그리는 작업
    if (scores_data[i] > threshold) { // score가 threshold 보다 큰 경우에만 수행
      const code = classes_data[i]; // 예측된 클래스 ID를 가져옴
      const klass = labels[code]; // 예측된 클래스 ID에 해당하는 클래스 이름을 'labels' 에서 가져옴
      const color = colors.get(code); // 예측된 클래스 ID에 해당하는 색상을 'colors' 에서 가져옴
      const score = (scores_data[i] * 100).toFixed(1); // 예측 점수를 백분율로 변환하여 소수점 한 자리까지 문자열로 저장
 
      let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);

      //console.log("[YOLO] (x1,y1,x2,y2)=[", x1, y1, x2, y2, "]"); 
      //console.log("ctx.width, ctx.height, xratio, yratio: ", ctx.width, ctx.height, xratio, yratio); 
 
      // 입력 이미지 크기: 640x640
      // 출력 장치 크기: 1080x2214
      // 좌표를 화면 크기(1080x2214)에 맞게 스케일링
      // x1 = x1 * (ctx.width / 640);
      // x2 = x2 * (ctx.width / 640);
      // y1 = y1 * (ctx.height / 640);
      // y2 = y2 * (ctx.height / 640);

      let width = x2 - x1;
      let height = y2 - y1; 

      // x1 = x1 * 1.1;
      y1 = y1 * 2;
      // width = width * 1.1;
      height = height * 2; 


      // flip horizontal
      // if (flipX) {
      //   x1 = ctx.width - x1 - width; // flipX가 true 인 경우 경계 상자의 X 좌표를 화면 너비에서 좌표와 너비를 뺀 값으로 설정하여 수평으로 뒤집음
      // }

      // Draw the bounding box.
      const color19 = colors.get(19); // Light Pink
      const color3 = colors.get(3); // Orange Red
      const color18 = colors.get(18); // Medium Orchid

      // ctx.fillStyle = Colors.hexToRgba(color18, 0.2); // 경계 상자의 반투명 배경색 설정
      // ctx.fillRect(x1, y1, width, height); // 경계 상자를 화면에 그림

      // ctx.strokeStyle = "#00FFFF"; // bounding box's line color is Cyan
      ctx.strokeStyle = color3; // bounding box's line color is Orange Red
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, width, height); // 경계 상자의 테두리를 화면에 그림
      
      // Draw the label background.
      ctx.fillStyle = color; // 레이블 배경색 설정 
      const textWidth = ctx.measureText(klass + " - " + score + "%").width; // 레이블 텍스트 너비 측정
      const textHeight = parseInt(font, 10); // base 10, 레이블 텍스트 높이 계산
      const yText = y1 - (textHeight + 2);
      ctx.fillRect(x1 - 1, yText < 0 ? 0 : yText, textWidth + 2, textHeight + 2);// 레이블 배경을 화면에 그림
      // console.log("[renderYoloBoxes 3] x - 1, yText < 0 ? 0, textWidth + 2, textHeight + 2", x - 1, yText, textWidth + 2, textHeight + 2);

      // Draw labels      
      ctx.fillStyle = "#000000"; // label's color is black
      // ctx.fillStyle = "#ffffff"; // label's color is white
      const krKlass = getKoreanKlass(code);

      // ctx.fillText(`${klass} [${krKlass}], (${score}%)`, x1 - 1, yText < 0 ? 0 : yText, width);// 레이블을 화면에 그림
      ctx.fillText(`${klass} (${score}%)`, x1 - 1, yText < 0 ? 0 : yText, width);// 레이블을 화면에 그림
    }
  }

  /**
   * Get Korean class name
   * @param {integer} code class code
   * @returns {string} Korean class name
   */
  function getKoreanKlass(code) {
    const koreanLabels = {
      0: "출혈",      // Bleeding
      1: "궤양",      // Corrosion
      2: "부식",      // Tumor
      3: "종양",      // Ulcer
      4: "안구증상"   // EyesSymptom
    };
    return koreanLabels[code] !== undefined ? koreanLabels[code] : code;
  }

};
