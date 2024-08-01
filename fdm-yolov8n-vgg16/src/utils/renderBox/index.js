import labels from "../labels.json"; // 결과를 출력하기 위해 클래스 레이블이 정의된 JSON 파일을 가져옴
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
  boxes_data,
  scores_data,
  classes_data,
  ratios,
  flipX = true // x축을 뒤집을지 여부
) => {
  ctx.clearRect(0, 0, ctx.width, ctx.height); // clean canvas, 캔버스를 초기화

  // font configs
  const font = `${Math.max(Math.round(Math.max(ctx.width, ctx.height) / 40), 14)}pt sans-serif`; // 텍스트 그릴 때 사용 폰트, 캔버스 너비와 높이 중 더 큰 값 기준으로 폰트 크기 설정(최소 폰트 크기 14pt)
  ctx.font = font;
  ctx.textBaseline = "top"; // text 베이스라인 top 설정

  const colors = new Colors(); // 색상 유틸리티 객체 초기화

  for (let i = 0; i < scores_data.length; ++i) { // 실제로 예측 경계 상자 화면에 그리는 작업
    if (scores_data[i] > threshold) { // score가 threshold가 큰 경우에만 수행
      const klass = labels[classes_data[i]]; // 예측된 클래스 ID에 해당하는 클래스 이름을 'labels' 에서 가져옴
      const color = colors.get(classes_data[i]); // 예측된 클래스 ID에 해당하는 색상을 'colors' 에서 가져옴
      const score = (scores_data[i] * 100).toFixed(1); // 예측 점수를 백분율로 변환하여 소수점 한 자리까지 문자열로 저장

      let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4); // 현재 예측에 해당하는 경계 상자의 좌표를 추출
      x1 *= ctx.width * ratios[0];
      x2 *= ctx.width * ratios[0];
      y1 *= ctx.height * ratios[1];
      y2 *= ctx.height * ratios[1];
      const width = x2 - x1;
      const height = y2 - y1;

      // flip horizontal
      let x;
      if (flipX) x = ctx.width - x1 - width; // flipX가 true 인 경우 경계 상자의 X 좌표를 화면 너비에서 좌표와 너비를 뺀 값으로 설정하여 수평으로 뒤집음
      else x = x1;

      // Draw the bounding box.
      // strokeRect not rendering!
      ctx.fillStyle = Colors.hexToRgba(color, 0.2); // 경계 상자의 반투명 배경색 설정
      ctx.fillRect(x, y1, width, height); // 경계 상자를 화면에 그림

      // Draw the label background.
      ctx.fillStyle = color; // 레이블 배경색 설정
      const textWidth = ctx.measureText(klass + " - " + score + "%").width; // 레이블 텍스트 너비 측정
      const textHeight = parseInt(font, 10); // base 10, 레이블 텍스트 높이 계산
      const yText = y1 - (textHeight + 2);
      ctx.fillRect(x - 1, yText < 0 ? 0 : yText, textWidth + 2, textHeight + 2);// 레이블 배경을 화면에 그림

      // Draw labels
      ctx.fillStyle = "#ffffff";
      ctx.fillText(klass + " - " + score + "%", x - 1, yText < 0 ? 0 : yText);// 레이블을 화면에 그림
    }
  }
  ctx.flush(); // 화면에 실제로 반영되도록 함
};
