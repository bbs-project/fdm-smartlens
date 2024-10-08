export class Colors {

// ultralytics color palette https://ultralytics.com/
// 1.#FF3838: Red Orange
// 2. #FF9D97: Light Coral
// 3. #FF701F: Orange Red
// 4. #FFB21D: Bright Orange
// 5. #CFD231: Yellow Green
// 6. #48F90A: Lime Green
// 7. #92CC17: Yellow Green
// 8. #3DDB86: Medium Sea Green
// 9. #1A9334: Forest Green
// 10. #00D4BB: Turquoise
// 11. #2C99A8: Light Sea Green
// 12. #00C2FF: Deep Sky Blue
// 13. #344593: Dark Slate Blue
// 14. #6473FF: Cornflower Blue
// 15. #0018EC: Blue
// 16. #8438FF: Purple
// 17. #520085: Dark Purple
// 18. #CB38FF: Medium Orchid
// 19. #FF95C8: Light Pink
// 20. #FF37C7: Hot Pink
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(
          ", "
        )}, ${alpha})`
      : null;
  };
}
