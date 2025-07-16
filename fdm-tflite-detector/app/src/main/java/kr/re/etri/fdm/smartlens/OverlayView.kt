package kr.re.etri.fdm.smartlens

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kr.re.etri.fdm.smartlens.Constants.DEFAULT_TEXT_SIZE
import kr.re.etri.fdm.smartlens.Constants.BOX_STROKE_WIDTH

// Displays the detected objects on the screen
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>() // list of detected object (bounding boxes)
    // 화면에 정보를 알려준 Paint 객체둘
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect() // 텍스트 크기 측정을 위한 Rect 객체

    init {
        initPaints(context)
    }

    fun clear() {
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints(context)
    }

    // Initialize 'Paint' objects to specify text and box styles
    private fun initPaints(context: Context?) {
        // text background
        textBackgroundPaint.apply {
            color = Color.BLACK
            style = Paint.Style.FILL
            textSize = DEFAULT_TEXT_SIZE
        }

        textPaint.apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            textSize = DEFAULT_TEXT_SIZE
        }

        boxPaint.apply {
            color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
            strokeWidth = BOX_STROKE_WIDTH
            style = Paint.Style.STROKE
        }
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Create a string with the class name and probability of the detected object
            val drawableText = "${it.clsName} ${(it.cnf * 100).toInt()}%" // E.g. "Ulcer 99%"
            // val drawableText = it.clsName

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)

        }
    }

    // Set detected bounding boxes and redraw full view
    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    // 텍스트 배경에 패딩 추가를 위해 상수 설정
    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}