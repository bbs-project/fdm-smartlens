package kr.re.etri.fdm.smartlens.data

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.text.SimpleDateFormat
import java.util.*

/**
 * 진단 결과 데이터 모델
 */
data class DiagnosisResult(
    val id: String,
    val timestamp: Long,
    val diseaseName: String,
    val diseaseCode: Int,
    val confidence: Float,
    val symptoms: List<SymptomInfo>,
    val imageUrl: String? = null,
    val notes: String? = null
)

data class SymptomInfo(
    val name: String,
    val confidence: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

/**
 * 진단 통계 데이터 모델
 */
data class DiagnosisStatistics(
    val totalDiagnoses: Int,
    val diseaseDistribution: Map<String, Int>,
    val averageConfidence: Float,
    val lastDiagnosisDate: Long?
)

/**
 * 진단 결과 저장 서비스
 */
class DiagnosisRepository(context: Context) {
    
    private val prefs: SharedPreferences = context.getSharedPreferences(
        PREFS_NAME, Context.MODE_PRIVATE
    )
    private val gson = Gson()
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.KOREA)
    
    companion object {
        private const val PREFS_NAME = "fdm_smartlens_diagnosis"
        private const val KEY_HISTORY = "diagnosis_history"
        private const val MAX_HISTORY_ITEMS = 100
    }
    
    /**
     * 진단 결과를 저장합니다
     */
    fun saveDiagnosis(
        diseaseName: String,
        diseaseCode: Int,
        confidence: Float,
        symptoms: List<SymptomInfo>,
        imageUrl: String? = null,
        notes: String? = null
    ): DiagnosisResult {
        val result = DiagnosisResult(
            id = generateId(),
            timestamp = System.currentTimeMillis(),
            diseaseName = diseaseName,
            diseaseCode = diseaseCode,
            confidence = confidence,
            symptoms = symptoms,
            imageUrl = imageUrl,
            notes = notes
        )
        
        val history = getHistory().toMutableList()
        history.add(0, result) // Add to beginning
        
        // Limit history size
        if (history.size > MAX_HISTORY_ITEMS) {
            history.removeAt(history.lastIndex)
        }
        
        saveHistory(history)
        return result
    }
    
    /**
     * 전체 진단 기록을 조회합니다
     */
    fun getHistory(): List<DiagnosisResult> {
        val json = prefs.getString(KEY_HISTORY, null) ?: return emptyList()
        val type = object : TypeToken<List<DiagnosisResult>>() {}.type
        return try {
            gson.fromJson(json, type)
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * 특정 진단 기록을 조회합니다
     */
    fun getDiagnosis(id: String): DiagnosisResult? {
        return getHistory().find { it.id == id }
    }
    
    /**
     * 진단 기록을 삭제합니다
     */
    fun deleteDiagnosis(id: String): Boolean {
        val history = getHistory().filter { it.id != id }.toMutableList()
        saveHistory(history)
        return true
    }
    
    /**
     * 전체 진단 기록을 삭제합니다
     */
    fun clearHistory() {
        prefs.edit().remove(KEY_HISTORY).apply()
    }
    
    /**
     * 최근 진단 결과를 조회합니다
     */
    fun getRecentDiagnoses(limit: Int = 10): List<DiagnosisResult> {
        return getHistory().take(limit)
    }
    
    /**
     * 통계 정보를 조회합니다
     */
    fun getStatistics(): DiagnosisStatistics {
        val history = getHistory()
        
        if (history.isEmpty()) {
            return DiagnosisStatistics(
                totalDiagnoses = 0,
                diseaseDistribution = emptyMap(),
                averageConfidence = 0f,
                lastDiagnosisDate = null
            )
        }
        
        val diseaseDistribution = history.groupingBy { it.diseaseName }
            .eachCount()
        
        val totalConfidence = history.sumOf { it.confidence.toDouble() }.toFloat()
        
        return DiagnosisStatistics(
            totalDiagnoses = history.size,
            diseaseDistribution = diseaseDistribution,
            averageConfidence = totalConfidence / history.size,
            lastDiagnosisDate = history.firstOrNull()?.timestamp
        )
    }
    
    /**
     * 진단 기록을 JSON 으로 내보냅니다
     */
    fun exportHistory(): String {
        return gson.toJson(getHistory())
    }
    
    /**
     * JSON 에서 진단 기록을 가져옵니다
     */
    fun importHistory(json: String): Boolean {
        return try {
            val type = object : TypeToken<List<DiagnosisResult>>() {}.type
            val history: List<DiagnosisResult> = gson.fromJson(json, type)
            saveHistory(history.toMutableList())
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * 기록을 저장합니다
     */
    private fun saveHistory(history: MutableList<DiagnosisResult>) {
        val json = gson.toJson(history)
        prefs.edit().putString(KEY_HISTORY, json).apply()
    }
    
    /**
     * 고유 ID 생성
     */
    private fun generateId(): String {
        return "diag_${System.currentTimeMillis()}_${(Math.random() * 1000000).toInt()}"
    }
    
    /**
     * 날짜 포맷팅
     */
    fun formatDate(timestamp: Long): String {
        return dateFormat.format(Date(timestamp))
    }
}
