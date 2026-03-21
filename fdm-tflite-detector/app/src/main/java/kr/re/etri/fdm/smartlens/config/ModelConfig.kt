package kr.re.etri.fdm.smartlens.config

/**
 * 모델 클래스 정보 데이터 모델
 */
data class ModelClassInfo(
    val id: Int,
    val name: String,
    val nameEn: String,
    val description: String? = null,
    val color: String,
    val icon: String
)

/**
 * 모델 설정 관리 클래스
 */
object ModelConfig {
    
    /**
     * YOLOv8 증상 클래스
     */
    val YOLO_CLASSES = listOf(
        ModelClassInfo(0, "Bleeding", "Bleeding", null, "#F44336", "drop"),
        ModelClassInfo(1, "Corrosion", "Corrosion", null, "#FF9800", "alert-circle"),
        ModelClassInfo(2, "Tumor", "Tumor", null, "#9C27B0", "circle-outline"),
        ModelClassInfo(3, "Ulcer", "Ulcer", null, "#E91E63", "alert-circle"),
        ModelClassInfo(4, "EyesSymptom", "Eye Symptom", null, "#2196F3", "eye")
    )
    
    /**
     * VGG16 질병 클래스
     */
    val VGG_CLASSES = listOf(
        ModelClassInfo(
            1, 
            "바이러스성출혈성패혈증", 
            "Viral Hemorrhagic Septicemia",
            "바이러스성 출혈성 패혈증",
            "#F44336",
            "drop"
        ),
        ModelClassInfo(
            2, 
            "림포시스티스병", 
            "Lymphocystis Disease",
            "림포시스티스 바이러스 감염",
            "#FF9800",
            "virus"
        ),
        ModelClassInfo(
            6, 
            "여윔병", 
            "Streptococcosis",
            "연쇄상구균 감염",
            "#4CAF50",
            "bacteria"
        ),
        ModelClassInfo(
            8, 
            "스쿠티카병", 
            "Scuticociliatosis",
            "스쿠티카 섬모충 감염",
            "#9C27B0",
            "bug"
        ),
        ModelClassInfo(
            11, 
            "연쇄구균증", 
            "Streptococcosis",
            "연쇄구균 감염",
            "#00BCD4",
            "bacteria"
        ),
        ModelClassInfo(
            13, 
            "비브리오병", 
            "Vibriosis",
            "비브리오 균 감염",
            "#E91E63",
            "bacteria"
        ),
        ModelClassInfo(
            19, 
            "에드워드병", 
            "Edwardsiellosis",
            "에드워드세균 감염",
            "#795548",
            "bacteria"
        )
    )
    
    // 모델 입력 텐서 크기
    const val INPUT_TENSOR_WIDTH = 640
    const val INPUT_TENSOR_HEIGHT = 640
    const val INPUT_TENSOR_CHANNELS = 3
    
    // VGG16 입력 크기
    const val VGG_INPUT_SIZE = 112
    
    // 신뢰도 임계값
    const val CONFIDENCE_THRESHOLD = 0.5f
    const val CONFIDENCE_THRESHOLD_LOW = 0.3f
    const val CONFIDENCE_THRESHOLD_HIGH = 0.8f
    
    // NMS IoU 임계값
    const val IOU_THRESHOLD = 0.45f
    
    /**
     * YOLO 클래스 이름 조회
     */
    fun getYoloClassName(id: Int, useEnglish: Boolean = false): String {
        return YOLO_CLASSES.find { it.id == id }?.let {
            if (useEnglish) it.nameEn else it.name
        } ?: "Unknown"
    }
    
    /**
     * VGG 클래스 이름 조회
     */
    fun getVggClassName(id: Int, useEnglish: Boolean = false): String {
        return VGG_CLASSES.find { it.id == id }?.let {
            if (useEnglish) it.nameEn else it.name
        } ?: "Unknown"
    }
    
    /**
     * VGG 클래스 정보 조회
     */
    fun getVggClassInfo(id: Int): ModelClassInfo? {
        return VGG_CLASSES.find { it.id == id }
    }
    
    /**
     * YOLO 클래스 정보 조회
     */
    fun getYoloClassInfo(id: Int): ModelClassInfo? {
        return YOLO_CLASSES.find { it.id == id }
    }
    
    /**
     * 클래스 색상 조회
     */
    fun getClassColor(id: Int, isVgg: Boolean = false): String {
        return (if (isVgg) VGG_CLASSES else YOLO_CLASSES)
            .find { it.id == id }?.color ?: "#999999"
    }
    
    /**
     * 클래스 아이콘 조회
     */
    fun getClassIcon(id: Int, isVgg: Boolean = false): String {
        return (if (isVgg) VGG_CLASSES else YOLO_CLASSES)
            .find { it.id == id }?.icon ?: "help-circle"
    }
    
    /**
     * 모든 VGG 클래스 ID 조회
     */
    fun getVggClassIds(): List<Int> {
        return VGG_CLASSES.map { it.id }
    }
    
    /**
     * 모든 YOLO 클래스 ID 조회
     */
    fun getYoloClassIds(): List<Int> {
        return YOLO_CLASSES.map { it.id }
    }
    
    /**
     * 질병 클래스 설명 조회
     */
    fun getDiseaseDescription(id: Int): String? {
        return VGG_CLASSES.find { it.id == id }?.description
    }
    
    /**
     * 새로운 질병 클래스 추가 (런타임)
     */
    fun addVggClass(classInfo: ModelClassInfo) {
        // 주의: 이 변경사항은 앱 재시작 시 초기화됨
        // 영구 저장을 위해서는 별도의 저장 메커니즘 필요
        val existingIndex = VGG_CLASSES.indexOfFirst { it.id == classInfo.id }
        if (existingIndex >= 0) {
            // 업데이트
            // 실제로는 불변 리스트이므로 새로운 리스트 생성 필요
        }
        // 새로운 클래스 추가는 ModelConfig 객체 재구성 필요
    }
}
