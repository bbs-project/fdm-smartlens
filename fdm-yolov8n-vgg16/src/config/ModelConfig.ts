/**
 * 모델 설정 및 클래스 정보 관리
 */

export interface ModelClassInfo {
  id: number;
  name: string;
  nameEn: string;
  description?: string;
  color: string;
  icon: string;
}

export interface ModelConfig {
  yoloClasses: ModelClassInfo[];
  vggClasses: ModelClassInfo[];
  inputTensorSize: number[];
  confidenceThreshold: number;
  iouThreshold: number;
}

/**
 * YOLOv8 증상 클래스
 */
export const YOLO_CLASSES: ModelClassInfo[] = [
  { id: 0, name: 'Bleeding', nameEn: 'Bleeding', color: '#F44336', icon: 'drop' },
  { id: 1, name: 'Corrosion', nameEn: 'Corrosion', color: '#FF9800', icon: '腐蚀' },
  { id: 2, name: 'Tumor', nameEn: 'Tumor', color: '#9C27B0', icon: 'circle-outline' },
  { id: 3, name: 'Ulcer', nameEn: 'Ulcer', color: '#E91E63', icon: 'alert-circle' },
  { id: 4, name: 'EyesSymptom', nameEn: 'Eye Symptom', color: '#2196F3', icon: 'eye' },
];

/**
 * VGG16 질병 클래스
 */
export const VGG_CLASSES: ModelClassInfo[] = [
  { 
    id: 1, 
    name: '바이러스성출혈성패혈증', 
    nameEn: 'Viral Hemorrhagic Septicemia',
    description: '바이러스성 출혈성 패혈증',
    color: '#F44336',
    icon: 'drop'
  },
  { 
    id: 2, 
    name: '림포시스티스병', 
    nameEn: 'Lymphocystis Disease',
    description: '림포시스티스 바이러스 감염',
    color: '#FF9800',
    icon: 'virus'
  },
  { 
    id: 6, 
    name: '여윔병', 
    nameEn: 'Streptococcosis',
    description: '연쇄상구균 감염',
    color: '#4CAF50',
    icon: 'bacteria'
  },
  { 
    id: 8, 
    name: '스쿠티카병', 
    nameEn: 'Scuticociliatosis',
    description: '스쿠티카 섬모충 감염',
    color: '#9C27B0',
    icon: 'bug'
  },
  { 
    id: 11, 
    name: '연쇄구균증', 
    nameEn: 'Streptococcosis',
    description: '연쇄구균 감염',
    color: '#00BCD4',
    icon: 'bacteria'
  },
  { 
    id: 13, 
    name: '비브리오병', 
    nameEn: 'Vibriosis',
    description: '비브리오 균 감염',
    color: '#E91E63',
    icon: 'bacteria'
  },
  { 
    id: 19, 
    name: '에드워드병', 
    nameEn: 'Edwardsiellosis',
    description: '에드워드세균 감염',
    color: '#795548',
    icon: 'bacteria'
  },
];

/**
 * 기본 모델 설정
 */
export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  yoloClasses: YOLO_CLASSES,
  vggClasses: VGG_CLASSES,
  inputTensorSize: [1, 3, 640, 640],
  confidenceThreshold: 0.5,
  iouThreshold: 0.45,
};

/**
 * 모델 클래스 유틸리티
 */
export class ModelUtils {
  /**
   * YOLO 클래스 이름 조회
   */
  static getYoloClassName(id: number, useEnglish = false): string {
    const cls = YOLO_CLASSES.find(c => c.id === id);
    return useEnglish ? cls?.nameEn || 'Unknown' : cls?.name || 'Unknown';
  }

  /**
   * VGG 클래스 이름 조회
   */
  static getVggClassName(id: number, useEnglish = false): string {
    const cls = VGG_CLASSES.find(c => c.id === id);
    return useEnglish ? cls?.nameEn || 'Unknown' : cls?.name || 'Unknown';
  }

  /**
   * VGG 클래스 정보 조회
   */
  static getVggClassInfo(id: number): ModelClassInfo | undefined {
    return VGG_CLASSES.find(c => c.id === id);
  }

  /**
   * 클래스 색상 조회
   */
  static getClassColor(id: number, isVgg = false): string {
    const classes = isVgg ? VGG_CLASSES : YOLO_CLASSES;
    return classes.find(c => c.id === id)?.color || '#999999';
  }

  /**
   * 클래스 아이콘 조회
   */
  static getClassIcon(id: number, isVgg = false): string {
    const classes = isVgg ? VGG_CLASSES : YOLO_CLASSES;
    return classes.find(c => c.id === id)?.icon || 'help-circle';
  }

  /**
   * 새로운 질병 클래스 추가
   */
  static addVggClass(classInfo: ModelClassInfo) {
    const existingIndex = VGG_CLASSES.findIndex(c => c.id === classInfo.id);
    if (existingIndex >= 0) {
      VGG_CLASSES[existingIndex] = classInfo;
    } else {
      VGG_CLASSES.push(classInfo);
    }
  }

  /**
   * 새로운 증상 클래스 추가
   */
  static addYoloClass(classInfo: ModelClassInfo) {
    const existingIndex = YOLO_CLASSES.findIndex(c => c.id === classInfo.id);
    if (existingIndex >= 0) {
      YOLO_CLASSES[existingIndex] = classInfo;
    } else {
      YOLO_CLASSES.push(classInfo);
    }
  }

  /**
   * 모든 VGG 클래스 ID 조회
   */
  static getVggClassIds(): number[] {
    return VGG_CLASSES.map(c => c.id);
  }

  /**
   * 클래스 매핑 정보 생성
   */
  static getClassMapping(isVgg = false): Record<number, string> {
    const classes = isVgg ? VGG_CLASSES : YOLO_CLASSES;
    return classes.reduce((acc, cls) => {
      acc[cls.id] = cls.name;
      return acc;
    }, {} as Record<number, string>);
  }
}

export default DEFAULT_MODEL_CONFIG;
