export const translations = {
  ko: {
    // Common
    app_name: '스마트 렌즈',
    cancel: '취소',
    confirm: '확인',
    delete: '삭제',
    save: '저장',
    close: '닫기',
    settings: '설정',
    
    // Camera/Diagnosis
    camera: '카메라',
    diagnosis: '진단',
    detecting: '감지 중...',
    no_detection: '감지된 객체가 없습니다',
    confidence: '신뢰도',
    symptoms: '증상',
    disease: '질병',
    normal: '정상',
    
    // History
    history: '기록',
    no_history: '진단 기록이 없습니다',
    history_description: '카메라로 질병을 진단하면 기록이 저장됩니다',
    delete_history: '진단 기록 삭제',
    delete_history_confirm: '이 진단 기록을 삭제하시겠습니까?',
    clear_all: '전체 삭제',
    clear_all_confirm: '모든 진단 기록을 삭제하시겠습니까?',
    detected_symptoms: '감지된 증상 ({} 개)',
    more_symptoms: '+{} 개',
    
    // Statistics
    statistics: '통계',
    total_diagnoses: '총 진단 횟수',
    average_confidence: '평균 신뢰도',
    last_diagnosis: '마지막 진단',
    disease_distribution: '질병별 분포',
    no_statistics: '아직 통계 데이터가 없습니다',
    diagnosis_guide: '진단 가이드',
    guide_1: '정기적인 진단으로 질병을 조기에 발견하세요',
    guide_2: '신뢰도가 80% 이상이면 결과를 신뢰할 수 있습니다',
    guide_3: '이상 증상이 발견되면 전문가와 상담하세요',
    
    // Time
    just_now: '방금 전',
    minutes_ago: '{} 분 전',
    hours_ago: '{} 시간 전',
    days_ago: '{} 일 전',
    
    // Messages
    camera_permission_denied: '카메라 권한이 거부되었습니다',
    loading_model: '모델 로딩 중...',
    model_load_complete: '모델 로딩 완료',
    diagnosis_saved: '진단 결과가 저장되었습니다',
    diagnosis_save_failed: '진단 결과 저장 실패',
  },
  en: {
    // Common
    app_name: 'Smart Lens',
    cancel: 'Cancel',
    confirm: 'Confirm',
    delete: 'Delete',
    save: 'Save',
    close: 'Close',
    settings: 'Settings',
    
    // Camera/Diagnosis
    camera: 'Camera',
    diagnosis: 'Diagnosis',
    detecting: 'Detecting...',
    no_detection: 'No objects detected',
    confidence: 'Confidence',
    symptoms: 'Symptoms',
    disease: 'Disease',
    normal: 'Normal',
    
    // History
    history: 'History',
    no_history: 'No diagnosis history',
    history_description: 'Diagnosis records will be saved when you diagnose diseases with the camera',
    delete_history: 'Delete Diagnosis Record',
    delete_history_confirm: 'Are you sure you want to delete this diagnosis record?',
    clear_all: 'Clear All',
    clear_all_confirm: 'Are you sure you want to delete all diagnosis records?',
    detected_symptoms: 'Detected Symptoms ({} items)',
    more_symptoms: '+{} more',
    
    // Statistics
    statistics: 'Statistics',
    total_diagnoses: 'Total Diagnoses',
    average_confidence: 'Average Confidence',
    last_diagnosis: 'Last Diagnosis',
    disease_distribution: 'Disease Distribution',
    no_statistics: 'No statistics data yet',
    diagnosis_guide: 'Diagnosis Guide',
    guide_1: 'Detect diseases early through regular diagnosis',
    guide_2: 'Results are reliable when confidence is above 80%',
    guide_3: 'Consult an expert if abnormal symptoms are found',
    
    // Time
    just_now: 'Just now',
    minutes_ago: '{} min ago',
    hours_ago: '{} hours ago',
    days_ago: '{} days ago',
    
    // Messages
    camera_permission_denied: 'Camera permission denied',
    loading_model: 'Loading model...',
    model_load_complete: 'Model loaded successfully',
    diagnosis_saved: 'Diagnosis result saved',
    diagnosis_save_failed: 'Failed to save diagnosis result',
  },
};

export type Language = 'ko' | 'en';
export type TranslationKey = keyof typeof translations.ko;

export class i18n {
  private static currentLanguage: Language = 'ko';

  static setLanguage(lang: Language) {
    this.currentLanguage = lang;
  }

  static getLanguage(): Language {
    return this.currentLanguage;
  }

  static t(key: TranslationKey, params?: Record<string, string | number>): string {
    let text = translations[this.currentLanguage][key] || translations.ko[key] || key;
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        text = text.replace(`{${key}}`, String(value));
      });
    }
    
    return text;
  }

  static getDiseaseName(koreanName: string): string {
    const diseaseMap: Record<string, string> = {
      '정상': 'Normal',
      '바이러스성출혈성패혈증': 'Viral Hemorrhagic Septicemia',
      '림포시스티스병': 'Lymphocystis Disease',
      '여윔병': 'Streptococcosis',
      '스쿠티카병': 'Scuticociliatosis',
      '연쇄구균증': 'Streptococcosis',
      '비브리오병': 'Vibriosis',
      '에드워드병': 'Edwardsiellosis',
    };

    if (this.currentLanguage === 'en') {
      return diseaseMap[koreanName] || koreanName;
    }
    return koreanName;
  }

  static getSymptomName(koreanName: string): string {
    const symptomMap: Record<string, string> = {
      'Bleeding': 'Bleeding',
      'Corrosion': 'Corrosion',
      'Tumor': 'Tumor',
      'Ulcer': 'Ulcer',
      'EyesSymptom': 'Eye Symptom',
    };

    if (this.currentLanguage === 'en') {
      return symptomMap[koreanName] || koreanName;
    }
    return koreanName;
  }
}

export default i18n;
