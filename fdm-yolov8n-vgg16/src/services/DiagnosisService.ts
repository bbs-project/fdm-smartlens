import AsyncStorage from '@react-native-async-storage/async-storage';

export interface DiagnosisResult {
  id: string;
  timestamp: number;
  diseaseName: string;
  diseaseCode: number;
  confidence: number;
  symptoms: Array<{
    name: string;
    confidence: number;
    boundingBox: { x1: number; y1: number; x2: number; y2: number };
  }>;
  imageUrl?: string;
  notes?: string;
}

const DIAGNOSIS_HISTORY_KEY = '@fdm_smartlens:diagnosis_history';
const MAX_HISTORY_ITEMS = 100;

/**
 * 진단 결과 저장 서비스
 */
export class DiagnosisService {
  /**
   * 진단 결과를 저장합니다
   */
  static async saveDiagnosis(result: Omit<DiagnosisResult, 'id' | 'timestamp'>): Promise<DiagnosisResult> {
    const newResult: DiagnosisResult = {
      ...result,
      id: this.generateId(),
      timestamp: Date.now(),
    };

    try {
      const history = await this.getHistory();
      history.unshift(newResult); // Add to beginning

      // Limit history size
      if (history.length > MAX_HISTORY_ITEMS) {
        history.splice(MAX_HISTORY_ITEMS);
      }

      await AsyncStorage.setItem(DIAGNOSIS_HISTORY_KEY, JSON.stringify(history));
      return newResult;
    } catch (error) {
      console.error('[DiagnosisService] Error saving diagnosis:', error);
      throw error;
    }
  }

  /**
   * 전체 진단 기록을 조회합니다
   */
  static async getHistory(): Promise<DiagnosisResult[]> {
    try {
      const data = await AsyncStorage.getItem(DIAGNOSIS_HISTORY_KEY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('[DiagnosisService] Error getting history:', error);
      return [];
    }
  }

  /**
   * 특정 진단 기록을 조회합니다
   */
  static async getDiagnosis(id: string): Promise<DiagnosisResult | null> {
    try {
      const history = await this.getHistory();
      return history.find(item => item.id === id) || null;
    } catch (error) {
      console.error('[DiagnosisService] Error getting diagnosis:', error);
      return null;
    }
  }

  /**
   * 진단 기록을 삭제합니다
   */
  static async deleteDiagnosis(id: string): Promise<boolean> {
    try {
      const history = await this.getHistory();
      const filtered = history.filter(item => item.id !== id);
      await AsyncStorage.setItem(DIAGNOSIS_HISTORY_KEY, JSON.stringify(filtered));
      return true;
    } catch (error) {
      console.error('[DiagnosisService] Error deleting diagnosis:', error);
      return false;
    }
  }

  /**
   * 전체 진단 기록을 삭제합니다
   */
  static async clearHistory(): Promise<boolean> {
    try {
      await AsyncStorage.removeItem(DIAGNOSIS_HISTORY_KEY);
      return true;
    } catch (error) {
      console.error('[DiagnosisService] Error clearing history:', error);
      return false;
    }
  }

  /**
   * 최근 진단 결과를 조회합니다
   */
  static async getRecentDiagnosis(limit: number = 10): Promise<DiagnosisResult[]> {
    const history = await this.getHistory();
    return history.slice(0, limit);
  }

  /**
   * 통계 정보를 조회합니다
   */
  static async getStatistics(): Promise<{
    totalDiagnoses: number;
    diseaseDistribution: Record<string, number>;
    averageConfidence: number;
    lastDiagnosisDate: number | null;
  }> {
    const history = await this.getHistory();

    if (history.length === 0) {
      return {
        totalDiagnoses: 0,
        diseaseDistribution: {},
        averageConfidence: 0,
        lastDiagnosisDate: null,
      };
    }

    const diseaseDistribution: Record<string, number> = {};
    let totalConfidence = 0;

    history.forEach(item => {
      // Count disease distribution
      diseaseDistribution[item.diseaseName] = (diseaseDistribution[item.diseaseName] || 0) + 1;
      totalConfidence += item.confidence;
    });

    return {
      totalDiagnoses: history.length,
      diseaseDistribution,
      averageConfidence: totalConfidence / history.length,
      lastDiagnosisDate: history[0]?.timestamp || null,
    };
  }

  /**
   * 고유 ID 생성
   */
  private static generateId(): string {
    return `diag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 진단 결과를 내보냅니다 (JSON)
   */
  static async exportHistory(): Promise<string> {
    const history = await this.getHistory();
    return JSON.stringify(history, null, 2);
  }

  /**
   * 진단 결과를 가져옵니다 (JSON)
   */
  static async importHistory(jsonData: string): Promise<boolean> {
    try {
      const history = JSON.parse(jsonData);
      if (!Array.isArray(history)) {
        throw new Error('Invalid format');
      }
      await AsyncStorage.setItem(DIAGNOSIS_HISTORY_KEY, JSON.stringify(history));
      return true;
    } catch (error) {
      console.error('[DiagnosisService] Error importing history:', error);
      return false;
    }
  }
}
