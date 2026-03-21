import { DiagnosisResult } from '../services/DiagnosisService';

/**
 * API 응답 타입
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

/**
 * 진단 결과 업로드 요청
 */
export interface UploadDiagnosisRequest {
  deviceId: string;
  diagnosis: DiagnosisResult;
  imageUrl?: string;
}

/**
 * 진단 기록 조회 응답
 */
export interface DiagnosisHistoryResponse {
  diagnoses: DiagnosisResult[];
  total: number;
  page: number;
  pageSize: number;
}

/**
 * 통계 정보 응답
 */
export interface StatisticsResponse {
  totalDiagnoses: number;
  diseaseDistribution: Record<string, number>;
  averageConfidence: number;
  recentDiagnoses: DiagnosisResult[];
}

/**
 * 사용자 정보
 */
export interface UserInfo {
  id: string;
  email: string;
  name: string;
  createdAt: number;
}

/**
 * API 설정
 */
const API_CONFIG = {
  baseUrl: process.env.REACT_NATIVE_API_BASE_URL || 'https://api.fdmsmartlens.com',
  timeout: 30000,
  version: 'v1',
};

/**
 * 백엔드 API 서비스
 */
export class ApiService {
  private static instance: ApiService;
  private authToken: string | null = null;
  private deviceId: string | null = null;

  private constructor() {}

  static getInstance(): ApiService {
    if (!ApiService.instance) {
      ApiService.instance = new ApiService();
    }
    return ApiService.instance;
  }

  /**
   * 장치 ID 설정
   */
  setDeviceId(id: string): void {
    this.deviceId = id;
  }

  /**
   * 인증 토큰 설정
   */
  setAuthToken(token: string): void {
    this.authToken = token;
  }

  /**
   * 인증 토큰 제거
   */
  clearAuthToken(): void {
    this.authToken = null;
  }

  /**
   * 진단 결과 업로드
   */
  async uploadDiagnosis(request: UploadDiagnosisRequest): Promise<ApiResponse<{ id: string }>> {
    return this.request<{ id: string }>(
      'POST',
      '/diagnoses',
      {
        ...request,
        deviceId: this.deviceId || request.deviceId,
      }
    );
  }

  /**
   * 진단 기록 조회
   */
  async getDiagnosisHistory(
    page: number = 1,
    pageSize: number = 20
  ): Promise<ApiResponse<DiagnosisHistoryResponse>> {
    return this.request<DiagnosisHistoryResponse>(
      'GET',
      `/diagnoses?page=${page}&pageSize=${pageSize}`
    );
  }

  /**
   * 특정 진단 기록 조회
   */
  async getDiagnosis(id: string): Promise<ApiResponse<DiagnosisResult>> {
    return this.request<DiagnosisResult>('GET', `/diagnoses/${id}`);
  }

  /**
   * 통계 정보 조회
   */
  async getStatistics(): Promise<ApiResponse<StatisticsResponse>> {
    return this.request<StatisticsResponse>('GET', '/statistics');
  }

  /**
   * 이미지 업로드
   */
  async uploadImage(imageUri: string): Promise<ApiResponse<{ imageUrl: string }>> {
    const formData = new FormData();
    
    // @ts-ignore - React Native FormData
    formData.append('image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: `diagnosis_${Date.now()}.jpg`,
    });

    return this.request<{ imageUrl: string }>('POST', '/images', formData);
  }

  /**
   * 사용자 정보 조회
   */
  async getUserInfo(): Promise<ApiResponse<UserInfo>> {
    return this.request<UserInfo>('GET', '/user');
  }

  /**
   * 일반 HTTP 요청
   */
  private async request<T>(
    method: string,
    path: string,
    body?: any
  ): Promise<ApiResponse<T>> {
    const url = `${API_CONFIG.baseUrl}/${API_CONFIG.version}${path}`;
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      return {
        success: true,
        data: data as T,
      };
    } catch (error) {
      console.error('[API] Request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * 파일 업로드 요청
   */
  private async uploadRequest<T>(
    method: string,
    path: string,
    formData: FormData
  ): Promise<ApiResponse<T>> {
    const url = `${API_CONFIG.baseUrl}/${API_CONFIG.version}${path}`;
    
    const headers: Record<string, string> = {};
    
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      return {
        success: true,
        data: data as T,
      };
    } catch (error) {
      console.error('[API] Upload failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed',
      };
    }
  }

  /**
   * API 연결 상태 확인
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_CONFIG.baseUrl}/health`, {
        method: 'GET',
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

export default ApiService;
