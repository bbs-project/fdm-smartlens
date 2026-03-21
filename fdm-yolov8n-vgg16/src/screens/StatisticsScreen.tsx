import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { DiagnosisService } from '../services/DiagnosisService';

interface Statistics {
  totalDiagnoses: number;
  diseaseDistribution: Record<string, number>;
  averageConfidence: number;
  lastDiagnosisDate: number | null;
}

const StatisticsScreen: React.FC = () => {
  const [stats, setStats] = useState<Statistics | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    const statistics = await DiagnosisService.getStatistics();
    setStats(statistics);
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadStatistics();
    setRefreshing(false);
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return '방금 전';
    if (diffMins < 60) return `${diffMins}분 전`;
    if (diffHours < 24) return `${diffHours}시간 전`;
    if (diffDays < 7) return `${diffDays}일 전`;
    
    return date.toLocaleDateString('ko-KR', {
      month: 'long',
      day: 'numeric',
    });
  };

  const getDiseaseColor = (diseaseName: string) => {
    if (diseaseName.includes('정상')) return '#4CAF50';
    if (diseaseName.includes('출혈')) return '#F44336';
    if (diseaseName.includes('종양')) return '#FF9800';
    if (diseaseName.includes('궤양')) return '#9C27B0';
    if (diseaseName.includes('스쿠티카')) return '#00BCD4';
    if (diseaseName.includes('비브리오')) return '#E91E63';
    if (diseaseName.includes('에드워드')) return '#795548';
    return '#2196F3';
  };

  const getMaxDiseaseCount = () => {
    if (!stats || Object.keys(stats.diseaseDistribution).length === 0) return 1;
    return Math.max(...Object.values(stats.diseaseDistribution));
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.headerTitle}>통계 대시보드</Text>
      </View>

      {/* Summary Cards */}
      <View style={styles.summaryContainer}>
        <View style={styles.summaryCard}>
          <MaterialCommunityIcons name="clipboard-text" size={32} color="#2196F3" />
          <Text style={styles.summaryValue}>{stats?.totalDiagnoses || 0}</Text>
          <Text style={styles.summaryLabel}>총 진단 횟수</Text>
        </View>

        <View style={styles.summaryCard}>
          <MaterialCommunityIcons name="percent" size={32} color="#4CAF50" />
          <Text style={styles.summaryValue}>
            {stats ? (stats.averageConfidence * 100).toFixed(1) : 0}%
          </Text>
          <Text style={styles.summaryLabel}>평균 신뢰도</Text>
        </View>

        <View style={styles.summaryCard}>
          <MaterialCommunityIcons name="calendar-check" size={32} color="#FF9800" />
          <Text style={styles.summaryValue}>
            {stats?.lastDiagnosisDate
              ? formatDate(stats.lastDiagnosisDate)
              : '-'}
          </Text>
          <Text style={styles.summaryLabel}>마지막 진단</Text>
        </View>
      </View>

      {/* Disease Distribution */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>질병별 분포</Text>
        
        {!stats || Object.keys(stats.diseaseDistribution).length === 0 ? (
          <View style={styles.emptySection}>
            <MaterialCommunityIcons name="chart-bar" size={48} color="#ccc" />
            <Text style={styles.emptyText}>아직 통계 데이터가 없습니다</Text>
          </View>
        ) : (
          <View style={styles.distributionContainer}>
            {Object.entries(stats.diseaseDistribution)
              .sort((a, b) => b[1] - a[1])
              .map(([diseaseName, count]) => {
                const percentage = (count / stats.totalDiagnoses) * 100;
                const barWidth = (count / getMaxDiseaseCount()) * 100;
                const color = getDiseaseColor(diseaseName);

                return (
                  <View key={diseaseName} style={styles.distributionItem}>
                    <View style={styles.distributionHeader}>
                      <Text style={styles.distributionName} numberOfLines={1}>
                        {diseaseName}
                      </Text>
                      <Text style={styles.distributionCount}>{count}회</Text>
                    </View>
                    
                    <View style={styles.barContainer}>
                      <View
                        style={[
                          styles.bar,
                          {
                            width: `${barWidth}%`,
                            backgroundColor: color,
                          },
                        ]}
                      />
                    </View>
                    
                    <Text style={styles.distributionPercentage}>
                      {percentage.toFixed(1)}%
                    </Text>
                  </View>
                );
              })}
          </View>
        )}
      </View>

      {/* Quick Info */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>진단 가이드</Text>
        <View style={styles.infoContainer}>
          <View style={styles.infoItem}>
            <MaterialCommunityIcons name="information" size={20} color="#2196F3" />
            <Text style={styles.infoText}>
              정기적인 진단으로 질병을 조기에 발견하세요
            </Text>
          </View>
          <View style={styles.infoItem}>
            <MaterialCommunityIcons name="lightbulb" size={20} color="#FF9800" />
            <Text style={styles.infoText}>
              신뢰도가 80% 이상이면 결과를 신뢰할 수 있습니다
            </Text>
          </View>
          <View style={styles.infoItem}>
            <MaterialCommunityIcons name="alert-circle" size={20} color="#F44336" />
            <Text style={styles.infoText}>
              이상 증상이 발견되면 전문가와 상담하세요
            </Text>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  summaryContainer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  summaryCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  summaryValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 8,
  },
  summaryLabel: {
    fontSize: 11,
    color: '#999',
    marginTop: 4,
    textAlign: 'center',
  },
  section: {
    backgroundColor: '#fff',
    padding: 16,
    marginTop: 12,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  distributionContainer: {
    gap: 16,
  },
  distributionItem: {
    gap: 8,
  },
  distributionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  distributionName: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  distributionCount: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  barContainer: {
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    borderRadius: 4,
  },
  distributionPercentage: {
    fontSize: 12,
    color: '#999',
    textAlign: 'right',
  },
  emptySection: {
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    fontSize: 14,
    color: '#999',
    marginTop: 12,
  },
  infoContainer: {
    gap: 12,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    padding: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#666',
    lineHeight: 20,
  },
});

export default StatisticsScreen;
