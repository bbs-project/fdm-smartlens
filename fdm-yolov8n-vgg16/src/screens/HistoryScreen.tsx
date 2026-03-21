import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
  RefreshControl,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { DiagnosisService, DiagnosisResult } from '../services/DiagnosisService';

interface HistoryScreenProps {
  onSelectDiagnosis?: (diagnosis: DiagnosisResult) => void;
}

const HistoryScreen: React.FC<HistoryScreenProps> = ({ onSelectDiagnosis }) => {
  const [diagnoses, setDiagnoses] = useState<DiagnosisResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDiagnoses();
  }, []);

  const loadDiagnoses = async () => {
    const history = await DiagnosisService.getHistory();
    setDiagnoses(history);
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDiagnoses();
    setRefreshing(false);
  };

  const handleDelete = (id: string) => {
    Alert.alert(
      '진단 기록 삭제',
      '이 진단 기록을 삭제하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            await DiagnosisService.deleteDiagnosis(id);
            await loadDiagnoses();
          },
        },
      ]
    );
  };

  const handleClearAll = () => {
    Alert.alert(
      '전체 삭제',
      '모든 진단 기록을 삭제하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            await DiagnosisService.clearHistory();
            await loadDiagnoses();
          },
        },
      ]
    );
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getDiseaseIcon = (diseaseName: string) => {
    if (diseaseName.includes('정상')) return 'check-circle';
    if (diseaseName.includes('출혈')) return 'drop';
    if (diseaseName.includes('종양')) return 'circle-outline';
    if (diseaseName.includes('궤양')) return 'alert-circle';
    return 'fish';
  };

  const getDiseaseColor = (diseaseName: string) => {
    if (diseaseName.includes('정상')) return '#4CAF50';
    if (diseaseName.includes('출혈')) return '#F44336';
    if (diseaseName.includes('종양')) return '#FF9800';
    if (diseaseName.includes('궤양')) return '#9C27B0';
    return '#2196F3';
  };

  const renderDiagnosisItem = ({ item }: { item: DiagnosisResult }) => (
    <TouchableOpacity
      style={styles.card}
      onPress={() => onSelectDiagnosis?.(item)}
      onLongPress={() => handleDelete(item.id)}
    >
      <View style={styles.cardHeader}>
        <View
          style={[
            styles.iconContainer,
            { backgroundColor: getDiseaseColor(item.diseaseName) + '20' },
          ]}
        >
          <MaterialCommunityIcons
            name={getDiseaseIcon(item.diseaseName) as any}
            size={24}
            color={getDiseaseColor(item.diseaseName)}
          />
        </View>
        <View style={styles.cardContent}>
          <Text style={styles.diseaseName}>{item.diseaseName}</Text>
          <Text style={styles.timestamp}>{formatDate(item.timestamp)}</Text>
        </View>
        <View style={styles.confidenceContainer}>
          <Text style={styles.confidenceValue}>
            {(item.confidence * 100).toFixed(1)}%
          </Text>
          <Text style={styles.confidenceLabel}>신뢰도</Text>
        </View>
      </View>

      {item.symptoms.length > 0 && (
        <View style={styles.symptomsContainer}>
          <Text style={styles.symptomsLabel}>
            감지된 증상 ({item.symptoms.length}개)
          </Text>
          <View style={styles.symptomsList}>
            {item.symptoms.slice(0, 3).map((symptom, index) => (
              <View key={index} style={styles.symptomTag}>
                <Text style={styles.symptomText}>{symptom.name}</Text>
              </View>
            ))}
            {item.symptoms.length > 3 && (
              <Text style={styles.moreSymptoms}>
                +{item.symptoms.length - 3}개
              </Text>
            )}
          </View>
        </View>
      )}
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>진단 기록</Text>
        <TouchableOpacity onPress={handleClearAll} style={styles.clearButton}>
          <MaterialCommunityIcons name="delete-sweep" size={24} color="#666" />
        </TouchableOpacity>
      </View>

      {diagnoses.length === 0 ? (
        <View style={styles.emptyContainer}>
          <MaterialCommunityIcons name="history" size={64} color="#ccc" />
          <Text style={styles.emptyText}>진단 기록이 없습니다</Text>
          <Text style={styles.emptySubtext}>
            카메라로 질병을 진단하면 기록이 저장됩니다
          </Text>
        </View>
      ) : (
        <FlatList
          data={diagnoses}
          renderItem={renderDiagnosisItem}
          keyExtractor={item => item.id}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
          contentContainerStyle={styles.listContent}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
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
  clearButton: {
    padding: 8,
  },
  listContent: {
    padding: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardContent: {
    flex: 1,
    marginLeft: 12,
  },
  diseaseName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  timestamp: {
    fontSize: 12,
    color: '#999',
  },
  confidenceContainer: {
    alignItems: 'flex-end',
  },
  confidenceValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#4CAF50',
  },
  confidenceLabel: {
    fontSize: 10,
    color: '#999',
  },
  symptomsContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  symptomsLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  symptomsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  symptomTag: {
    backgroundColor: '#e3f2fd',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginRight: 6,
    marginBottom: 4,
  },
  symptomText: {
    fontSize: 11,
    color: '#1976d2',
  },
  moreSymptoms: {
    fontSize: 11,
    color: '#999',
    alignSelf: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    fontSize: 16,
    color: '#999',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#ccc',
    marginTop: 8,
    textAlign: 'center',
  },
});

export default HistoryScreen;
