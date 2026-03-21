import React, { useState } from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import CameraView from '../CameraView';
import HistoryScreen from '../screens/HistoryScreen';
import StatisticsScreen from '../screens/StatisticsScreen';
import { yoloModelURI, vggModelURI } from '../modelHandler';
import * as tf from '@tensorflow/tfjs';
import { DiagnosisResult } from '../services/DiagnosisService';

type TabType = 'camera' | 'history' | 'statistics';

interface TabConfig {
  id: TabType;
  label: string;
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
}

const TABS: TabConfig[] = [
  { id: 'camera', label: '진단', icon: 'camera' },
  { id: 'history', label: '기록', icon: 'history' },
  { id: 'statistics', label: '통계', icon: 'chart-bar' },
];

const App = () => {
  const [activeTab, setActiveTab] = useState<TabType>('camera');
  const [yoloModel, setYoloModel] = useState<tf.GraphModel | null>(null);
  const [vggModel, setVggModel] = useState<tf.GraphModel | null>(null);
  const [loading, setLoading] = useState({ loading: true, progress: 0, error: null });

  const handleDiagnosisComplete = async (result: Omit<DiagnosisResult, 'id' | 'timestamp'>) => {
    // 진단 완료 시 자동으로 기록에 저장
    try {
      const { DiagnosisService } = await import('../services/DiagnosisService');
      await DiagnosisService.saveDiagnosis(result);
    } catch (error) {
      console.error('[App] Error saving diagnosis:', error);
    }
  };

  const renderScreen = () => {
    switch (activeTab) {
      case 'camera':
        return (
          <CameraView
            type="back"
            yoloModel={yoloModel}
            vggModel={vggModel}
            inputTensorSize={[1, 3, 640, 640]}
            config={{ threshold: 0.25 }}
            onDiagnosisComplete={handleDiagnosisComplete}
          />
        );
      case 'history':
        return <HistoryScreen onSelectDiagnosis={(diag) => console.log('Selected:', diag)} />;
      case 'statistics':
        return <StatisticsScreen />;
      default:
        return null;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        {renderScreen()}
      </View>
      
      <View style={styles.tabBar}>
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <TouchableOpacity
              key={tab.id}
              style={[styles.tab, isActive && styles.activeTab]}
              onPress={() => setActiveTab(tab.id)}
            >
              <MaterialCommunityIcons
                name={tab.icon}
                size={24}
                color={isActive ? '#2196F3' : '#999'}
              />
              <Text
                style={[
                  styles.tabLabel,
                  { color: isActive ? '#2196F3' : '#999' },
                ]}
              >
                {tab.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    paddingBottom: 20,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
  },
  activeTab: {
    backgroundColor: '#f5f5f5',
  },
  tabLabel: {
    fontSize: 12,
    marginTop: 4,
  },
});

export default App;
