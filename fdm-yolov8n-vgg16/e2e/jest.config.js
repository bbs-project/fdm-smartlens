module.exports = {
  preset: 'react-native',
  rootDir: '..',
  testMatch: ['<rootDir>/e2e/**/*.test.js'],
  testTimeout: 120000,
  maxWorkers: 1,
  setupFilesAfterEnv: ['./e2e/init.js'],
  skipNodesDetection: true,
};
