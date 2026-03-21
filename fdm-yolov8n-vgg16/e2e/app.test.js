const { device, element, by, expect, waitFor } = require('detox');

describe('FDM SmartLens E2E Tests', () => {
  beforeAll(async () => {
    await device.launchApp({
      permissions: { camera: 'YES' },
    });
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  afterAll(async () => {
    await device.terminateApp();
  });

  describe('App Launch', () => {
    it('should show loading screen on launch', async () => {
      await expect(element(by.text(/Loading models.../))).toBeVisible();
    });

    it('should show camera permission request', async () => {
      await waitFor(element(by.type('UIAlertController')))
        .toBeVisible()
        .withTimeout(5000);
    });
  });

  describe('Camera View', () => {
    it('should display camera preview after loading', async () => {
      // Wait for models to load
      await waitFor(element(by.id('camera-view')))
        .toBeVisible()
        .withTimeout(60000);
    });

    it('should show flip camera button', async () => {
      await expect(element(by.text('Flip Camera'))).toBeVisible();
    });

    it('should flip camera when button is tapped', async () => {
      const flipButton = element(by.text('Flip Camera'));
      await flipButton.tap();
      
      // Add assertion for camera state change if available
    });
  });

  describe('Model Loading', () => {
    it('should show progress during model loading', async () => {
      const loadingText = element(by.text(/Loading models... \d+%/));
      await expect(loadingText).toBeVisible();
    });

    it('should complete model loading within timeout', async () => {
      await waitFor(element(by.id('camera-view')))
        .toBeVisible()
        .withTimeout(120000);
    });
  });

  describe('Error Handling', () => {
    it('should handle camera permission denial gracefully', async () => {
      await device.denyPermissions();
      await expect(element(by.text(/permission/i))).toBeVisible();
    });
  });
});
