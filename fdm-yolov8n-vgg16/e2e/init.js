const detox = require('detox');
const adapter = require('detox/integration/jest');

beforeAll(async () => {
  await detox.init();
}, 300000);

afterAll(async () => {
  await adapter.cleanup();
  await detox.cleanup();
});
