import { defineConfig } from '@playwright/test';

/**
 * Milestone 1 runs pure Node assertions over the golden dataset, so no browser
 * project is declared yet. UI projects get added when the UI adapter lands.
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: Boolean(process.env['CI']),
  retries: 0,
  reporter: process.env['CI'] ? [['list'], ['html', { open: 'never' }]] : [['list']],
});