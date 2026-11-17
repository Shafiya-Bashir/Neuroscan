const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  console.log("Launching Chrome...");
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 960, deviceScaleFactor: 2 });

  console.log("Navigating to http://localhost:3000...");
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0', timeout: 30000 });

  // Wait for the patient buttons to be present
  await page.waitForSelector('button');

  const screenshotsDir = path.join(__dirname, 'public', 'reports', 'screenshots');
  const figuresDir = path.join(__dirname, 'reports', 'figures', 'screenshots');
  fs.mkdirSync(screenshotsDir, { recursive: true });
  fs.mkdirSync(figuresDir, { recursive: true });

  // 1. Capture Glioma (Case 0)
  console.log("Capturing 1/5: High-Grade Glioma...");
  // Find button containing "Glioblastoma" or click the first patient button
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && text.includes('Glioblastoma')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 1000));
  const gliomaPath = path.join(screenshotsDir, 'ui_glioma_diagnostic.png');
  await page.screenshot({ path: gliomaPath });
  fs.copyFileSync(gliomaPath, path.join(figuresDir, 'ui_glioma_diagnostic.png'));
  console.log("✓ Saved Glioma screenshot");

  // 2. Capture Meningioma (Case 1)
  console.log("Capturing 2/5: Atypical Meningioma...");
  const buttons2 = await page.$$('button');
  for (const btn of buttons2) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && text.includes('Meningioma')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 1000));
  const menPath = path.join(screenshotsDir, 'ui_meningioma_diagnostic.png');
  await page.screenshot({ path: menPath });
  fs.copyFileSync(menPath, path.join(figuresDir, 'ui_meningioma_diagnostic.png'));
  console.log("✓ Saved Meningioma screenshot");

  // 3. Capture Pituitary (Case 2)
  console.log("Capturing 3/5: Pituitary Macroadenoma...");
  const buttons3 = await page.$$('button');
  for (const btn of buttons3) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && text.includes('Pituitary')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 1000));
  const pitPath = path.join(screenshotsDir, 'ui_pituitary_diagnostic.png');
  await page.screenshot({ path: pitPath });
  fs.copyFileSync(pitPath, path.join(figuresDir, 'ui_pituitary_diagnostic.png'));
  console.log("✓ Saved Pituitary screenshot");

  // 4. Capture Healthy (Case 3)
  console.log("Capturing 4/5: Normal Healthy Brain...");
  const buttons4 = await page.$$('button');
  for (const btn of buttons4) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && (text.includes('Normal') || text.includes('Healthy'))) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 1000));
  const heaPath = path.join(screenshotsDir, 'ui_healthy_diagnostic.png');
  await page.screenshot({ path: heaPath });
  fs.copyFileSync(heaPath, path.join(figuresDir, 'ui_healthy_diagnostic.png'));
  console.log("✓ Saved Healthy screenshot");

  // 5. Capture Difference Map Mode
  console.log("Capturing 5/5: Difference Map Mode...");
  // First click Glioblastoma again
  const buttons5 = await page.$$('button');
  for (const btn of buttons5) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && text.includes('Glioblastoma')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 500));
  // Then click "Difference Map" button
  const modeButtons = await page.$$('button');
  for (const btn of modeButtons) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && text.includes('Difference Map')) {
      await btn.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 1000));
  const diffPath = path.join(screenshotsDir, 'ui_difference_map_mode.png');
  await page.screenshot({ path: diffPath });
  fs.copyFileSync(diffPath, path.join(figuresDir, 'ui_difference_map_mode.png'));
  console.log("✓ Saved Difference Map screenshot");

  await browser.close();
  console.log("All 5 distinct screenshots successfully captured!");
})();
