const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');


(async () => {
  const browser = await puppeteer.connect({
    browserURL: 'http://localhost:9222',
    defaultViewport: null,
  });


// Simple wait helper function
function delay(time) {
  return new Promise(resolve => setTimeout(resolve, time));
}

// Read raw-struct.txt from the same directory
const rawStruct = fs.readFileSync(path.join(__dirname, 'raw-struct.txt'), 'utf8');


  const pages = await browser.pages();
  const page = pages.length ? pages[0] : await browser.newPage();

  // await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36');
// await page.setViewport({ width: 1920, height: 1080 });


  // 1️⃣ Navigate to Google AI Studio
  await page.goto('https://aistudio.google.com/generate-speech', {
    waitUntil: 'networkidle2',
  });
  console.log("✅ Navigated to AI Studio");

  await delay(500)

  // 2️⃣ Click model dropdown
  await page.waitForSelector('#model-selector > div', { visible: true });
  await page.click('#model-selector > div');
  console.log("🖱 Clicked model dropdown");

  // 3️⃣ Select Gemini 2.5 Pro Preview TTS
  await page.waitForSelector('#mat-option-1 div.base-model-subtitle', { visible: true });
  await page.click('#mat-option-1 div.base-model-subtitle');
  console.log("🎯 Selected Gemini 2.5 Pro Preview TTS");
  await delay(500)

  // 1️⃣ Click first voice settings expansion panel toggle (Zephyr)
  await page.waitForSelector('ms-voice-settings:nth-of-type(1) mat-expansion-panel > div path', { visible: true });
  await page.click('ms-voice-settings:nth-of-type(1) mat-expansion-panel > div path');
  console.log('🖱 Clicked first voice settings expansion panel');

  // 2️⃣ Select Enceladus voice (#mat-option-12)
  await page.waitForSelector('#mat-option-12 > span > div', { visible: true });
  await page.click('#mat-option-12 > span > div');
  console.log('🎯 Selected Enceladus voice');
  await delay(500)
  // 3️⃣ Click second voice dropdown trigger (Puck)
  await page.waitForSelector('ms-voice-settings:nth-of-type(2) mat-select-trigger', { visible: true });
  await page.click('ms-voice-settings:nth-of-type(2) mat-select-trigger');
  console.log('🖱 Clicked second voice dropdown');

  // 4️⃣ Select Gacrux voice (#mat-option-54)
  await page.waitForSelector('#mat-option-54 > span > div', { visible: true });
  await page.click('#mat-option-54 > span > div');
  console.log('🎯 Selected Gacrux voice');
  await delay(500)

  //wait for textarea
  await page.waitForSelector('div > textarea', { visible: true });

  //set the textarea to empty
  await page.evaluate(() => {
    const textarea = document.querySelector('div > textarea');
    if (textarea) textarea.value = ''; // set to empty or any string
  });

  //focus on the text area
  await page.click('div > textarea');
  console.log("focused on text area")

  //type in text area
  await page.type('div > textarea', rawStruct);
  console.log("replaced input")

  //run button
  await page.waitForSelector('div.speech-prompt-footer button > span')
  await page.click('div.speech-prompt-footer button > span')
  console.log("running")

  await page.screenshot({ path: 'debug-screenshot.png', fullPage: true });
console.log('Screenshot saved!2');

try {
  await page.waitForSelector('audio', { timeout: 60000 * 1 });
  console.log('Audio element found!');
} catch (error) {
  console.error('❌ Error while waiting for audio element:', error);

  // Always take a screenshot on any error
  await page.screenshot({ path: 'error-screenshot.png', fullPage: true });
  console.log('🖼️ Screenshot saved as error-screenshot.png');




  throw error; // optionally rethrow if you want to fail the script
}

  // Extract the base64 data from the audio src
  const base64Data = await page.evaluate(() => {
    const audio = document.querySelector('audio');
    if (!audio || !audio.src.startsWith('data:')) return null;
    return audio.src;
  });

  if (!base64Data) {
    console.log('No base64 audio data found.');
    await browser.close();
    return;
  }

  // Parse the Base64 from the data URI
  const matches = base64Data.match(/^data:audio\/\w+;base64,(.+)$/);
  if (!matches || matches.length !== 2) {
    console.error('Invalid Base64 audio format.');
    await browser.close();
    return;
  }

  const buffer = Buffer.from(matches[1], 'base64');
  fs.writeFileSync('downloaded-audio.wav', buffer); // change extension if needed

  console.log('Audio downloaded successfully.');


  // 💤 Keep the browser open
  process.stdin.resume();
})();
