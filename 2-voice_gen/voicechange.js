const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");
const { execSync, spawn } = require("child_process");

const start_time = Date.now();
let end_time = null;

//telephone, megaphone, oldRadio
const VOICE_TYPE = "telephone"

const killChromeScript = path.join(__dirname, "kill-chrome-debug.sh");
const launchChromeScript = path.join(__dirname, "load-chrome.sh");

function runShellScriptSync(scriptPath, name) {
  try {
    execSync(`bash "${scriptPath}"`, {
      stdio: "inherit",
      cwd: __dirname,
    });
  } catch (err) {
    console.error(` Error running ${name}:`, err.message);
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

(async () => {
  const projectName = process.argv[2];

  if (!projectName) {
    console.error("Usage: node voicechange.js <project_name>");
    process.exit(1);
  }

  // Paths setup
  const scriptDir = __dirname;
  const baseDir = path.resolve(scriptDir, "..");
  const projectFolder = path.join(baseDir, "0-project-files", projectName);

  // Kill previous Chrome processes on debugging port
  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");

  // Launch Chrome headless in background
  console.log("🚀 Launching Chrome...");
  const chromeProcess = spawn("bash", [launchChromeScript], {
    cwd: __dirname,
    detached: true,
    stdio: "ignore",
  });

  chromeProcess.unref();

  // Wait for Chrome to start
  console.log("⏳ Waiting for Chrome to start...");
  await delay(3000);

  const browser = await puppeteer.connect({
    browserURL: "http://localhost:9222",
    defaultViewport: null,
  });

  const pages = await browser.pages();
  const page = pages.length ? pages[0] : await browser.newPage();

  await page.setUserAgent(
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
  );

  await page.goto("https://voicechanger.io", {
    waitUntil: "networkidle2",
  });
  console.log("✅ Navigated to VoiceChanger.io");

  await page.waitForNetworkIdle();

  // Find the input[type="file"] element
  const inputUploadHandle = await page.$('input[type="file"]');

  const wavFilePath = path.join(projectFolder, `${projectName}.wav`);

  await inputUploadHandle.uploadFile(wavFilePath);

  await page.waitForNetworkIdle();
  console.log("✅ Uploaded audio file");

  // Find and click the voice card
  const cardClicked = await page.evaluate((voiceType) => {
    const cards = document.querySelectorAll('.voice-card');
    for (const card of cards) {
      if (card.onclick && card.onclick.toString().includes(`'${voiceType}'`)) {
        card.click();
        return true;
      }
    }
    return false;
  }, VOICE_TYPE);

  if (cardClicked) {
    console.log(`✅ Clicked voice effect: ${VOICE_TYPE}`);
  } else {
    console.error(`Voice effect '${VOICE_TYPE}' not found.`);
  }

  // Wait for the audio to be processed and appear
  console.log("⏳ Waiting for processed audio...");
  await page.waitForSelector('#output-audio-tag', { timeout: 60000 });
  console.log("✅ Audio element appeared");

  // Wait for the blob URL to be available in the src attribute
  console.log("⏳ Waiting for blob URL...");
  const blobUrl = await page.waitForFunction(() => {
    const audioElement = document.querySelector('#output-audio-tag');
    if (!audioElement) return false;
    console.log('Current src:', audioElement.src);
    return audioElement.src && audioElement.src.startsWith('blob:');
  }, { timeout: 30000 }).then(async () => {
    return await page.evaluate(() => {
      const audioElement = document.querySelector('#output-audio-tag');
      return audioElement.src;
    });
  }).catch(async () => {
    // Debug what we actually have
    const debugInfo = await page.evaluate(() => {
      const audioElement = document.querySelector('#output-audio-tag');
      return {
        found: !!audioElement,
        src: audioElement ? audioElement.src : null,
        hasControls: audioElement ? audioElement.hasAttribute('controls') : false,
        outerHTML: audioElement ? audioElement.outerHTML : null
      };
    });
    console.log('Debug info:', debugInfo);
    return null;
  });

  if (!blobUrl || !blobUrl.startsWith('blob:')) {
    console.error("❌ Could not find processed audio blob URL");
    process.exit(1);
  }

  console.log("✅ Found blob URL:", blobUrl);

  console.log("📥 Downloading processed audio...");

  // Download the blob as buffer
  const audioBuffer = await page.evaluate(async (url) => {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return Array.from(new Uint8Array(arrayBuffer));
  }, blobUrl);

  // Rename original file to _original.wav
  const originalWavPath = path.join(projectFolder, `${projectName}.wav`);
  const backupWavPath = path.join(projectFolder, `${projectName}_original.wav`);
  
  if (fs.existsSync(originalWavPath)) {
    fs.renameSync(originalWavPath, backupWavPath);
    console.log(`✅ Renamed original file to ${projectName}_original.wav`);
  }

  // Save the new processed audio as projectName.wav
  const newAudioBuffer = Buffer.from(audioBuffer);
  fs.writeFileSync(originalWavPath, newAudioBuffer);
  console.log(`✅ Saved processed audio as ${projectName}.wav`);


  await browser.close();
  
  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
  end_time = Date.now();
  console.log(`⏱️ Total time taken: ${(end_time - start_time) / 1000} seconds`);
})();
