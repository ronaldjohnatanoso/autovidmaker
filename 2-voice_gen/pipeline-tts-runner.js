const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");
const { execSync, spawn } = require("child_process");


const killChromeScript = path.join(__dirname, "kill-chrome-debug.sh");
const launchChromeScript = path.join(__dirname, "load-chrome.sh");


function runShellScriptSync(scriptPath, name) {
  try {
    execSync(`bash "${scriptPath}"`, {
      stdio: "inherit",
      cwd: __dirname, // 🔧 Fix: ensure script path is resolved relative to this file
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
    console.error("Usage: node tts-runner.js <project_name>");
    process.exit(1);
  }

  // Paths setup
  // Script directory
  const scriptDir = __dirname;
  // One level up
  const baseDir = path.resolve(scriptDir, "..");
  // Project folder path: ../0-project-files/<projectName>
  const projectFolder = path.join(baseDir, "0-project-files", projectName);

  // Raw text file path: ../0-project-files/<projectName>/<projectName>_raw.txt
  const rawTextFile = path.join(projectFolder, `${projectName}_raw.txt`);
  if (!fs.existsSync(rawTextFile)) {
    console.error(`❌ Raw text file not found: ${rawTextFile}`);
    process.exit(1);
  }

  //find the config.json in the project folder
  const configFilePath = path.join(projectFolder, "config.json");
  if (!fs.existsSync(configFilePath)) {
    console.error(`❌ Config file not found: ${configFilePath}`);
    process.exit(1);
  }
  const config = JSON.parse(fs.readFileSync(configFilePath, "utf8"));
  const estimatedMilliseconds = config["stages"]["voice_gen"]["estimated_duration_ms"];

  // Output audio path: ../0-project-files/<projectName>/<projectName>.wav
  const outputAudioPath = path.join(projectFolder, `${projectName}.wav`);

  // Kill previous Chrome processes on debugging port
  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");

  // Launch Chrome headless in background, headless
  console.log("🚀 Launching Chrome...");
  const chromeProcess = spawn("bash", [launchChromeScript], {
    cwd: __dirname, // 🔧 Fix: make sure it runs from this folder
    detached: true,
    stdio: "ignore",
  });

  chromeProcess.unref();

  // Wait a bit for Chrome to start
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

  await page.goto("https://aistudio.google.com/generate-speech", {
    waitUntil: "networkidle2",
  });
  console.log("✅ Navigated to AI Studio");

  await delay(500);

  // Click model dropdown
  await page.waitForSelector("#model-selector > div", { visible: true });
  await page.click("#model-selector > div");
  console.log("🖱 Clicked model dropdown");

  // Select Gemini 2.5 Pro Preview TTS
  await page.waitForSelector("#mat-option-1 div.base-model-subtitle", {
    visible: true,
  });
  await page.click("#mat-option-1 div.base-model-subtitle");
  console.log("🎯 Selected Gemini 2.5 Pro Preview TTS");
  await delay(500);

  // Click first voice settings expansion panel toggle (Zephyr)
  await page.waitForSelector(
    "ms-voice-settings:nth-of-type(1) mat-expansion-panel > div path",
    { visible: true }
  );
  await page.click(
    "ms-voice-settings:nth-of-type(1) mat-expansion-panel > div path"
  );
  console.log("🖱 Clicked first voice settings expansion panel");

  // Select Enceladus voice (#mat-option-12)
  await page.waitForSelector("#mat-option-12 > span > div", { visible: true });
  await page.click("#mat-option-12 > span > div");
  console.log("🎯 Selected Enceladus voice");
  await delay(500);

  // Click second voice dropdown trigger (Puck)
  await page.waitForSelector(
    "ms-voice-settings:nth-of-type(2) mat-select-trigger",
    { visible: true }
  );
  await page.click("ms-voice-settings:nth-of-type(2) mat-select-trigger");
  console.log("🖱 Clicked second voice dropdown");

  // Select Gacrux voice (#mat-option-54)
  await page.waitForSelector("#mat-option-54 > span > div", { visible: true });
  await page.click("#mat-option-54 > span > div");
  console.log("🎯 Selected Gacrux voice");
  await delay(500);

  // Wait for textarea, clear and type raw text
  await page.waitForSelector("div > textarea", { visible: true });

  await page.evaluate(() => {
    const textarea = document.querySelector("div > textarea");
    if (textarea) textarea.value = "";
  });

  await page.click("div > textarea");
  console.log("Focused on text area");

  await delay(812);

  // Read raw text file content
  const textContent = fs.readFileSync(rawTextFile, "utf8");

  // Type the content directly into the textarea
  await page.type("div > textarea", textContent);

  await delay(521);

  // Click run button
  await page.waitForSelector("div.speech-prompt-footer button > span");
  await page.click("div.speech-prompt-footer button > span");
  console.log("Running TTS generation");

  //POSSIBLE PROBLEM
  try {
    // await page.waitForNetworkIdle({
    //   timeout: estimatedMilliseconds + 5000, // Wait for estimated duration + 10 seconds
    //   idleTime: 5000, // Wait for 5 seconds of network idle
    // })
    await page.waitForSelector("audio", { visible: true, timeout: estimatedMilliseconds + 30000 });
    console.log("🔊 Audio element found!");
  } catch (error) {
    console.error("❌ Error while waiting for audio element:", error);
    //save screenshot to the project folder
    console.log("📸 Saving error screenshot...");
    const screenshotPath = path.join(projectFolder, `${projectName}_error.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`🖼 Screenshot saved to ${screenshotPath}`);

    throw error;
  }

  const base64Data = await page.evaluate(() => {
    const audio = document.querySelector("audio");
    return audio?.src?.startsWith("data:") ? audio.src : null;
  });

  if (!base64Data) {
    console.log("⚠️ No base64 audio data found.");
    await browser.close();
    runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
    process.exit(1);
  }

  const matches = base64Data.match(/^data:audio\/\w+;base64,(.+)$/);
  if (!matches || matches.length !== 2) {
    console.error("❌ Invalid Base64 audio format.");
    await browser.close();
    runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
    process.exit(1);
  }

  const buffer = Buffer.from(matches[1], "base64");

  fs.writeFileSync(outputAudioPath, buffer);
  console.log(`✅ Audio saved to ${outputAudioPath}`);

  await browser.close();
  console.log("✅ Browser closed");


  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
})();
