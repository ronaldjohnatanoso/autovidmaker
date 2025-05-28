const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");
const { execSync, spawn } = require("child_process");

const killChromeScript = path.join(__dirname, "kill-chrome-debug.sh");
const launchChromeScript = path.join(__dirname, "run-chrome.sh");

function runShellScriptSync(scriptPath, name) {
  try {
    execSync(`bash "${scriptPath}"`, { stdio: "inherit" });
  } catch (err) {
    console.error(`❌ Error running ${name}:`, err.message);
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

(async () => {
  // Step 1: Kill previous Chrome with debugging port
  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");

  // Step 2: Launch Chrome in background
  console.log("🚀 Launching Chrome...");
  const chromeProcess = spawn("bash", [launchChromeScript, "--headless"], {
    detached: true,
    stdio: "ignore",
  });
  chromeProcess.unref();

  // Step 3: Wait for Chrome to fully start
  console.log("⏳ Waiting for Chrome to start...");
  await delay(3000);

  const browser = await puppeteer.connect({
    browserURL: "http://localhost:9222",
    defaultViewport: null,
  });

  const rawStruct = fs.readFileSync(
    path.join(__dirname, "raw-struct.txt"),
    "utf8"
  );

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
  await page.waitForSelector("#model-selector > div", { visible: true });
  await page.click("#model-selector > div");
  console.log("🖱 Clicked model dropdown");

  await page.waitForSelector("#mat-option-1 div.base-model-subtitle", {
    visible: true,
  });
  await page.click("#mat-option-1 div.base-model-subtitle");
  console.log("🎯 Selected Gemini 2.5 Pro Preview TTS");
  await delay(500);

  await page.waitForSelector(
    "ms-voice-settings:nth-of-type(1) mat-expansion-panel > div path",
    { visible: true }
  );
  await page.click(
    "ms-voice-settings:nth-of-type(1) mat-expansion-panel > div path"
  );
  console.log("🖱 Clicked first voice settings expansion panel");

  await page.waitForSelector("#mat-option-12 > span > div", { visible: true });
  await page.click("#mat-option-12 > span > div");
  console.log("🎯 Selected Enceladus voice");
  await delay(500);

  await page.waitForSelector(
    "ms-voice-settings:nth-of-type(2) mat-select-trigger",
    { visible: true }
  );
  await page.click("ms-voice-settings:nth-of-type(2) mat-select-trigger");
  console.log("🖱 Clicked second voice dropdown");

  await page.waitForSelector("#mat-option-54 > span > div", { visible: true });
  await page.click("#mat-option-54 > span > div");
  console.log("🎯 Selected Gacrux voice");
  await delay(500);

  await page.waitForSelector("div > textarea", { visible: true });

  await page.evaluate(() => {
    const textarea = document.querySelector("div > textarea");
    if (textarea) textarea.value = "";
  });

  await page.click("div > textarea");
  console.log("focused on text area");

  await page.type("div > textarea", rawStruct);
  console.log("replaced input");

  await page.waitForSelector("div.speech-prompt-footer button > span");
  await page.click("div.speech-prompt-footer button > span");
  console.log("running");

  await page.screenshot({ path: "debug-screenshot.png", fullPage: true });
  console.log("🖼 Screenshot saved");

  try {
    await page.waitForSelector("audio", { timeout: 60000 });
    console.log("🔊 Audio element found!");
  } catch (error) {
    console.error("❌ Error while waiting for audio:", error);
    await page.screenshot({ path: "error-screenshot.png", fullPage: true });
    console.log("🖼 Error screenshot saved");
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
    return;
  }

  const matches = base64Data.match(/^data:audio\/\w+;base64,(.+)$/);
  if (!matches || matches.length !== 2) {
    console.error("❌ Invalid Base64 audio format.");
    await browser.close();
    runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
    return;
  }

  const buffer = Buffer.from(matches[1], "base64");
  fs.writeFileSync("downloaded-audio.wav", buffer);
  console.log("✅ Audio downloaded successfully.");

  await browser.close();
  console.log("✅ Browser closed");

  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
})();
