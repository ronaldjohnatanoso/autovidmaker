const puppeteer = require("puppeteer-core");
const { execSync, spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

// Shell scripts
const killChromeScript = path.join(__dirname, "kill-chrome-debug.sh");
const launchChromeScript = path.join(__dirname, "load-chrome.sh");

// Get project name from CLI
const projectName = process.argv[2];
if (!projectName) {
  console.error("❌ Usage: node minimal-puppeteer.js <project_name>");
  process.exit(1);
}
console.log(`📁 Project: ${projectName}`);

// Utility to run a shell script synchronously
function runShellScriptSync(scriptPath, name) {
  try {
    execSync(`bash "${scriptPath}"`, {
      stdio: "inherit",
      cwd: __dirname,
    });
  } catch (err) {
    console.error(`Error running ${name}:`, err.message);
  }
}

// Delay helper
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
function randomDelay(min = 40, max = 120) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

const ORIENTATION = '16:9'; // Set the desired orientation here

const jsonPath = path.join(
  __dirname,
  "../0-project-files",
  projectName,
  `${projectName}_img_prompts.json`
);
console.log(`📄 JSON Path: ${jsonPath}`);

const imagesPath = path.join(path.dirname(jsonPath), "images");
console.log(`📁 images Path: ${imagesPath}`);
try {
  fs.mkdirSync(imagesPath, { recursive: true });
  console.log(`📁 Created subfolder for images: ${imagesPath}`);
} catch (error) {
  console.error("Error creating subfolder path:", error.message);
}

const raw = fs.readFileSync(jsonPath, "utf8");
const prompts = JSON.parse(raw);

const MAX_TIMEOUTS = 3; //maximum number of timeouts before giving up

(async () => {
  // Step 1: Kill any previous Chrome on debugging port
  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");

  // Step 2: Launch Chrome
  console.log("🚀 Launching Chrome...");
  const chromeProcess = spawn("bash", [launchChromeScript, "--headless"], {
    cwd: __dirname,
    detached: true,
    stdio: "ignore",
  });
  chromeProcess.unref();

  // Step 3: Wait for Chrome to initialize
  console.log("⏳ Waiting for Chrome to start...");
  await delay(3000);

  // Step 4: Connect via Puppeteer
  const browser = await puppeteer.connect({
    browserURL: "http://localhost:9222",
    defaultViewport: null,
  });

  // Step 5: Open a new tab
  const page = (await browser.pages())[0] || (await browser.newPage());
  await page.goto("http://www.meta.ai", {
    waitUntil: "networkidle2",
  });

  // Check if prompts are already done
  let timeoutCount = 0;
  let isFirstPrompt = true;
  while (true) {
    const prompt = prompts.find((p) => p["gen-status"] !== "done");
    if (!prompt) {
      console.log("✅ All prompts processed!");
      break;
    }

    // Set the prompt status to "in-progress"
    prompt["gen-status"] = "in-progress";
    prompt["gen-start"] = new Date().toISOString();
    fs.writeFileSync(jsonPath, JSON.stringify(prompts, null, 2));
    console.log(`📝 Updated prompt status to "in-progress"`);

    const pPromise = page
      .waitForSelector("p", { visible: true })
      .then((el) => ({ el, selector: "p" }))
      .catch(() => null);

    const textareaPromise = page
      .waitForSelector("textarea", { visible: true })
      .then((el) => ({ el, selector: "textarea" }))
      .catch(() => null);

    const inputTextRaceResult = await Promise.race([pPromise, textareaPromise]);

    if (inputTextRaceResult) {
      const { el, selector } = inputTextRaceResult;
      await el.click();
      console.log(`🖱️ Clicked element: ${selector}`);
    } else {
      throw new Error("❌ Neither <p> nor <textarea> appeared within timeout");
    }

    // Select all text
    await page.keyboard.down("Control");
    await page.keyboard.press("A");
    await page.keyboard.up("Control");

    // Delete selected text
    await page.keyboard.press("Backspace");

    let text = prompt["image_prompt"];

    await delay(520); // Wait a bit to see the text
    //img promts
    await page.keyboard.type("Imagine " + text);
    console.log(`📝 Entered prompt: ${text}`);

    //check img size first
    //do a promise race on three selectors
    const o1_1Promise = page
      .waitForSelector("text/1:1", { visible: true })
      .then((el) => ({ el, selector: "1:1" }))
      .catch(() => null);
    const o9_16Promise = page
      .waitForSelector("text/9:16", { visible: true })
      .then((el) => ({ el, selector: "9:16" }))
      .catch(() => null);
    const o16_9Promise = page
      .waitForSelector("text/16:9", { visible: true })
      .then((el) => ({ el, selector: "16:19" }))
      .catch(() => null);

    if (isFirstPrompt) {
      const orientationPromise = await Promise.race([o1_1Promise, o9_16Promise, o16_9Promise]);
      if (orientationPromise) {
        const { el, selector } = orientationPromise;
        await el.click();
        console.log(`🖱️ Clicked orientation: ${selector}`);
        await delay(200)
      } else {
        console.error("❌ No orientation options appeared within timeout");
        console.error("Exiting script due to missing orientation options.");
        process.exit(1);
      }
  
      //now we choose the correct orientation we want
      await page.waitForSelector('text/' + ORIENTATION, { visible: true });
      await page.click('text/' + ORIENTATION);
      console.log(`🖱️ Selected orientation: ${ORIENTATION}`);
      await delay(randomDelay(300, 700)); // Random delay between 1-2 seconds
  
    }



    const oldSPromise = page
      .waitForSelector("div.xxko97p > div.x3pnbk8 svg", { visible: true })
      .then((el) => ({ el, selector: "old-svg" }))
      .catch(() => null);

    const newSPromise = page
      .waitForSelector("text/Generate", { visible: true })
      .then((el) => ({ el, selector: "Generate-text" }))
      .catch(() => null);

    const regenSPromise = page
      .waitForSelector("text/Regenerate", { visible: true })
      .then((el) => ({ el, selector: "Regenerate-text" }))
      .catch(() => null);  

    // Race the two promises
    const generateRaceResult = await Promise.race([oldSPromise, newSPromise]);

    if (generateRaceResult) {
      const { el, selector } = generateRaceResult;
      await el.click();
      console.log(`🖱️ Clicked element: ${selector}`);
      console.log("🖱 Clicked generate button");
      isFirstPrompt = false; // Set to false after the first prompt
      
    } else {
      throw new Error("❌ Neither submit buttons appeared within timeout");
    }



    try {
      await page.waitForNetworkIdle({
        timeout: 60000, // 60 seconds timeout
        idleTime: 5000, // Wait for 5 seconds of idle network
      });
    } catch (error) {
      console.error("❌ Network idle timeout reached. Retrying...");
      timeoutCount++;
      if (timeoutCount >= MAX_TIMEOUTS) {
        console.error("❌ Maximum timeouts reached. Exiting...");
        throw new Error("Maximum timeouts reached");
      }
      continue; // Retry the loop
    }
    await delay(400)


    //make the temp download folder
    const tempDownloadPath = path.join(imagesPath, prompt["tag"]);
    await fs.mkdirSync(tempDownloadPath, { recursive: true });

    // Get CDP session
    const client = await page.target().createCDPSession();
    // Set download behavior (works in both headed and new headless)

    await client.send("Page.setDownloadBehavior", {
      behavior: "allow",
      downloadPath: tempDownloadPath, // Update to use tempDownloadPath
    });
    console.log(`📁 Downloads will be saved to: ${tempDownloadPath}`); // Update log message
    // await delay(1000*60); // Wait for download to start

    // await page.waitForSelector(
    //   "div.xc26acl > div > div > div:nth-of-type(1) div:nth-of-type(1) > svg"
    // );

    await page.click(
      "div.xc26acl > div > div > div:nth-of-type(1) div:nth-of-type(1) > svg"
    );

    console.log("🖱 Clicked download button");

    try {
      await page.waitForNetworkIdle({
        timeout: 60000, // 60 seconds timeout
        idleTime: 3000, // Wait for 5 seconds of idle network
      });

      //rename the image file
      const downloadedFiles = fs.readdirSync(tempDownloadPath);
      if (downloadedFiles.length === 0) {
        throw new Error("❌ No files downloaded");
      }
      const downloadedFile = downloadedFiles[0];
      const newFileName = `${prompt["tag"]}.png`;
      const oldFilePath = path.join(tempDownloadPath, downloadedFile);
      const newFilePath = path.join(tempDownloadPath, newFileName);
      fs.renameSync(oldFilePath, newFilePath);
      console.log(`📝 Renamed downloaded file to: ${newFileName}`);

      // Move the file to the final images folder
      const finalFilePath = path.join(imagesPath, newFileName);
      fs.renameSync(newFilePath, finalFilePath);
      console.log(`📁 Moved file to final images folder: ${finalFilePath}`);

      //delete the temp folder, make sure its empty
      const files = fs.readdirSync(tempDownloadPath);
      if (files.length > 0) {
        console.warn(
          `⚠️ Temp download folder is not empty: ${tempDownloadPath}`
        );
      } else {
        fs.rmSync(tempDownloadPath, { recursive: true });
        console.log(
          `🗑️ Deleted temporary download folder: ${tempDownloadPath}`
        );
      }

      // Update prompt status to "done"
      prompt["gen-status"] = "done";
      prompt["gen-end"] = new Date().toISOString();
      prompt["gen-duration"] =
        (new Date(prompt["gen-end"]) - new Date(prompt["gen-start"])) / 1000; // Duration in seconds
      fs.writeFileSync(jsonPath, JSON.stringify(prompts, null, 2));
      console.log(`📝 Updated prompt status to "done"`);
    } catch (error) {
      console.error("❌ Network idle timeout reached. Retrying...");
      timeoutCount++;
      if (timeoutCount >= MAX_TIMEOUTS) {
        console.error("❌ Maximum timeouts reached. Exiting...");
        throw new Error("Maximum timeouts reached");
      }
      await page.reload({ waitUntil: "networkidle0" }); // Refresh page
      continue; // Retry the loop
    }

  }

  runShellScriptSync(killChromeScript, "kill-chrome-debug.sh");
})();
