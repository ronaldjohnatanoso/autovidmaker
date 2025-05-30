const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

(async () => {
  const browser = await puppeteer.connect({
    browserURL: "http://localhost:9222",
    defaultViewport: null,
  });

  // Simple wait helper function
  function delay(time) {
    return new Promise((resolve) => setTimeout(resolve, time));
  }

  // Read raw-struct.txt from the same directory
  const rawStruct = fs.readFileSync(
    path.join(__dirname, "raw-struct.txt"),
    "utf8"
  );

  const pages = await browser.pages();
  const page = pages.length ? pages[0] : await browser.newPage();

  //   await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36');
  // await page.setViewport({ width: 1920, height: 1080 });

  //   // 1️⃣ Navigate to Google AI Studio
  //   await page.goto('https://aistudio.google.com/generate-speech', {
  //     waitUntil: 'networkidle2',
  //   });
  //   console.log("✅ Navigated to AI Studio");

  // 💤 Keep the browser open
  process.stdin.resume();
})();
