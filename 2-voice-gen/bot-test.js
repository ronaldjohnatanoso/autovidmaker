const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

(async () => {
  const browser = await puppeteer.connect({
    browserURL: 'http://localhost:9222',
    defaultViewport: null,
  });

  const pages = await browser.pages();
  const page = pages.length ? pages[0] : await browser.newPage();


  
  // Set user agent to a real one (from a headful session)
  await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36');

  // 🧠 Manual stealth patches
  await page.evaluateOnNewDocument(() => {
    // 🛡️ WebDriver: false
    Object.defineProperty(navigator, 'webdriver', {
      get: () => false,
    });

    // 🛡️ Languages
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });

    // 🛡️ Plugins
    Object.defineProperty(navigator, 'plugins', {
      get: () => [
        { name: 'Chrome PDF Plugin' },
        { name: 'Chrome PDF Viewer' },
        { name: 'Native Client' },
        { name: 'Shockwave Flash' },
      ],
    });

    // 🛡️ Mimic PluginArray type
    Object.setPrototypeOf(navigator.plugins, PluginArray.prototype);

    // 🛡️ WebGL vendor & renderer
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function (param) {
      if (param === 37445) return 'Google Inc. (Intel)'; // UNMASKED_VENDOR_WEBGL
      if (param === 37446) return 'ANGLE (Intel, Mesa Intel(R) Graphics (ADL GT2), OpenGL 4.6)';
      return getParameter.call(this, param);
    };
  });

  console.log("🌐 Navigating to bot.sannysoft.com...");
  await page.goto('https://bot.sannysoft.com/', {
    waitUntil: 'networkidle2',
    timeout: 60000,
  });

  console.log("📸 Taking screenshot...");
  await page.screenshot({ path: 'bot-test.png', fullPage: true });

  console.log("✅ Screenshot saved as bot-test.png");

  // Keep browser open or disconnect manually
  // await browser.disconnect();
})();
