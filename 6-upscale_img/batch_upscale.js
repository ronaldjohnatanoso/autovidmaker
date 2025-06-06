const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp'); // Add sharp for image resizing

// Get project name from CLI
const projectName = process.argv[2];
if (!projectName) {
  console.error("❌ Usage: node minimal-puppeteer.js <project_name>");
  process.exit(1);
}
console.log(`📁 Project: ${projectName}`);

// Base directory of the project
const baseDir = path.resolve(__dirname, '../0-project-files', projectName);

// Paths relative to script, not working dir
const inputDir = path.join(baseDir, 'images');
const outputDir = path.join(baseDir, 'upscaled_images');
const scaledOutputDir = path.join(baseDir, 'images_1080p'); // New folder for 1080p images

// Check if the input directory exists
if (!fs.existsSync(inputDir)) {
  console.error(`❌ Input directory ${inputDir} does not exist.`);
  process.exit(1);
}

// Create the output directories if they do not exist
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}
if (!fs.existsSync(scaledOutputDir)) {
  fs.mkdirSync(scaledOutputDir, { recursive: true });
}

const statusFile = path.join(baseDir, '.status.json');

const data = fs.readFileSync(statusFile, 'utf8');
const statusData = JSON.parse(data);

// Update status
statusData.stages.upscale_img.status = 'in_progress';
statusData.stages.upscale_img.started_at = new Date().toISOString();
// Write updated status back to file
fs.writeFileSync(statusFile, JSON.stringify(statusData, null, 2), 'utf8');

const cmd = path.resolve(__dirname, './realesrgan-ncnn-vulkan');
const args = [
  '-i', inputDir,
  '-o', outputDir,
  '-n', 'realesrgan-x4plus',
  '-f', 'png'
];

// Run the upscaling command
execFile(cmd, args, { stdio: 'ignore' }, async (error, stdout, stderr) => {
  if (error) {
    console.error('❌ Error:', error.message);
    return;
  }
  if (stderr) {

  }

  console.log('✅ Upscaling completed successfully.');

  // Downscale images to 1080p
  const files = fs.readdirSync(outputDir);
  for (const file of files) {
    const inputPath = path.join(outputDir, file);
    const outputPath = path.join(scaledOutputDir, file);

    try {
      await sharp(inputPath)
        .resize({ width: 1920, height: 1080, fit: 'inside' }) // Resize to fit within 1920x1080
        .toFile(outputPath);
      console.log(`✅ Scaled image saved: ${outputPath}`);
    } catch (err) {
      console.error(`❌ Error scaling image ${file}:`, err.message);
    }
  }

  // Update status to completed after scaling finishes
  const endTime = new Date();
  const startTime = new Date(statusData.stages.upscale_img.started_at);

  statusData.stages.upscale_img.status = 'completed';
  statusData.stages.upscale_img.completed_at = endTime.toISOString();
  statusData.stages.upscale_img.duration = endTime - startTime;

  fs.writeFileSync(statusFile, JSON.stringify(statusData, null, 2), 'utf8');
  console.log('✅ All images scaled to 1080p successfully.');
});