const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');

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

// Check if the input directory exists
if (!fs.existsSync(inputDir)) {
  console.error(`❌ Input directory ${inputDir} does not exist.`);
  process.exit(1);
}

// Create the output directory if it does not exist
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

const statusFile = path.join(baseDir, '.status.json');


const data =  fs.readFileSync(statusFile, 'utf8');
const statusData = JSON.parse(data);

// Update status
statusData.stages.upscale_img.status = 'in_progress';
// started_at
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
execFile(cmd, args, { stdio: 'ignore' }, (error, stdout, stderr) => {
  if (error) {
    console.error('❌ Error:', error.message);
    return;
  }
  if (stderr) {
    console.error('❌ Stderr:', stderr);
    return;
  }
  
  // Update status to completed after command finishes
  const endTime = new Date();
  const startTime = new Date(statusData.stages.upscale_img.started_at);

  statusData.stages.upscale_img.status = 'completed';
  statusData.stages.upscale_img.completed_at = endTime.toISOString();
  statusData.stages.upscale_img.duration = endTime - startTime;

  fs.writeFileSync(statusFile, JSON.stringify(statusData, null, 2), 'utf8');
  console.log('✅ Upscaling completed successfully.');
});