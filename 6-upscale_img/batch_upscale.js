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

// Check if upscaled images already exist
const upscaledExists = fs.existsSync(outputDir) && fs.readdirSync(outputDir).length > 0;

if (upscaledExists) {
  console.log('✅ Upscaled images already exist, skipping upscaling step.');
  processImages();
} else {
  // Run the upscaling command
  execFile(cmd, args, { stdio: 'ignore' }, async (error, stdout, stderr) => {
    if (error) {
      console.error('❌ Error:', error.message);
      return;
    }
    if (stderr) {

    }

    console.log('✅ Upscaling completed successfully.');
    processImages();
  });
}

async function processImages() {
  // Downscale images to 1080p
  const files = fs.readdirSync(outputDir);
  for (const file of files) {
    const inputPath = path.join(outputDir, file);
    const outputPath = path.join(scaledOutputDir, file);

    try {
      // First, get the image metadata to determine dimensions
      const image = sharp(inputPath);
      const metadata = await image.metadata();
      
      // Resize to fit within 1920x1080
      const resizedImage = await image
        .resize({ width: 1920, height: 1080, fit: 'inside' })
        .png() // Ensure we have a format that supports transparency
        .toBuffer();

      // Load the resized image for watermark removal
      const resizedMetadata = await sharp(resizedImage).metadata();
      const { width, height } = resizedMetadata;

      // Define circle parameters - doubled radius, positioned at corner
      const circleRadius = Math.floor(Math.min(width, height) * 0.24); // Doubled radius
      
      // Extract quarter circle area from bottom-left corner (center at corner)
      const extractX = 0; // Start from edge
      const extractY = Math.floor(height - circleRadius); // Start from bottom edge
      
      // Create a quarter circle mask (bottom-right quarter of a full circle)
      const quarterCircleMask = Buffer.from(
        `<svg width="${circleRadius}" height="${circleRadius}">
          <circle cx="0" cy="${circleRadius}" r="${circleRadius}" fill="white"/>
        </svg>`
      );

      // Extract the quarter circle area from bottom-left
      const extractedQuarterCircle = await sharp(resizedImage)
        .extract({ 
          left: extractX, 
          top: extractY, 
          width: circleRadius, 
          height: circleRadius 
        })
        .composite([
          {
            input: quarterCircleMask,
            blend: 'dest-in'
          }
        ])
        .png()
        .toBuffer();

      // Mirror the extracted quarter circle horizontally
      const mirroredQuarterCircle = await sharp(extractedQuarterCircle)
        .flop() // Horizontal flip
        .png()
        .toBuffer();

      // Position for bottom-right corner (covering watermark)
      const pasteX = Math.floor(width - circleRadius);
      const pasteY = Math.floor(height - circleRadius);

      // Composite the mirrored quarter circle onto the bottom-right corner
      const finalImage = await sharp(resizedImage)
        .composite([
          {
            input: mirroredQuarterCircle,
            left: pasteX,
            top: pasteY,
            blend: 'over'
          }
        ])
        .png()
        .toFile(outputPath);

      console.log(`✅ Scaled image with watermark removal saved: ${outputPath}`);
    } catch (err) {
      console.error(`❌ Error processing image ${file}:`, err.message);
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
}