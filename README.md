# AutoVidMaker 🎬

An automated video production pipeline that transforms structured scripts into complete videos with AI-generated voices, images, and captions. Simply provide a script and let the system handle asset generation, timing, and video assembly - completely free!

## 🚀 Features

- **Automated Voice Generation**: Text-to-speech with multiple voice options and effects
- **AI Image Generation**: Contextual image creation based on script content
- **Smart Caption Timing**: Word-level subtitle synchronization
- **Image Upscaling**: Enhanced image quality using Real-ESRGAN
- **GPU-Accelerated Rendering**: OpenGL-based video composition
- **YouTube Integration**: Direct upload to YouTube with metadata
- **Pipeline Orchestration**: Automated workflow management

## 📁 Project Structure

```
autovidmaker/
├── 0-project-files/          # Generated project folders
├── 1-script-staging/         # Script validation and preparation
├── 2-voice_gen/             # Text-to-speech generation
├── 3-captions/              # Subtitle and timing generation
├── 4-img_prompts/           # Image prompt alignment
├── 5-image_gen/             # AI image generation
├── 6-upscale_img/           # Image quality enhancement
├── 7-img_stitch/            # Video rendering and composition
├── 8-yt-upload/             # YouTube upload automation
├── common_assets/           # Shared audio and media files
└── orchestrator.py          # Main pipeline controller
```

## 🛠️ Prerequisites

### System Requirements
- **Python 3.10+** with conda/miniconda
- **Node.js 16+** with npm
- **Chrome/Chromium** browser
- **CUDA-compatible GPU** (recommended for rendering)
- **FFmpeg** installed system-wide

### Dependencies Installation

1. **Python Environment Setup**:
```bash
conda create -n autovidmaker python=3.10
conda activate autovidmaker
pip install -r requirements.txt
```

2. **Node.js Dependencies**:
```bash
# Voice generation
cd 2-voice_gen && npm install

# Image generation  
cd ../5-image_gen && npm install

# Image upscaling
cd ../6-upscale_img && npm install
```

3. **CUDA/cuDNN Setup** (for GPU acceleration):
```bash
# Option 1: System package
sudo apt-get install libcudnn8

# Option 2: Conda environment (recommended)
export LD_LIBRARY_PATH=/path/to/conda/envs/autovidmaker/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

## 🎯 Quick Start

### Method 1: Full Pipeline Control

1. **Create your script** (`projectname.txt`) with structured content:
```
TITLE: Your Video Title
DESCRIPTION: Video description for YouTube
TAGS: tag1, tag2, tag3

[SCENE_1]
VOICE: Your narration text here
IMAGE_PROMPT: A detailed description for AI image generation

[SCENE_2]
VOICE: More narration
IMAGE_PROMPT: Another image description
```

2. **Stage and validate** your script:
```bash
python 1-script-staging/validate_and_move.py projectname.txt
```

3. **Run the complete pipeline**:
```bash
python orchestrator.py projectname --autoprocess-all
```

### Method 2: Quick Run (Default Settings)

1. **Place your script** as `input.txt` in the `1-script-staging/` folder
2. **Run with defaults**:
```bash
python orchestrator.py
```

This automatically validates and processes your script through the entire pipeline.

## 🔧 Pipeline Stages

### 1. Script Staging (`1-script-staging/`)
- Validates script format and structure
- Creates project folder with configuration
- Generates `config.json` for pipeline settings

### 2. Voice Generation (`2-voice_gen/`)
- **Chrome Profile Setup**: `sh copy-chrome-prof.sh`
- **TTS Processing**: Uses Google's text-to-speech via Puppeteer
- **Voice Effects**: Configurable voice types and audio effects
- **Output**: `{project}.wav` audio file

### 3. Caption Generation (`3-captions/`)
- **Word-level Timing**: Precise subtitle synchronization
- **SRT Generation**: Standard subtitle format
- **Output**: `{project}_wordlevel.srt`

### 4. Image Prompt Alignment (`4-img_prompts/`)
- **Timing Analysis**: Aligns image prompts with audio segments
- **Prompt Processing**: Optimizes prompts for AI generation
- **Output**: `{project}_img_prompts.json`

### 5. Image Generation (`5-image_gen/`)
- **AI Image Creation**: Generates contextual images
- **Batch Processing**: Handles multiple prompts efficiently
- **Output**: `images/` folder with generated assets

### 6. Image Upscaling (`6-upscale_img/`)
- **Real-ESRGAN**: AI-powered image enhancement
- **Quality Improvement**: Upscales to 1080p resolution
- **Output**: `upscaled_images/` and `images_1080p/`

### 7. Video Rendering (`7-img_stitch/`)
- **OpenGL Acceleration**: GPU-powered video composition
- **Multi-threading**: Parallel segment processing
- **Audio Sync**: Perfect timing with captions and images
- **Output**: `{project}_vertical.mp4`

### 8. YouTube Upload (`8-yt-upload/`)
- **Automated Upload**: Direct publishing to YouTube
- **Metadata Integration**: Title, description, and tags
- **Authentication**: OAuth2 token management

## ⚙️ Configuration

### Voice Settings (`config.json`)
```json
{
  "voice_type": "male",
  "voice_effects": ["reverb", "echo"],
  "speed": 1.0,
  "pitch": 0
}
```

### Chrome Profile Setup
```bash
# Copy existing Chrome profile for TTS
cd 2-voice_gen
sh copy-chrome-prof.sh

# Launch Chrome in debug mode
sh run-chrome.sh
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Library Not Found**:
```bash
# Permanent fix for conda environment
mkdir -p ~/miniconda3/envs/autovidmaker/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH' > ~/miniconda3/envs/autovidmaker/etc/conda/activate.d/env_vars.sh
```

**Chrome Connection Issues**:
```bash
# Kill existing Chrome debug sessions
cd 2-voice_gen
sh kill-chrome-debug.sh

# Restart Chrome in debug mode
sh run-chrome.sh
```

**FFmpeg Binary Issues**:
```python
# In Python scripts, specify FFmpeg path
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
```

## 📝 Script Writing Tips

### Structure Best Practices
- **Click-bait Titles**: Engaging titles that deliver on promises
- **Strong Openings**: First part should exceed expectations
- **Clear Scenes**: Separate content into distinct segments
- **Detailed Prompts**: Specific image descriptions for better AI generation

### Voice Variations
- Indian accent: `read this in indian accent`
- Drunk southern: `read this like a drunk old man with a southern american accent`
- Cockney/Australian: Specify accent in voice instructions

## 🎨 Assets and Media

### Background Audio (`common_assets/`)
- `choir.mp3` - Choir background
- `horror_piano.mp3` - Horror atmosphere
- `mystery.mp3` - Mystery/suspense
- `wander.mp3` - Ambient wandering

### Project Examples
Check `0-project-files/` for example projects:
- `jesus1/` through `jesus43/` - Religious content examples
- `demotivational/` - Motivational content
- `indianTutorial/` - Tutorial format

## 🚀 Advanced Usage

### Manual Stage Execution
```bash
# Run specific pipeline stage
python orchestrator.py projectname --stage voice_gen

# Check pipeline status
python orchestrator.py projectname --status
```

### Custom Rendering Options
- **CPU Rendering**: `moviepy-cpu-render.py`
- **GPU Rendering**: `opengl-gpu-render.py` (recommended)
- **Multi-threaded**: `v4_multi_thread_segment_breaking.py`

---

## 📋 Legacy README Content

automated video maker, just give a structured script and it will scrape assests totally free, and assemble all to a video

1) produce the projectname.txt - this contains the structured info on your entire script, see examples
for detail
2) stage the input script and validate it, the project file folder should now exist.
3) use the orchestrator to - python orchestrator projectname --autoprocess-all  
    to run the project through all the stages

<<<<<<< HEAD

=======
--- QUICK RUN ---
>>>>>>> c393bcab484b83f1862805cc88951b15a7d667f0

1) alternatively, make the input.txt script and put that inside the staging folder 
and make sure its the only there
2) do , python orchestrator, this automatically validates and runs it through the pipeline until it creates the video

NOTE: First method allows you to change the config.json for settings like voice type/effects
Second method just uses the default provided in the validator python script

Go inside each stage folder for testing
