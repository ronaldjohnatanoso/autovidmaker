import logging
import subprocess
import numpy as np
import moderngl  # OpenGL wrapper for Python
import psutil
import sys
import time
import json
import os
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import threading
import queue
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import tempfile
import multiprocessing as mp
import re
import signal
import atexit
import weakref

# Global cleanup tracking
active_processes = []
active_contexts = []
active_executors = []  # Add this to track executors
active_multiprocessing_processes = []  # Add this to track MP processes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Video output settings
WIDTH = 1920
HEIGHT = 1080
FRAME_RATE = 30
VRAM_THRESHOLD = 0.95
SEGMENT_DURATION = 20
MAX_WORKERS = 6
SUBTITLE_FONT_SIZE = 64 * 2

# Audio settings
BACKGROUND_MUSIC_FILE = "wander.mp3"  # Change this to your BGM filename
BGM_VOLUME = 0.5                     # Background music volume (0.0 to 1.0)
NARRATION_VOLUME = 1.0                # Narration audio volume (0.0 to 1.0)

# Set to True to only process and preview first 10 seconds
PREVIEW_MODE = False

def cleanup_processes():
    """Clean up all active processes"""
    global active_processes, active_executors, active_multiprocessing_processes
    logging.info("Cleaning up active processes...")
    
    # Clean up ProcessPoolExecutor processes
    for executor in active_executors[:]:
        try:
            logging.info("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
            active_executors.remove(executor)
        except Exception as e:
            logging.warning(f"Error shutting down executor: {e}")
    
    # Clean up multiprocessing processes
    for proc in active_multiprocessing_processes[:]:
        try:
            if proc.is_alive():
                logging.info(f"Terminating multiprocessing process {proc.pid}")
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    logging.warning(f"Force killing multiprocessing process {proc.pid}")
                    proc.kill()
                    proc.join()
            active_multiprocessing_processes.remove(proc)
        except Exception as e:
            logging.warning(f"Error cleaning up multiprocessing process: {e}")
    
    # Clean up FFmpeg processes
    for proc in active_processes[:]:
        try:
            if proc.poll() is None:
                logging.info(f"Terminating FFmpeg process {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Force killing FFmpeg process {proc.pid}")
                    proc.kill()
                    proc.wait()
            active_processes.remove(proc)
        except Exception as e:
            logging.warning(f"Error cleaning up FFmpeg process: {e}")

def cleanup_contexts():
    """Clean up OpenGL contexts"""
    global active_contexts
    for ctx_ref in active_contexts[:]:
        try:
            ctx = ctx_ref()
            if ctx:
                ctx.release()
        except:
            pass
    active_contexts.clear()

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logging.info(f"Received signal {signum}, cleaning up...")
    cleanup_processes()
    cleanup_contexts()
    
    # Force kill any remaining Python processes
    if psutil:
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    logging.info(f"Force terminating child process {child.pid}")
                    child.terminate()
                except:
                    pass
            
            # Wait a bit then kill remaining
            time.sleep(2)
            for child in children:
                try:
                    if child.is_running():
                        logging.info(f"Force killing child process {child.pid}")
                        child.kill()
                except:
                    pass
        except Exception as e:
            logging.warning(f"Error during process cleanup: {e}")
    
    sys.exit(1)

# Register cleanup handlers
atexit.register(cleanup_processes)
atexit.register(cleanup_contexts)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_srt_file(srt_path):
    """Parse SRT subtitle file and return list of subtitle entries"""
    if not os.path.exists(srt_path):
        logging.warning(f"SRT file not found: {srt_path}")
        return []
    
    subtitles = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to get subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Parse timing line (format: 00:00:00,000 --> 00:00:07,253)
            timing_line = lines[1]
            timing_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', timing_line)
            
            if not timing_match:
                continue
            
            # Convert to seconds
            start_h, start_m, start_s, start_ms = map(int, timing_match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, timing_match.groups()[4:])
            
            start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
            end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
            
            # Get subtitle text (can be multiple lines)
            text = '\n'.join(lines[2:]).strip()
            
            subtitles.append({
                'start_time': start_time,
                'end_time': end_time,
                'text': text,
                'duration': end_time - start_time
            })
    
    except Exception as e:
        logging.error(f"Error parsing SRT file: {e}")
        return []
    
    logging.info(f"Loaded {len(subtitles)} subtitle entries")
    return subtitles

def create_text_texture(text, font_size=48, max_width=1600):
    """Create text texture with stroke effect for GPU rendering"""
    if not text or text.strip() == "":
        # Return empty texture
        return np.zeros((64, 64, 4), dtype=np.uint8), 64, 64
    
    try:
        # Find available font
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/TTF/arial.ttf"
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = PIL.ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
        
        if font is None:
            font = PIL.ImageFont.load_default()
        
        # Split text into lines and measure with proper descender handling
        lines = text.split('\n')
        line_heights = []
        line_widths = []
        line_ascents = []
        line_descents = []
        
        # Test string with descenders to get proper metrics
        test_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890.,;:!?()[]{}\"'`~@#$%^&*-_+=|\\/<>ypqgjQ"
        test_bbox = font.getbbox(test_string)
        font_ascent = abs(test_bbox[1])  # Distance from baseline to top
        font_descent = test_bbox[3] - font_ascent  # Distance from baseline to bottom
        
        for line in lines:
            if line.strip():  # Non-empty line
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                # Use consistent line height based on font metrics
                line_height = font_ascent + font_descent
            else:  # Empty line
                line_width = 0
                line_height = font_ascent + font_descent
            
            line_widths.append(line_width)
            line_heights.append(line_height)
            line_ascents.append(font_ascent)
            line_descents.append(font_descent)
        
        # Calculate total dimensions
        text_width = min(max(line_widths) if line_widths else 0, max_width)
        line_spacing = int(font_size * 0.2)
        text_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        
        # Add padding for stroke effect and descenders
        stroke_width = max(2, font_size // 16)
        padding = stroke_width * 3 + font_descent  # Extra padding for descenders
        
        img_width = text_width + padding * 2
        img_height = text_height + padding * 2
        
        # Create image with transparency
        img = PIL.Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(img)
        
        # Draw text with stroke effect
        y_offset = padding + font_ascent  # Start from baseline, not top
        
        for i, line in enumerate(lines):
            if not line.strip():  # Skip empty lines but maintain spacing
                y_offset += line_heights[i] + line_spacing
                continue
                
            # Center each line
            line_width = line_widths[i] if i < len(line_widths) else 0
            x_pos = (img_width - line_width) // 2
            
            # Draw stroke (black outline) - draw from baseline
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx*dx + dy*dy <= stroke_width*stroke_width:
                        draw.text((x_pos + dx, y_offset + dy), line, font=font, fill=(0, 0, 0, 255), anchor="ls")
            
            # Draw main text (white) - draw from baseline
            draw.text((x_pos, y_offset), line, font=font, fill=(255, 255, 255, 255), anchor="ls")
            
            y_offset += line_heights[i] + line_spacing
        
        # Convert to numpy array
        img_array = np.array(img)
        return img_array, img_width, img_height
        
    except Exception as e:
        logging.warning(f"Text texture generation failed: {e}")
        # Return empty texture
        fallback = np.zeros((64, 64, 4), dtype=np.uint8)
        return fallback, 64, 64

def get_subtitle_for_time(subtitles, current_time):
    """Get the subtitle that should be displayed at the given time"""
    for subtitle in subtitles:
        if subtitle['start_time'] <= current_time <= subtitle['end_time']:
            return subtitle
    return None

def load_project_data(project_name):
    """Load project images and timing data from project folder structure"""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent / "0-project-files" / project_name
    
    # Check if project exists
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    
    # Load timing data
    timing_file = project_dir / f"{project_name}_img_prompts.json"
    if not timing_file.exists():
        raise FileNotFoundError(f"Timing file not found: {timing_file}")
    
    with open(timing_file, 'r') as f:
        timing_data = json.load(f)
    
    # Load subtitles
    srt_file = project_dir / f"{project_name}_wordlevel.srt"
    subtitles = parse_srt_file(srt_file)
    
    # Find images directory
    images_dir = None
    for dir_name in ['images_1080p']:
        test_dir = project_dir / dir_name
        if test_dir.exists():
            images_dir = test_dir
            break
    
    if not images_dir:
        raise FileNotFoundError(f"No images directory found in {project_dir}")
    
    logging.info(f"Using images from: {images_dir}")

    # Validate and load images
    image_timeline = []
    total_duration = 0
    
    for entry in timing_data:
        tag = entry['tag']
        start_time = float(entry['start'])
        end_time = float(entry['end_adjusted'])
        duration = end_time - start_time
        
        # Find image file
        image_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            test_file = images_dir / f"{tag}{ext}"
            if test_file.exists():
                image_file = test_file
                break
        
        if not image_file:
            logging.warning(f"Image not found for tag: {tag}")
            continue
        
        image_timeline.append({
            'tag': tag,
            'image_path': str(image_file),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        })
        
        total_duration = max(total_duration, end_time)
    
    logging.info(f"Loaded {len(image_timeline)} images, total duration: {total_duration:.2f}s")
    return image_timeline, total_duration, subtitles

def create_image_video_segments(image_timeline, total_duration, output_dir):
    """Create video segments from image timeline data"""
    segments = []
    
    # Calculate number of segments
    num_segments = int(np.ceil(total_duration / SEGMENT_DURATION))
    
    for segment_idx in range(num_segments):
        segment_start = segment_idx * SEGMENT_DURATION
        segment_end = min((segment_idx + 1) * SEGMENT_DURATION, total_duration)
        
        # Find images that appear in this segment
        segment_images = []
        for img_data in image_timeline:
            # Check if image overlaps with this segment
            if not (img_data['end_time'] <= segment_start or img_data['start_time'] >= segment_end):
                # Calculate when this image appears/disappears within the segment
                img_segment_start = max(0, img_data['start_time'] - segment_start)
                img_segment_end = min(segment_end - segment_start, img_data['end_time'] - segment_start)
                
                segment_images.append({
                    'image_path': img_data['image_path'],
                    'tag': img_data['tag'],
                    'segment_start': img_segment_start,
                    'segment_end': img_segment_end,
                    'global_start': img_data['start_time'],
                    'global_end': img_data['end_time']
                })
        
        if not segment_images:
            # Create a black frame segment if no images
            segment_images = [{
                'image_path': None,  # Will create black frame
                'tag': 'black',
                'segment_start': 0,
                'segment_end': segment_end - segment_start,
                'global_start': segment_start,
                'global_end': segment_end
            }]
        
        segment_output = os.path.join(output_dir, f"segment_{segment_idx:06d}.mp4")
        
        segments.append({
            'output': segment_output,
            'start_time': segment_start,
            'end_time': segment_end,
            'duration': segment_end - segment_start,
            'images': segment_images,
            'index': segment_idx
        })
    
    return segments

def load_and_resize_image(image_path, target_width=WIDTH, target_height=HEIGHT):
    """Load and resize image to target dimensions with proper aspect ratio"""
    if image_path is None:
        # Create black frame
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    try:
        img = PIL.Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate scaling to fit within target dimensions while maintaining aspect ratio
        img_width, img_height = img.size
        scale_w = target_width / img_width
        scale_h = target_height / img_height
        scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
        
        # Create black background and center the image
        final_img = PIL.Image.new('RGB', (target_width, target_height), (0, 0, 0))
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        final_img.paste(img, (offset_x, offset_y))
        
        # Convert to numpy array
        return np.array(final_img)
        
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

def generate_composite_frame(segment_images, frame_time, segment_start_time):
    """Generate a composite frame from multiple images based on timing"""
    # Start with black background
    composite = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Composite images based on timing
    for img_data in segment_images:
        # Check if this image should be visible at this frame time
        if img_data['segment_start'] <= frame_time <= img_data['segment_end']:
            # Load and composite the image
            img_frame = load_and_resize_image(img_data['image_path'])
            
            # Simple replacement (could be enhanced with cross-fades)
            composite = img_frame
    
    return composite.astype(np.uint8)

def load_shader_file(filename):
    """Load shader source code from a .glsl file"""
    shader_path = Path(__file__).parent / "shaders" / filename
    try:
        with open(shader_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Shader file not found: {shader_path}")
        raise

def get_gpu_memory_usage():
    """Check GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True
        )
        used, total = map(int, result.stdout.strip().split('\n')[0].split(','))
        return used / total
    except Exception as e:
        logging.warning(f"Failed to retrieve GPU memory usage: {e}")
        return 0

def process_image_segment(segment_info, subtitles, progress_queue=None):
    """Process one video segment with shader effects applied to image timeline"""
    global active_processes, active_contexts
    
    segment_idx = segment_info['index']
    output_file = segment_info['output']
    segment_duration = segment_info['duration']
    segment_images = segment_info['images']
    global_start_time = segment_info['start_time']
    
    total_frames = int(segment_duration * FRAME_RATE)
    
    ctx = None
    ffmpeg_out = None
    
    try:
        logging.info(f"Processing image segment {segment_idx}: {total_frames} frames")
        logging.info(f"Segment {segment_idx} images: {[img['tag'] for img in segment_images]}")
        
        # Create OpenGL context
        ctx = moderngl.create_standalone_context(backend='egl')
        active_contexts.append(weakref.ref(ctx))
        
        # Load shaders
        vertex_shader = load_shader_file("vertex.glsl")
        fragment_shader = load_shader_file("fragment.glsl")
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Create quad
        quad = ctx.buffer(np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4'))
        
        vao = ctx.vertex_array(prog, [(quad, '2f 2f', 'in_vert', 'in_uv')])
        
        # Create textures and framebuffer
        tex = ctx.texture((WIDTH, HEIGHT), 3)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Create subtitle texture (will be updated per frame)
        subtitle_tex = ctx.texture((64, 64), 4)  # Start with small texture
        subtitle_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((WIDTH, HEIGHT), 3)])
        
        # Setup FFmpeg output with NVIDIA GPU encoding
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FRAME_RATE),
            '-i', '-', '-an', 
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',                 # Fastest preset
            '-tune', 'll',                   # Low latency
            '-rc', 'cbr',                    # Constant bitrate
            '-b:v', '6M',                    # Lower bitrate for speed
            '-pix_fmt', 'yuv420p', 
            output_file
        ]
        
        logging.info(f"Starting FFmpeg for segment {segment_idx} with NVENC GPU encoding")
        
        ffmpeg_out = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Track the process for cleanup
        active_processes.append(ffmpeg_out)
        
        current_subtitle_text = ""
        
        # Generate frames
        for frame_idx in range(total_frames):
            # Check for interruption
            if frame_idx % 30 == 0:  # Check every 30 frames
                if ffmpeg_out.poll() is not None:
                    stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
                    logging.error(f"FFmpeg terminated early in segment {segment_idx}: {stderr_output}")
                    break
            
            frame_time = frame_idx / FRAME_RATE  # Time within this segment
            global_time = global_start_time + frame_time  # Global video time
            
            # Generate composite frame from images
            composite_frame = generate_composite_frame(segment_images, frame_time, global_start_time)
            
            # Validate composite frame
            if composite_frame is None or composite_frame.size == 0:
                logging.warning(f"Empty composite frame at time {frame_time} in segment {segment_idx}")
                composite_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            # Get subtitle for current time
            current_subtitle = get_subtitle_for_time(subtitles, global_time)
            subtitle_text = current_subtitle['text'] if current_subtitle else ""
            
            # Update subtitle texture if text changed
            if subtitle_text != current_subtitle_text:
                current_subtitle_text = subtitle_text
                if subtitle_text:
                    text_data, text_width, text_height = create_text_texture(subtitle_text, font_size=SUBTITLE_FONT_SIZE)
                    # Recreate texture with new dimensions
                    subtitle_tex = ctx.texture((text_width, text_height), 4)
                    subtitle_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    subtitle_tex.write(text_data.tobytes())
                else:
                    # Empty subtitle
                    empty_data = np.zeros((64, 64, 4), dtype=np.uint8)
                    subtitle_tex = ctx.texture((64, 64), 4)
                    subtitle_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    subtitle_tex.write(empty_data.tobytes())
                    text_width, text_height = 64, 64
            
            # Upload frame to GPU
            tex.write(composite_frame.tobytes())
            
            # Render with shaders
            fbo.use()
            tex.use(0)
            subtitle_tex.use(1)
            prog['tex'] = 0
            prog['subtitle_tex'] = 1
            prog['u_time'] = global_time
            
            # Set subtitle properties
            if subtitle_text:
                # Position subtitle at bottom center
                subtitle_scale_x = min(text_width / WIDTH * 0.8, 0.8)  # Max 80% of screen width
                subtitle_scale_y = text_height / HEIGHT * subtitle_scale_x * (WIDTH / text_width)
                subtitle_x = 0.5 - subtitle_scale_x / 2  # Center horizontally
                subtitle_y = 0.85 - subtitle_scale_y    # Near bottom
                
                prog['subtitle_size'] = [subtitle_scale_x, subtitle_scale_y]
                prog['subtitle_pos'] = [subtitle_x, subtitle_y]
                prog['show_subtitle'] = 1.0
            else:
                prog['subtitle_size'] = [0.0, 0.0]
                prog['subtitle_pos'] = [0.0, 0.0]
                prog['show_subtitle'] = 0.0
            
            vao.render(moderngl.TRIANGLE_STRIP)
            
            # Read processed frame
            out_data = fbo.read(components=3, alignment=1)
            
            try:
                ffmpeg_out.stdin.write(out_data)
                ffmpeg_out.stdin.flush()
            except BrokenPipeError as e:
                stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
                logging.error(f"Broken pipe in segment {segment_idx} at frame {frame_idx}: {stderr_output}")
                break
            except Exception as e:
                logging.error(f"Error writing to FFmpeg: {e}")
                break
            
            # Update progress
            if progress_queue:
                try:
                    progress_queue.put({
                        'segment_index': segment_idx,
                        'frames_processed': frame_idx + 1,
                        'total_frames': total_frames
                    })
                except Exception:
                    pass
            
            # Check VRAM usage
            gpu_usage = get_gpu_memory_usage()
            if gpu_usage >= VRAM_THRESHOLD:
                logging.error(f"⚠️ VRAM threshold exceeded in segment {segment_idx}")
                break
        
        # Clean up FFmpeg
        if ffmpeg_out:
            try:
                ffmpeg_out.stdin.close()
                return_code = ffmpeg_out.wait(timeout=30)  # 30 second timeout
            except subprocess.TimeoutExpired:
                logging.warning(f"FFmpeg timeout for segment {segment_idx}, terminating")
                ffmpeg_out.terminate()
                try:
                    ffmpeg_out.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ffmpeg_out.kill()
                return_code = -1
            
            # Remove from active processes
            if ffmpeg_out in active_processes:
                active_processes.remove(ffmpeg_out)
        
        # Clean up OpenGL context
        if ctx:
            try:
                ctx.release()
            except:
                pass
        
        # Check FFmpeg return code
        if return_code != 0:
            stderr_output = ffmpeg_out.stderr.read().decode('utf-8') if ffmpeg_out else ""
            logging.error(f"FFmpeg failed for segment {segment_idx} with return code {return_code}: {stderr_output}")
            return {
                'success': False,
                'segment_index': segment_idx,
                'error': f"FFmpeg failed: {stderr_output}"
            }
        
        if progress_queue:
            try:
                progress_queue.put({
                    'segment_index': segment_idx,
                    'completed': True,
                    'frames_processed': total_frames
                })
            except Exception:
                pass
        
        logging.info(f"Completed image segment {segment_idx}")
        return {
            'success': True,
            'segment_index': segment_idx,
            'output_file': output_file,
            'frames_processed': total_frames
        }
        
    except KeyboardInterrupt:
        logging.info(f"Segment {segment_idx} interrupted by user")
        raise
    except Exception as e:
        logging.error(f"Error processing image segment {segment_idx}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'segment_index': segment_idx,
            'error': str(e)
        }
    finally:
        # Ensure cleanup happens even on exception
        if ffmpeg_out and ffmpeg_out in active_processes:
            try:
                if ffmpeg_out.poll() is None:
                    ffmpeg_out.terminate()
                    ffmpeg_out.wait(timeout=5)
                active_processes.remove(ffmpeg_out)
            except:
                pass
        
        if ctx:
            try:
                ctx.release()
            except:
                pass

def progress_monitor(progress_queue, num_segments, segment_info_dict):
    """Monitor progress for all segments with timeout"""
    progress_bars = {}
    for segment_idx in range(num_segments):
        segment_info = segment_info_dict[segment_idx]
        total_frames = int(segment_info['duration'] * FRAME_RATE)
        
        progress_bars[segment_idx] = tqdm(
            total=total_frames,
            desc=f"Segment {segment_idx:02d}",
            position=segment_idx,
            leave=True,
            unit="frame",
            dynamic_ncols=True
        )
    
    completed_segments = set()
    timeout_count = 0
    max_timeouts = 10  # Reduce timeout to 10 seconds
    
    try:
        while len(completed_segments) < num_segments:
            try:
                update = progress_queue.get(timeout=1.0)
                timeout_count = 0  # Reset timeout counter on successful update
                segment_idx = update['segment_index']
                
                if 'error' in update:
                    progress_bars[segment_idx].set_description(f"Segment {segment_idx:02d} ERROR")
                    progress_bars[segment_idx].close()
                    completed_segments.add(segment_idx)
                elif 'completed' in update:
                    progress_bars[segment_idx].set_description(f"Segment {segment_idx:02d} ✓")
                    progress_bars[segment_idx].close()
                    completed_segments.add(segment_idx)
                elif 'frames_processed' in update:
                    frames_processed = update['frames_processed']
                    total_frames = update.get('total_frames', progress_bars[segment_idx].total)
                    
                    current_pos = progress_bars[segment_idx].n
                    progress_bars[segment_idx].update(frames_processed - current_pos)
                    
                    percentage = (frames_processed / total_frames) * 100 if total_frames > 0 else 0
                    progress_bars[segment_idx].set_description(
                        f"Segment {segment_idx:02d} ({percentage:.1f}%)"
                    )
            
            except queue.Empty:
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    logging.warning("Progress monitor timed out - forcing exit")
                    break
                continue
            except Exception as e:
                logging.warning(f"Progress monitor error: {e}")
                continue
    
    except KeyboardInterrupt:
        logging.info("Progress monitor interrupted")
    finally:
        # Force close all progress bars
        for pbar in progress_bars.values():
            try:
                if not pbar.disable and hasattr(pbar, 'close'):
                    pbar.close()
            except:
                pass
        logging.info("Progress monitor cleanup completed")

def concatenate_segments(processed_segments, final_output):
    """Concatenate processed segments using FFmpeg with GPU acceleration"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for segment in sorted(processed_segments, key=lambda x: x['segment_index']):
            f.write(f"file '{segment['output_file']}'\n")
    
    try:
        # Use GPU-accelerated concatenation
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c:v', 'h264_nvenc',  # GPU encoding for concatenation
            '-preset', 'p4',       # Balanced preset
            '-rc', 'vbr',          # Variable bitrate
            '-cq', '23',           # Quality
            '-b:v', '8M',          # Target bitrate
            '-maxrate', '12M',     # Max bitrate
            '-bufsize', '16M',     # Buffer size
            '-pix_fmt', 'yuv420p',
            final_output
        ]
        
        logging.info("Concatenating with GPU acceleration...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"✓ GPU concatenation successful: {final_output}")
        
    except subprocess.CalledProcessError as e:
        # Fallback to copy mode (no re-encoding)
        logging.warning("GPU concatenation failed, using copy mode")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # Just copy, no re-encoding
                final_output
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info(f"✓ Copy concatenation successful: {final_output}")
        except subprocess.CalledProcessError as e2:
            logging.error(f"Both GPU and copy concatenation failed: {e2}")
            raise
        
    finally:
        os.unlink(concat_file)

def play_video_preview(video_file):
    """Play processed video using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            logging.error("Failed to open processed video")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps
        
        cv2.namedWindow('Preview - Shader Effects', cv2.WINDOW_AUTOSIZE)
        logging.info("Playing preview... Press 'q' or ESC to quit, SPACE to restart")
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            height, width = frame.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow('Preview - Shader Effects', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logging.error(f"Error playing preview: {e}")

def process_segment_wrapper(args):
    """Wrapper function for multiprocessing"""
    segment_info, subtitles, progress_queue = args
    return process_image_segment(segment_info, subtitles, progress_queue)

def add_audio_to_video(video_file, narration_audio, bgm_audio, final_output, total_duration):
    """Add narration and background music to video using FFmpeg GPU acceleration"""
    try:
        # Build FFmpeg command with GPU acceleration
        cmd = ['ffmpeg', '-y']
        
        # Input video
        cmd.extend(['-i', video_file])
        
        # Input narration audio (if exists)
        narration_exists = narration_audio and os.path.exists(narration_audio)
        if narration_exists:
            cmd.extend(['-i', narration_audio])
        
        # Input background music (if exists)
        bgm_exists = bgm_audio and os.path.exists(bgm_audio)
        if bgm_exists:
            cmd.extend(['-i', bgm_audio])
        
        # Video codec - keep existing encoding
        cmd.extend(['-c:v', 'copy'])  # Copy video without re-encoding
        
        # Audio processing
        if narration_exists and bgm_exists:
            # Mix narration and background music
            audio_input_index = 1  # Narration
            bgm_input_index = 2    # Background music
            
            # Create complex filter to mix audio with configurable volumes
            filter_complex = (
                f"[{bgm_input_index}:a]aloop=loop=-1:size=2e+09[bg];"  # Loop BGM indefinitely
                f"[bg]volume={BGM_VOLUME}[bg_quiet];"  # Set BGM volume
                f"[{audio_input_index}:a]volume={NARRATION_VOLUME}[narration_vol];"  # Set narration volume
                f"[narration_vol][bg_quiet]amix=inputs=2:duration=first:dropout_transition=2[mixed]"
            )
            cmd.extend(['-filter_complex', filter_complex])
            cmd.extend(['-map', '0:v'])  # Map video from first input
            cmd.extend(['-map', '[mixed]'])  # Map mixed audio
            
        elif narration_exists:
            # Only narration with volume control
            if NARRATION_VOLUME != 1.0:
                cmd.extend(['-filter_complex', f'[1:a]volume={NARRATION_VOLUME}[narration_vol]'])
                cmd.extend(['-map', '0:v'])  # Map video
                cmd.extend(['-map', '[narration_vol]'])  # Map volume-adjusted narration
            else:
                cmd.extend(['-map', '0:v'])  # Map video
                cmd.extend(['-map', '1:a'])  # Map narration audio as-is
            
        elif bgm_exists:
            # Only background music with volume control
            cmd.extend(['-filter_complex', f'[1:a]aloop=loop=-1:size=2e+09,volume={BGM_VOLUME}[bg]'])
            cmd.extend(['-map', '0:v'])  # Map video
            cmd.extend(['-map', '[bg]'])  # Map looped BGM with volume
            
        else:
            # No audio - just copy video
            cmd.extend(['-map', '0:v'])
            cmd.extend(['-an'])  # No audio
        
        # Audio codec and settings
        if narration_exists or bgm_exists:
            cmd.extend(['-c:a', 'aac'])
            cmd.extend(['-b:a', '128k'])
            cmd.extend(['-ar', '48000'])
        
        # Duration and output
        cmd.extend(['-t', str(total_duration)])  # Limit to video duration
        cmd.append(final_output)
        
        logging.info("Adding audio to video...")
        if narration_exists:
            logging.info(f"  📢 Narration: {narration_audio} (volume: {NARRATION_VOLUME})")
        if bgm_exists:
            logging.info(f"  🎵 Background music: {bgm_audio} (volume: {BGM_VOLUME})")
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"✅ Audio processing completed: {final_output}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to add audio: {e}")
        raise
    except Exception as e:
        logging.error(f"❌ Audio processing error: {e}")
        raise

def find_audio_files(project_name, script_dir):
    """Find narration and background music files"""
    project_dir = script_dir.parent / "0-project-files" / project_name
    
    # Find narration audio
    narration_file = project_dir / f"{project_name}.wav"
    narration_audio = str(narration_file) if narration_file.exists() else None
    
    # Find background music
    common_assets_dir = script_dir.parent / "common_assets"
    bgm_file = common_assets_dir / BACKGROUND_MUSIC_FILE
    bgm_audio = str(bgm_file) if bgm_file.exists() else None
    
    # Log what we found
    if narration_audio:
        logging.info(f"🎤 Found narration: {narration_audio}")
    else:
        logging.info(f"⚠️  No narration found: {project_dir / f'{project_name}.wav'}")
    
    if bgm_audio:
        logging.info(f"🎵 Found background music: {bgm_audio}")
    else:
        logging.info(f"⚠️  No background music found: {common_assets_dir / BACKGROUND_MUSIC_FILE}")
    
    return narration_audio, bgm_audio

def main():
    parser = argparse.ArgumentParser(description='Create video from project images with shader effects and subtitles')
    parser.add_argument('project_name', help='Name of the project folder')
    parser.add_argument('--preview', action='store_true', help='Create and preview first segment only')
    parser.add_argument('--output', help='Output video file path (default: project_name.mp4)')
    
    args = parser.parse_args()
    
    global PREVIEW_MODE, active_executors, active_multiprocessing_processes
    PREVIEW_MODE = args.preview
    
    try:
        # Load project data including subtitles
        image_timeline, total_duration, subtitles = load_project_data(args.project_name)
        
        # Create segments FIRST to determine preview duration
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create segments from image timeline
            segments = create_image_video_segments(image_timeline, total_duration, temp_dir)
            logging.info(f"Created {len(segments)} segments")
            
            if PREVIEW_MODE:
                # Use first segment duration for preview
                first_segment = segments[0]
                preview_duration = first_segment['duration']
                total_duration = preview_duration
                
                # Filter timeline for preview (first segment only)
                image_timeline = [img for img in image_timeline if img['start_time'] < preview_duration]
                for img in image_timeline:
                    img['end_time'] = min(img['end_time'], preview_duration)
                
                # Keep only the first segment
                segments = [first_segment]
                
                logging.info(f"Preview mode: Using first segment duration of {preview_duration:.2f}s")
        
            # Set output file paths
            script_dir = Path(__file__).parent
            
            if args.output:
                final_output = args.output
            else:
                project_dir = script_dir.parent / "0-project-files" / args.project_name
                suffix = "_preview" if PREVIEW_MODE else ""
                final_output = str(project_dir / f"{args.project_name}{suffix}.mp4")
            
            # Create temporary video output (without audio)
            temp_video_output = final_output.replace('.mp4', '_temp_video.mp4')
            
            logging.info(f"Creating video: {final_output}")
            logging.info(f"Duration: {total_duration:.2f}s")
            logging.info(f"Subtitles: {len(subtitles)} entries")
            logging.info(f"Max workers: {MAX_WORKERS}")
            
            processed_segments = []
            failed_segments = []
            
            # Handle single vs multiple segments
            if len(segments) == 1:
                # Single segment processing
                segment = segments[0]
                logging.info(f"Processing single segment with progress bar...")
                
                total_frames = int(segment['duration'] * FRAME_RATE)
                progress_bar = tqdm(
                    total=total_frames,
                    desc="Processing",
                    unit="frame",
                    dynamic_ncols=True
                )
                
                class ProgressTracker:
                    def __init__(self, pbar):
                        self.pbar = pbar
                        self.last_update = 0
                    
                    def update(self, current_frame):
                        if current_frame > self.last_update:
                            self.pbar.update(current_frame - self.last_update)
                            self.last_update = current_frame
                
                tracker = ProgressTracker(progress_bar)
                
                result = process_image_segment_with_progress(segment, subtitles, tracker)
                progress_bar.close()
                
                if result['success']:
                    processed_segments = [result]
                    logging.info(f"✓ Segment completed successfully")
                else:
                    failed_segments = [result]
                    logging.error(f"✗ Segment failed: {result.get('error', 'Unknown error')}")
            else:
                # Multiple segments processing
                logging.info(f"Processing {len(segments)} segments with {MAX_WORKERS} workers...")
                
                manager = mp.Manager()
                progress_queue = manager.Queue()
                segment_info_dict = {segment['index']: segment for segment in segments}
                
                progress_process = mp.Process(
                    target=progress_monitor,
                    args=(progress_queue, len(segments), segment_info_dict)
                )
                progress_process.start()
                active_multiprocessing_processes.append(progress_process)
                
                executor = None
                try:
                    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
                    active_executors.append(executor)
                    
                    logging.info(f"Starting {MAX_WORKERS} worker processes...")
                    
                    future_to_segment = {
                        executor.submit(process_segment_wrapper, (segment, subtitles, progress_queue)): segment 
                        for segment in segments
                    }
                    
                    for future in as_completed(future_to_segment):
                        segment = future_to_segment[future]
                        try:
                            result = future.result(timeout=3600)
                            if result['success']:
                                processed_segments.append(result)
                                logging.info(f"✓ Segment {result['segment_index']} completed successfully")
                            else:
                                failed_segments.append(result)
                                logging.error(f"✗ Segment {result['segment_index']} failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            logging.error(f"Exception in segment processing: {e}")
                            failed_segments.append({
                                'segment_index': segment['index'],
                                'error': str(e)
                            })
                
                except Exception as e:
                    logging.error(f"Error during parallel processing: {e}")
                finally:
                    # Cleanup
                    if executor:
                        try:
                            executor.shutdown(wait=True, cancel_futures=True)
                            if executor in active_executors:
                                active_executors.remove(executor)
                        except Exception as e:
                            logging.warning(f"Error shutting down executor: {e}")
                    
                    try:
                        progress_queue.put({'stop': True})
                    except:
                        pass
                    
                    if progress_process.is_alive():
                        progress_process.join(timeout=5)
                        if progress_process.is_alive():
                            logging.warning("Force terminating progress monitor")
                            progress_process.terminate()
                            progress_process.join(timeout=2)
                            if progress_process.is_alive():
                                progress_process.kill()
                                progress_process.join()
                    
                    if progress_process in active_multiprocessing_processes:
                        active_multiprocessing_processes.remove(progress_process)
                
                print("\n" * (len(segments) + 3))
                logging.info("All segments processing completed")
            
            if failed_segments:
                logging.error(f"Failed to process {len(failed_segments)} segments")
                for seg in failed_segments:
                    logging.error(f"  Segment {seg['segment_index']}: {seg.get('error', 'Unknown error')}")
                return
            
            # Create video without audio first
            if len(processed_segments) == 1:
                # Single segment - just copy/move the file
                single_segment = processed_segments[0]
                try:
                    import shutil
                    shutil.move(single_segment['output_file'], temp_video_output)
                    logging.info(f"✓ Moved single segment to {temp_video_output}")
                except Exception as e:
                    logging.error(f"Failed to move single segment: {e}")
                    return
            else:
                # Multiple segments - concatenate with GPU
                logging.info("🔗 Concatenating processed segments with GPU...")
                try:
                    concatenate_segments(processed_segments, temp_video_output)
                    logging.info("✓ Concatenation completed successfully")
                except Exception as e:
                    logging.error(f"Concatenation failed: {e}")
                    return
                
                # Clean up segment files
                for segment in processed_segments:
                    try:
                        os.unlink(segment['output_file'])
                        logging.info(f"🗑️ Cleaned up {segment['output_file']}")
                    except Exception as e:
                        logging.warning(f"Failed to clean up {segment['output_file']}: {e}")
            
            # Find audio files
            narration_audio, bgm_audio = find_audio_files(args.project_name, script_dir)
            
            # Add audio to video
            if narration_audio or bgm_audio:
                logging.info("🎧 Processing audio...")
                try:
                    add_audio_to_video(temp_video_output, narration_audio, bgm_audio, final_output, total_duration)
                    
                    # Clean up temporary video file
                    try:
                        os.unlink(temp_video_output)
                        logging.info(f"🗑️ Cleaned up temporary video: {temp_video_output}")
                    except Exception as e:
                        logging.warning(f"Failed to clean up temporary video: {e}")
                        
                except Exception as e:
                    logging.error(f"Failed to add audio: {e}")
                    # If audio processing fails, use the video without audio
                    try:
                        import shutil
                        shutil.move(temp_video_output, final_output)
                        logging.warning(f"⚠️  Using video without audio: {final_output}")
                    except Exception as e2:
                        logging.error(f"Failed to move video without audio: {e2}")
                        return
            else:
                # No audio files found - just rename temp video to final
                try:
                    import shutil
                    shutil.move(temp_video_output, final_output)
                    logging.info(f"✓ No audio files found, using video only: {final_output}")
                except Exception as e:
                    logging.error(f"Failed to move video: {e}")
                    return
            
            logging.info(f"✅ Video created successfully: {final_output}")
            
            # Check if output file exists and show size
            if os.path.exists(final_output):
                file_size = os.path.getsize(final_output) / (1024 * 1024)  # MB
                logging.info(f"📊 Output file size: {file_size:.2f} MB")
                logging.info(f"📁 Video saved at: {final_output}")
                
                # FORCE PREVIEW TO START
                if PREVIEW_MODE:
                    logging.info("🎬 LAUNCHING VIDEO PREVIEW NOW...")
                    play_video_preview(final_output)
                else:
                    logging.info(f"🎥 To preview the video, run: python {__file__} {args.project_name} --preview")
            else:
                logging.error("❌ Output file was not created!")
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        cleanup_processes()
        cleanup_contexts()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to create video: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def process_image_segment_with_progress(segment_info, subtitles, progress_tracker):
    """Process segment with direct progress tracking for single segment mode"""
    global active_processes, active_contexts
    
    segment_idx = segment_info['index']
    output_file = segment_info['output']
    segment_duration = segment_info['duration']
    segment_images = segment_info['images']
    global_start_time = segment_info['start_time']
    
    total_frames = int(segment_duration * FRAME_RATE)
    
    ctx = None
    ffmpeg_out = None
    
    try:
        logging.info(f"Processing image segment {segment_idx}: {total_frames} frames")
        logging.info(f"Segment {segment_idx} images: {[img['tag'] for img in segment_images]}")
        
        # Create OpenGL context
        ctx = moderngl.create_standalone_context(backend='egl')
        active_contexts.append(weakref.ref(ctx))
        
        # Load shaders
        vertex_shader = load_shader_file("vertex.glsl")
        fragment_shader = load_shader_file("fragment.glsl")
        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Create quad
        quad = ctx.buffer(np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4'))
        
        vao = ctx.vertex_array(prog, [(quad, '2f 2f', 'in_vert', 'in_uv')])
        
        # Create textures and framebuffer
        tex = ctx.texture((WIDTH, HEIGHT), 3)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Create subtitle texture (will be updated per frame)
        subtitle_tex = ctx.texture((64, 64), 4)  # Start with small texture
        subtitle_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((WIDTH, HEIGHT), 3)])
        
        # Setup FFmpeg output with NVIDIA GPU encoding
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FRAME_RATE),
            '-i', '-', '-an', 
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',                 # Fastest preset
            '-tune', 'll',                   # Low latency
            '-rc', 'cbr',                    # Constant bitrate
            '-b:v', '6M',                    # Lower bitrate for speed
            '-pix_fmt', 'yuv420p', 
            output_file
        ]
        
        logging.info(f"Starting FFmpeg for segment {segment_idx} with NVENC GPU encoding")
        
        ffmpeg_out = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Track the process for cleanup
        active_processes.append(ffmpeg_out)
        
        current_subtitle_text = ""
        
        # Generate frames
        for frame_idx in range(total_frames):
            # Check for interruption more frequently
            if frame_idx % 10 == 0:  # Check every 10 frames
                if ffmpeg_out.poll() is not None:
                    stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
                    logging.error(f"FFmpeg terminated early in segment {segment_idx}: {stderr_output}")
                    break
            
            frame_time = frame_idx / FRAME_RATE  # Time within this segment
            global_time = global_start_time + frame_time  # Global video time
            
            # Generate composite frame from images
            composite_frame = generate_composite_frame(segment_images, frame_time, global_start_time)
            
            # Validate composite frame
            if composite_frame is None or composite_frame.size == 0:
                logging.warning(f"Empty composite frame at time {frame_time} in segment {segment_idx}")
                composite_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            # Get subtitle for current time
            current_subtitle = get_subtitle_for_time(subtitles, global_time)
            subtitle_text = current_subtitle['text'] if current_subtitle else ""
            
            # Update subtitle texture if text changed
            if subtitle_text != current_subtitle_text:
                current_subtitle_text = subtitle_text
                if subtitle_text:
                    text_data, text_width, text_height = create_text_texture(subtitle_text, font_size=SUBTITLE_FONT_SIZE)
                    # Recreate texture with new dimensions
                    subtitle_tex = ctx.texture((text_width, text_height), 4)
                    subtitle_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    subtitle_tex.write(text_data.tobytes())
                else:
                    # Empty subtitle
                    empty_data = np.zeros((64, 64, 4), dtype=np.uint8)
                    subtitle_tex = ctx.texture((64, 64), 4)
                    subtitle_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    subtitle_tex.write(empty_data.tobytes())
                    text_width, text_height = 64, 64
            
            # Upload frame to GPU
            tex.write(composite_frame.tobytes())
            
            # Render with shaders
            fbo.use()
            tex.use(0)
            subtitle_tex.use(1)
            prog['tex'] = 0
            prog['subtitle_tex'] = 1
            prog['u_time'] = global_time
            
            # Set subtitle properties
            if subtitle_text:
                # Position subtitle at bottom center
                subtitle_scale_x = min(text_width / WIDTH * 0.8, 0.8)  # Max 80% of screen width
                subtitle_scale_y = text_height / HEIGHT * subtitle_scale_x * (WIDTH / text_width)
                subtitle_x = 0.5 - subtitle_scale_x / 2  # Center horizontally
                subtitle_y = 0.85 - subtitle_scale_y    # Near bottom
                
                prog['subtitle_size'] = [subtitle_scale_x, subtitle_scale_y]
                prog['subtitle_pos'] = [subtitle_x, subtitle_y]
                prog['show_subtitle'] = 1.0
            else:
                prog['subtitle_size'] = [0.0, 0.0]
                prog['subtitle_pos'] = [0.0, 0.0]
                prog['show_subtitle'] = 0.0
            
            vao.render(moderngl.TRIANGLE_STRIP)
            
            # Read processed frame
            out_data = fbo.read(components=3, alignment=1)
            
            try:
                ffmpeg_out.stdin.write(out_data)
                ffmpeg_out.stdin.flush()
            except BrokenPipeError as e:
                stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
                logging.error(f"Broken pipe in segment {segment_idx} at frame {frame_idx}: {stderr_output}")
                break
            except Exception as e:
                logging.error(f"Error writing to FFmpeg: {e}")
                break
            
            # Update progress
            progress_tracker.update(frame_idx + 1)
            
            # Check VRAM usage
            gpu_usage = get_gpu_memory_usage()
            if gpu_usage >= VRAM_THRESHOLD:
                logging.error(f"⚠️ VRAM threshold exceeded in segment {segment_idx}")
                break
        
        # Clean up FFmpeg
        if ffmpeg_out:
            try:
                ffmpeg_out.stdin.close()
                return_code = ffmpeg_out.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.warning(f"FFmpeg timeout for segment {segment_idx}, terminating")
                ffmpeg_out.terminate()
                try:
                    ffmpeg_out.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ffmpeg_out.kill()
                return_code = -1
            
            # Remove from active processes
            if ffmpeg_out in active_processes:
                active_processes.remove(ffmpeg_out)
        
        # Clean up OpenGL context
        if ctx:
            try:
                ctx.release()
            except:
                pass
        
        # Check FFmpeg return code
        if return_code != 0:
            stderr_output = ffmpeg_out.stderr.read().decode('utf-8') if ffmpeg_out else ""
            logging.error(f"FFmpeg failed for segment {segment_idx} with return code {return_code}: {stderr_output}")
            return {
                'success': False,
                'segment_index': segment_idx,
                'error': f"FFmpeg failed: {stderr_output}"
            }
        
        logging.info(f"Completed image segment {segment_idx}")
        return {
            'success': True,
            'segment_index': segment_idx,
            'output_file': output_file,
            'frames_processed': total_frames
        }
        
    except KeyboardInterrupt:
        logging.info(f"Segment {segment_idx} interrupted by user")
        raise
    except Exception as e:
        logging.error(f"Error processing image segment {segment_idx}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'segment_index': segment_idx,
            'error': str(e)
        }
    finally:
        # Ensure cleanup happens even on exception
        if ffmpeg_out and ffmpeg_out in active_processes:
            try:
                if ffmpeg_out.poll() is None:
                    ffmpeg_out.terminate()
                    ffmpeg_out.wait(timeout=5)
                active_processes.remove(ffmpeg_out)
            except:
                pass
        
        if ctx:
            try:
                ctx.release()
            except:
                pass

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"🏁 Program completed in {end_time - start_time:.2f} seconds.")