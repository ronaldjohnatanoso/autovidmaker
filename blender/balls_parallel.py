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

# Set to True to only process and preview first 10 seconds
PREVIEW_MODE = False

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
    
    # Find images directory
    images_dir = None
    for dir_name in [ 'images_1080p']:
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
    return image_timeline, total_duration

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

def process_image_segment(segment_info, progress_queue=None):
    """Process one video segment with shader effects applied to image timeline"""
    segment_idx = segment_info['index']
    output_file = segment_info['output']
    segment_duration = segment_info['duration']
    segment_images = segment_info['images']
    global_start_time = segment_info['start_time']
    
    total_frames = int(segment_duration * FRAME_RATE)
    
    try:
        logging.info(f"Processing image segment {segment_idx}: {total_frames} frames")
        logging.info(f"Segment {segment_idx} images: {[img['tag'] for img in segment_images]}")
        
        # Create OpenGL context
        ctx = moderngl.create_standalone_context(backend='egl')
        
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
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((WIDTH, HEIGHT), 3)])
        
        # Setup FFmpeg output with better error handling
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FRAME_RATE),
            '-i', '-', '-an', '-vcodec', 'libx264',  # Changed from h264_nvenc to libx264
            '-preset', 'fast',  # Added preset for faster encoding
            '-pix_fmt', 'yuv420p', output_file
        ]
        
        logging.info(f"Starting FFmpeg for segment {segment_idx}: {' '.join(ffmpeg_cmd)}")
        
        ffmpeg_out = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE, 
            stderr=subprocess.PIPE,  # Capture stderr instead of DEVNULL
            stdout=subprocess.PIPE
        )
        
        # Generate frames
        for frame_idx in range(total_frames):
            frame_time = frame_idx / FRAME_RATE  # Time within this segment
            global_time = global_start_time + frame_time  # Global video time
            
            # Generate composite frame from images
            composite_frame = generate_composite_frame(segment_images, frame_time, global_start_time)
            
            # Validate composite frame
            if composite_frame is None or composite_frame.size == 0:
                logging.warning(f"Empty composite frame at time {frame_time} in segment {segment_idx}")
                composite_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            # Upload frame to GPU
            tex.write(composite_frame.tobytes())
            
            # Render with shaders
            fbo.use()
            tex.use(0)
            prog['tex'] = 0
            prog['u_time'] = global_time  # Use global time for continuous effects
            
            vao.render(moderngl.TRIANGLE_STRIP)
            
            # Read processed frame
            out_data = fbo.read(components=3, alignment=1)
            
            # Check if FFmpeg process is still alive
            if ffmpeg_out.poll() is not None:
                # FFmpeg has terminated
                stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
                logging.error(f"FFmpeg terminated early in segment {segment_idx}: {stderr_output}")
                break
            
            try:
                ffmpeg_out.stdin.write(out_data)
                ffmpeg_out.stdin.flush()  # Add flush to ensure data is sent
            except BrokenPipeError as e:
                stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
                logging.error(f"Broken pipe in segment {segment_idx} at frame {frame_idx}: {stderr_output}")
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
        
        # Clean up
        ffmpeg_out.stdin.close()
        return_code = ffmpeg_out.wait()
        
        # Check FFmpeg return code
        if return_code != 0:
            stderr_output = ffmpeg_out.stderr.read().decode('utf-8')
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
        
    except Exception as e:
        logging.error(f"Error processing image segment {segment_idx}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'segment_index': segment_idx,
            'error': str(e)
        }

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
    max_timeouts = 30  # 30 seconds without updates = timeout
    
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
                    logging.warning("Progress monitor timed out - assuming all segments completed")
                    break
                continue
            except Exception as e:
                logging.warning(f"Progress monitor error: {e}")
                continue
    
    finally:
        for pbar in progress_bars.values():
            if not pbar.disable:
                pbar.close()

def concatenate_segments(processed_segments, final_output):
    """Concatenate processed segments using FFmpeg"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for segment in sorted(processed_segments, key=lambda x: x['segment_index']):
            f.write(f"file '{segment['output_file']}'\n")
    
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            final_output
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logging.info(f"Successfully concatenated segments to {final_output}")
        
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
    segment_info, progress_queue = args
    return process_image_segment(segment_info, progress_queue)

def main():
    parser = argparse.ArgumentParser(description='Create video from project images with shader effects')
    parser.add_argument('project_name', help='Name of the project folder')
    parser.add_argument('--preview', action='store_true', help='Create and preview first 10 seconds only')
    parser.add_argument('--output', help='Output video file path (default: project_name.mp4)')
    
    args = parser.parse_args()
    
    global PREVIEW_MODE
    PREVIEW_MODE = args.preview
    
    try:
        # Load project data
        image_timeline, total_duration = load_project_data(args.project_name)
        
        if PREVIEW_MODE:
            total_duration = min(10.0, total_duration)
            # Filter timeline for preview
            image_timeline = [img for img in image_timeline if img['start_time'] < total_duration]
            for img in image_timeline:
                img['end_time'] = min(img['end_time'], total_duration)
        
        # Set output file
        if args.output:
            final_output = args.output
        else:
            script_dir = Path(__file__).parent
            project_dir = script_dir.parent / "0-project-files" / args.project_name
            suffix = "_preview" if PREVIEW_MODE else ""
            final_output = str(project_dir / f"{args.project_name}{suffix}.mp4")
        
        logging.info(f"Creating video: {final_output}")
        logging.info(f"Duration: {total_duration:.2f}s")
        
        # Create temporary directory for segments
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create segments from image timeline
            segments = create_image_video_segments(image_timeline, total_duration, temp_dir)
            logging.info(f"Created {len(segments)} segments")
            
            # Create segment info dictionary
            segment_info_dict = {segment['index']: segment for segment in segments}
            
            # Process segments in parallel
            with mp.Manager() as manager:
                progress_queue = manager.Queue()
                
                # Start progress monitor
                progress_thread = threading.Thread(
                    target=progress_monitor,
                    args=(progress_queue, len(segments), segment_info_dict)
                )
                progress_thread.daemon = True
                progress_thread.start()
                
                processed_segments = []
                failed_segments = []
                
                # Process segments
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_segment = {
                        executor.submit(process_segment_wrapper, (segment, progress_queue)): segment 
                        for segment in segments
                    }
                    
                    for future in as_completed(future_to_segment):
                        segment = future_to_segment[future]
                        try:
                            result = future.result()
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
                
                # Wait for progress monitor with timeout
                progress_thread.join(timeout=10)
                if progress_thread.is_alive():
                    logging.warning("Progress monitor didn't finish - continuing anyway")
            
            print("\n" * (len(segments) + 2))
            
            if failed_segments:
                logging.error(f"Failed to process {len(failed_segments)} segments")
                for seg in failed_segments:
                    logging.error(f"  Segment {seg['segment_index']}: {seg.get('error', 'Unknown error')}")
                return
            
            # Concatenate segments
            logging.info("Concatenating processed segments...")
            concatenate_segments(processed_segments, final_output)
            
            # Clean up segment files
            for segment in processed_segments:
                try:
                    os.unlink(segment['output_file'])
                    logging.info(f"Cleaned up {segment['output_file']}")
                except Exception as e:
                    logging.warning(f"Failed to clean up {segment['output_file']}: {e}")
            
            logging.info(f"✓ Video created successfully: {final_output}")
            
            # Check if output file exists and show size
            if os.path.exists(final_output):
                file_size = os.path.getsize(final_output) / (1024 * 1024)  # MB
                logging.info(f"Output file size: {file_size:.2f} MB")
            else:
                logging.error("Output file was not created!")
            
            # Play preview if requested
            if PREVIEW_MODE:
                play_video_preview(final_output)
    
    except Exception as e:
        logging.error(f"Failed to create video: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Program completed in {end_time - start_time:.2f} seconds.")