import logging
import subprocess
import numpy as np
import moderngl
import psutil
import sys
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

WIDTH = 1920
HEIGHT = 1080
FRAME_RATE = 30
VRAM_THRESHOLD = 0.95  # 95%
SEGMENT_DURATION = 30  # seconds per segment
MAX_WORKERS = 5  # Leave one core free

# Define pulsing red shader
FRAGMENT_SHADER = '''
#version 330
uniform sampler2D tex;
uniform float u_time;
in vec2 uv;
out vec4 color;

void main() {
    vec4 c = texture(tex, uv);
    
    // Create a pulsing effect using sine wave
    float pulse = sin(u_time * 3.14159) * 0.5 + 0.5;
    
    // Apply red overlay
    vec3 red_overlay = vec3(pulse, 0.0, 0.0);
    vec3 result = mix(c.rgb, c.rgb + red_overlay, 0.3);
    
    color = vec4(result, c.a);
}
'''

VERTEX_SHADER = '''
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 uv;
void main() {
    uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
'''

def get_video_info(input_file):
    """Get video duration, fps, and frame count using ffprobe."""
    try:
        # Get duration
        duration_result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', input_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True
        )
        duration = float(duration_result.stdout.strip())
        
        # Get fps and frame count
        stream_result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,nb_frames',
             '-of', 'default=noprint_wrappers=1:nokey=1', input_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True
        )
        lines = stream_result.stdout.strip().split('\n')
        fps_fraction = lines[0]
        frame_count_str = lines[1] if len(lines) > 1 else 'N/A'
        
        # Calculate fps from fraction
        num, den = map(int, fps_fraction.split('/'))
        fps = num / den
        
        # Calculate frame count if not available
        if frame_count_str == 'N/A' or not frame_count_str:
            frame_count = int(duration * fps)
            logging.info(f"Frame count not available, calculated: {frame_count} frames")
        else:
            frame_count = int(frame_count_str)
        
        return duration, fps, frame_count
    except Exception as e:
        logging.error(f"Failed to get video info: {e}")
        raise

def create_segments(input_file, output_dir, segment_duration):
    """Split video into segments with exact frame boundaries."""
    duration, fps, total_frames = get_video_info(input_file)
    segments = []
    
    frames_per_segment = int(segment_duration * fps)
    
    for i in range(0, total_frames, frames_per_segment):
        start_frame = i
        end_frame = min(i + frames_per_segment, total_frames)
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        segment_file = os.path.join(output_dir, f"segment_{i:06d}.mp4")
        
        # Extract segment with exact timing and re-encode to ensure consistency
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file,
            '-ss', f'{start_time:.6f}',
            '-t', f'{end_time - start_time:.6f}',
            '-c:v', 'libx264', '-preset', 'ultrafast',  # Re-encode for consistency
            '-avoid_negative_ts', 'make_zero',
            segment_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        segments.append({
            'input': segment_file,
            'output': os.path.join(output_dir, f"processed_{i:06d}.mp4"),
            'start_time': start_time,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'index': i // frames_per_segment,
            'fps': fps
        })
    
    return segments, fps

def get_gpu_memory_usage():
    """Check GPU memory usage using nvidia-smi if available."""
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

def get_segment_frame_count(input_file):
    """Get the total number of frames in a segment."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-count_frames', '-show_entries', 'stream=nb_read_frames',
             '-of', 'csv=p=0', input_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True
        )
        return int(result.stdout.strip())
    except:
        # Fallback: estimate from duration and fps
        duration, fps, _ = get_video_info(input_file)
        return int(duration * fps)

def process_segment(segment_info, progress_queue=None):
    """Process a single video segment with individual progress tracking."""
    input_file = segment_info['input']
    output_file = segment_info['output']
    start_time = segment_info['start_time']
    segment_index = segment_info['index']
    fps = segment_info['fps']
    
    # Get total frames for this segment
    total_frames = get_segment_frame_count(input_file)
    
    try:
        logging.info(f"Processing segment {segment_index}: {input_file} ({total_frames} frames)")
        
        # Initialize OpenGL context for this process
        ctx = moderngl.create_standalone_context(backend='egl')
        
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        prog['u_time'] = 0.0
        
        quad = ctx.buffer(np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4'))
        
        vao = ctx.vertex_array(
            prog,
            [(quad, '2f 2f', 'in_vert', 'in_uv')]
        )
        
        tex = ctx.texture((WIDTH, HEIGHT), 3)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((WIDTH, HEIGHT), 3)])
        
        # Setup FFmpeg for this segment
        ffmpeg_in = subprocess.Popen(
            ['ffmpeg', '-hwaccel', 'cuda', '-c:v', 'h264_cuvid', '-i', input_file,
             '-vf', f'scale={WIDTH}:{HEIGHT}',
             '-f', 'rawvideo',
             '-pix_fmt', 'rgb24', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        
        ffmpeg_out = subprocess.Popen(
            ['ffmpeg',
             '-y', '-f', 'rawvideo',
             '-vcodec', 'rawvideo',
             '-pix_fmt', 'rgb24',
             '-s', f'{WIDTH}x{HEIGHT}',
             '-r', str(fps),
             '-i', '-', '-an',
             '-vcodec', 'h264_nvenc',
             '-pix_fmt', 'yuv420p',
             output_file],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        
        frame_count = 0
        
        while True:
            raw_frame = ffmpeg_in.stdout.read(WIDTH * HEIGHT * 3)
            if not raw_frame:
                break
            
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            tex.write(frame.tobytes())
            fbo.use()
            tex.use(0)
            prog['tex'] = 0
            prog['u_time'] = (start_time + frame_count / fps)
            vao.render(moderngl.TRIANGLE_STRIP)
            
            out_data = fbo.read(components=3, alignment=1)
            ffmpeg_out.stdin.write(out_data)
            
            # Update progress (use try-except for safety)
            if progress_queue:
                try:
                    progress_queue.put({
                        'segment_index': segment_index,
                        'frames_processed': frame_count + 1,
                        'total_frames': total_frames
                    })
                except Exception as e:
                    # If queue fails, continue without progress updates
                    pass
            
            # Check VRAM usage
            gpu_usage = get_gpu_memory_usage()
            if gpu_usage >= VRAM_THRESHOLD:
                logging.error(f"⚠️ VRAM threshold exceeded in segment {segment_index}")
                break
            
            frame_count += 1
        
        ffmpeg_in.terminate()
        ffmpeg_out.stdin.close()
        ffmpeg_out.wait()
        
        # Signal completion
        if progress_queue:
            try:
                progress_queue.put({
                    'segment_index': segment_index,
                    'completed': True,
                    'frames_processed': frame_count
                })
            except Exception:
                pass
        
        logging.info(f"Completed segment {segment_index}")
        return {
            'success': True,
            'segment_index': segment_index,
            'output_file': output_file,
            'frames_processed': frame_count
        }
        
    except Exception as e:
        logging.error(f"Error processing segment {segment_index}: {e}")
        if progress_queue:
            try:
                progress_queue.put({
                    'segment_index': segment_index,
                    'error': str(e)
                })
            except Exception:
                pass
        return {
            'success': False,
            'segment_index': segment_index,
            'error': str(e)
        }

def progress_monitor(progress_queue, num_segments, segment_info_dict):
    """Monitor and display progress for all segments."""
    # Create progress bars for each segment
    progress_bars = {}
    for segment_idx in range(num_segments):
        segment_info = segment_info_dict[segment_idx]
        total_frames = get_segment_frame_count(segment_info['input'])
        
        progress_bars[segment_idx] = tqdm(
            total=total_frames,
            desc=f"Segment {segment_idx:02d}",
            position=segment_idx,
            leave=True,
            unit="frame",
            dynamic_ncols=True
        )
    
    completed_segments = set()
    
    try:
        while len(completed_segments) < num_segments:
            try:
                update = progress_queue.get(timeout=1.0)
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
                    
                    # Update progress bar
                    current_pos = progress_bars[segment_idx].n
                    progress_bars[segment_idx].update(frames_processed - current_pos)
                    
                    # Update description with percentage
                    percentage = (frames_processed / total_frames) * 100 if total_frames > 0 else 0
                    progress_bars[segment_idx].set_description(
                        f"Segment {segment_idx:02d} ({percentage:.1f}%)"
                    )
            
            except queue.Empty:
                continue
            except Exception as e:
                logging.warning(f"Progress monitor error: {e}")
                continue
    
    finally:
        # Close any remaining progress bars
        for pbar in progress_bars.values():
            if not pbar.disable:
                pbar.close()

def process_segment_wrapper(args):
    """Wrapper function for multiprocessing."""
    segment_info, progress_queue = args
    return process_segment(segment_info, progress_queue)

def concatenate_segments(processed_segments, final_output):
    """Concatenate processed segments using FFmpeg."""
    # Create file list for FFmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for segment in sorted(processed_segments, key=lambda x: x['segment_index']):
            f.write(f"file '{segment['output_file']}'\n")
    
    try:
        # Concatenate segments
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            final_output
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logging.info(f"Successfully concatenated segments to {final_output}")
        
    finally:
        os.unlink(concat_file)

def main():
    input_file = "input.mp4"
    final_output = "output.mp4"
    
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found")
        return
    
    # Get actual video properties
    duration, fps, total_frames = get_video_info(input_file)
    logging.info(f"Input video: {duration:.2f}s, {fps:.2f}fps, {total_frames} frames")
    
    # Create temporary directory for segments
    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Creating segments in {temp_dir}")
        
        # Create segments with actual fps
        segments, actual_fps = create_segments(input_file, temp_dir, SEGMENT_DURATION)
        logging.info(f"Created {len(segments)} segments at {actual_fps:.2f}fps")
        
        # Create segment info dictionary for progress monitor
        segment_info_dict = {segment['index']: segment for segment in segments}
        
        # Create shared progress queue using Manager
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            
            # Start progress monitor in a separate thread
            progress_thread = threading.Thread(
                target=progress_monitor,
                args=(progress_queue, len(segments), segment_info_dict)
            )
            progress_thread.daemon = True
            progress_thread.start()
            
            # Process segments in parallel
            processed_segments = []
            failed_segments = []
            
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all segment processing jobs with progress queue
                future_to_segment = {
                    executor.submit(process_segment_wrapper, (segment, progress_queue)): segment 
                    for segment in segments
                }
                
                # Collect results
                for future in as_completed(future_to_segment):
                    segment = future_to_segment[future]
                    try:
                        result = future.result()
                        if result['success']:
                            processed_segments.append(result)
                        else:
                            failed_segments.append(result)
                    except Exception as e:
                        logging.error(f"Exception in segment processing: {e}")
                        failed_segments.append({
                            'segment_index': segment['index'],
                            'error': str(e)
                        })
            
            # Wait for progress monitor to finish
            progress_thread.join(timeout=5)
        
        print("\n" * (len(segments) + 2))  # Clear progress bars area
        
        # Check if all segments processed successfully
        if failed_segments:
            logging.error(f"Failed to process {len(failed_segments)} segments")
            for failed in failed_segments:
                logging.error(f"Segment {failed['segment_index']}: {failed.get('error', 'Unknown error')}")
            return
        
        # Concatenate processed segments
        logging.info("Concatenating processed segments")
        concatenate_segments(processed_segments, final_output)
        
        # Clean up segment files
        for segment in processed_segments:
            try:
                os.unlink(segment['output_file'])
            except:
                pass

if __name__ == '__main__':
    logging.info("Starting parallel video processing with individual segment progress.")
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Program completed in {end_time - start_time:.2f} seconds.")