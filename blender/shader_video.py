import ffmpeg
import numpy as np
from PIL import Image
import moderngl
import os
from tqdm import tqdm
import subprocess
import tempfile
import psutil
import gc
import time

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);  // Flip UV coordinates
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
uniform float time;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 color = texture(tex, v_uv);

    // Time-based scanline effect to maintain continuity across segments
    float scanline = sin((v_uv.y * 800.0) + (time * 10.0)) * 0.1;
    color.rgb -= scanline;

    f_color = color;
}
"""

# Global constants for VRAM management
TOTAL_VRAM_MB = 4096  # RTX 3050 total VRAM
KILL_SWITCH_THRESHOLD = 0.90  # 90% VRAM usage threshold
SAFETY_BUFFER = 0.75  # Use only 75% of available VRAM per segment

def get_gpu_memory_info():
    """Get available GPU memory in MB"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = info.used // (1024 * 1024)
        free_mb = info.free // (1024 * 1024)
        total_mb = info.total // (1024 * 1024)
        return free_mb, total_mb, used_mb
    except:
        # Fallback estimation for RTX 3050 (4GB)
        used_estimate = 1024  # Conservative estimate
        free_estimate = TOTAL_VRAM_MB - used_estimate
        return free_estimate, TOTAL_VRAM_MB, used_estimate

def check_vram_kill_switch():
    """Check if VRAM usage exceeds kill switch threshold"""
    _, total_vram, used_vram = get_gpu_memory_info()
    usage_percent = used_vram / total_vram
    
    if usage_percent >= KILL_SWITCH_THRESHOLD:
        print(f"🚨 KILL SWITCH ACTIVATED! VRAM usage: {used_vram}MB/{total_vram}MB ({usage_percent*100:.1f}%)")
        print("Stopping script to prevent system crash...")
        return True
    return False

def estimate_frame_vram_usage(width, height):
    """Estimate VRAM usage per frame in MB with detailed breakdown"""
    # Input texture: width * height * 3 bytes (RGB)
    input_texture = width * height * 3
    
    # Mipmaps: approximately 1.33x original texture size
    mipmap_overhead = input_texture * 0.33
    
    # Framebuffer: width * height * 3 bytes (RGB output)
    framebuffer = width * height * 3
    
    # OpenGL overhead and temporary buffers
    gl_overhead = (input_texture + framebuffer) * 0.2
    
    total_bytes = input_texture + mipmap_overhead + framebuffer + gl_overhead
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'total_mb': total_mb,
        'input_texture_mb': input_texture / (1024 * 1024),
        'mipmap_mb': mipmap_overhead / (1024 * 1024),
        'framebuffer_mb': framebuffer / (1024 * 1024),
        'overhead_mb': gl_overhead / (1024 * 1024)
    }

def calculate_optimal_segment_frames(width, height, total_frames, fps):
    """Calculate optimal frames per segment with detailed VRAM analysis"""
    # Check kill switch before starting
    if check_vram_kill_switch():
        raise RuntimeError("VRAM usage too high to continue safely")
    
    free_vram, total_vram, used_vram = get_gpu_memory_info()
    frame_vram = estimate_frame_vram_usage(width, height)
    
    print(f"\n🔍 VRAM Analysis:")
    print(f"  Total VRAM: {total_vram}MB")
    print(f"  Used VRAM: {used_vram}MB ({used_vram/total_vram*100:.1f}%)")
    print(f"  Free VRAM: {free_vram}MB ({free_vram/total_vram*100:.1f}%)")
    print(f"\n📏 Frame VRAM Breakdown ({width}x{height}):")
    print(f"  Input Texture: {frame_vram['input_texture_mb']:.2f}MB")
    print(f"  Mipmaps: {frame_vram['mipmap_mb']:.2f}MB")
    print(f"  Framebuffer: {frame_vram['framebuffer_mb']:.2f}MB")
    print(f"  GL Overhead: {frame_vram['overhead_mb']:.2f}MB")
    print(f"  Total per frame: {frame_vram['total_mb']:.2f}MB")
    
    # Calculate safe VRAM to use (with buffer)
    safe_vram = free_vram * SAFETY_BUFFER
    
    # Calculate frames per segment
    frames_per_segment = int(safe_vram / frame_vram['total_mb'])
    frames_per_segment = max(1, min(frames_per_segment, total_frames))
    
    # Calculate estimated VRAM usage for this segment
    segment_vram_usage = frames_per_segment * frame_vram['total_mb']
    
    print(f"\n📊 Segment Planning:")
    print(f"  Safe VRAM to use: {safe_vram:.2f}MB ({SAFETY_BUFFER*100:.0f}% of free)")
    print(f"  Frames per segment: {frames_per_segment}")
    print(f"  Estimated segment VRAM: {segment_vram_usage:.2f}MB")
    print(f"  Segment duration: {frames_per_segment/fps:.2f}s")
    
    return frames_per_segment, frame_vram['total_mb']

def log_segment_vram_status(segment_idx, frames_count, estimated_vram_per_frame):
    """Log VRAM status at the start of each segment"""
    if check_vram_kill_switch():
        raise RuntimeError(f"VRAM kill switch activated during segment {segment_idx}")
    
    free_vram, total_vram, used_vram = get_gpu_memory_info()
    estimated_segment_usage = frames_count * estimated_vram_per_frame
    
    print(f"\n🎬 Segment {segment_idx} VRAM Check:")
    print(f"  Current VRAM usage: {used_vram}MB/{total_vram}MB ({used_vram/total_vram*100:.1f}%)")
    print(f"  Available VRAM: {free_vram}MB")
    print(f"  Frames to process: {frames_count}")
    print(f"  Estimated usage: {estimated_segment_usage:.2f}MB")
    print(f"  Safety margin: {free_vram - estimated_segment_usage:.2f}MB")
    
    if estimated_segment_usage > free_vram:
        print(f"⚠️  WARNING: Estimated usage exceeds available VRAM!")
        return False
    return True

def cleanup_gl_resources(*resources):
    """Clean up OpenGL resources safely"""
    for resource in resources:
        if resource is not None:
            try:
                resource.release()
            except:
                pass  # Resource might already be released

def get_video_info(input_file):
    """Get video metadata"""
    probe = ffmpeg.probe(input_file)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = eval(video_stream['r_frame_rate'])
    duration = float(video_stream['duration'])
    total_frames = int(duration * fps)
    
    return width, height, fps, duration, total_frames

def decode_video_segment(input_file, start_time, duration):
    """Decode a segment of video frames"""
    process = (
        ffmpeg
        .input(input_file, ss=start_time, t=duration)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.DEVNULL)  # Suppress FFmpeg logs
    )
    
    probe = ffmpeg.probe(input_file, cmd='ffprobe')
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    frames = []
    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        frames.append(frame)
    
    process.wait()
    return frames

def run_shader_on_frame(ctx, prog, vao, frame, global_time):
    """Render frame texture with shader, read back pixels"""
    height, width = frame.shape[:2]
    
    # Check VRAM before processing frame
    if check_vram_kill_switch():
        raise RuntimeError("VRAM kill switch activated during frame processing")

    tex = None
    fbo = None
    
    try:
        # Create texture from frame
        tex = ctx.texture((width, height), 3, frame.tobytes())
        tex.build_mipmaps()

        # Create framebuffer to render to
        fbo = ctx.simple_framebuffer((width, height))
        fbo.use()

        ctx.clear(0.0, 0.0, 0.0, 0.0)
        tex.use()

        # Set time uniform for temporal continuity
        prog['tex'].value = 0
        prog['time'].value = global_time

        # Render fullscreen quad
        vao.render()

        # Read pixels from framebuffer
        data = fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img = np.flip(img, axis=0)  # Flip vertically
        
        return img
        
    finally:
        # Always clean up GPU resources
        cleanup_gl_resources(tex, fbo)

def process_video_segments(input_file, output_file):
    """Process video in segments and concatenate"""
    try:
        # Initial VRAM check
        if check_vram_kill_switch():
            raise RuntimeError("Initial VRAM usage too high to start processing")
        
        # Get video info
        width, height, fps, duration, total_frames = get_video_info(input_file)
        print(f"\n🎥 Video Info: {width}x{height}, {fps}fps, {duration:.2f}s, {total_frames} frames")
        
        # Calculate optimal segment size
        frames_per_segment, vram_per_frame = calculate_optimal_segment_frames(width, height, total_frames, fps)
        
        # Create OpenGL context
        ctx = moderngl.create_context(standalone=True, backend='egl')
        
        # Setup fullscreen quad
        vertices = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
            -1,  1,  0, 1,
            -1,  1,  0, 1,
             1, -1,  1, 0,
             1,  1,  1, 1,
        ], dtype='f4')
        
        vbo = ctx.buffer(vertices.tobytes())
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vao = ctx.vertex_array(prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])
        
        # Create temporary directory for segments
        temp_dir = tempfile.mkdtemp()
        segment_files = []
        
        # Progress tracking
        total_frames_processed = 0
        start_time = time.time()
        
        try:
            # Process segments
            current_time = 0
            segment_index = 0
            segment_duration = frames_per_segment / fps
            
            while current_time < duration:
                remaining_time = duration - current_time
                current_segment_duration = min(segment_duration, remaining_time)
                expected_frames = int(current_segment_duration * fps)
                
                # Log VRAM status for this segment
                if not log_segment_vram_status(segment_index + 1, expected_frames, vram_per_frame):
                    print("⚠️  Reducing segment size due to VRAM constraints")
                    expected_frames = max(1, expected_frames // 2)
                    current_segment_duration = expected_frames / fps
                
                # Progress info
                elapsed_time = time.time() - start_time
                video_seconds_processed = current_time
                progress_percent = (current_time / duration) * 100
                
                print(f"\n🎬 Processing segment {segment_index + 1}")
                print(f"   📊 Progress: {progress_percent:.1f}% ({video_seconds_processed:.1f}s/{duration:.1f}s)")
                print(f"   📈 Frames processed: {total_frames_processed}/{total_frames}")
                print(f"   ⏱️  Elapsed time: {elapsed_time:.1f}s")
                print(f"   Time range: {current_time:.2f}s - {current_time + current_segment_duration:.2f}s")
                print(f"   Expected frames: {expected_frames}")
                
                # Decode segment frames
                frames = decode_video_segment(input_file, current_time, current_segment_duration)
                
                if not frames:
                    break
                
                actual_frames = len(frames)
                print(f"   Actual frames decoded: {actual_frames}")
                
                # Create output for this segment
                segment_output = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
                segment_files.append(segment_output)
                
                # Encode this segment (suppress FFmpeg logs)
                process_out = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
                    .output(segment_output, pix_fmt='yuv420p', vcodec='libx264', crf=18, preset='fast')
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stderr=subprocess.DEVNULL)
                )
                
                # Process frames in this segment with progress tracking
                for frame_idx, frame in enumerate(frames):
                    global_time = current_time + (frame_idx / fps)  # Maintain global time continuity
                    out_frame = run_shader_on_frame(ctx, prog, vao, frame, global_time)
                    process_out.stdin.write(out_frame.tobytes())
                    
                    # Update progress counters
                    total_frames_processed += 1
                    
                    # Show frame progress every 30 frames
                    if frame_idx % 30 == 0:
                        frame_progress = (frame_idx + 1) / actual_frames * 100
                        print(f"     Frame {frame_idx + 1}/{actual_frames} ({frame_progress:.1f}%) - Total: {total_frames_processed}/{total_frames}")
                    
                    # Periodic VRAM check during frame processing
                    if frame_idx % 10 == 0 and check_vram_kill_switch():
                        process_out.stdin.close()
                        process_out.terminate()
                        raise RuntimeError(f"VRAM kill switch activated during segment {segment_index + 1}, frame {frame_idx}")
                
                process_out.stdin.close()
                process_out.wait()
                
                # Aggressive cleanup after each segment
                del frames
                gc.collect()
                ctx.gc()  # ModernGL garbage collection
                
                # Final VRAM and progress check after segment
                free_vram, total_vram, used_vram = get_gpu_memory_info()
                elapsed_time = time.time() - start_time
                avg_fps = total_frames_processed / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (total_frames - total_frames_processed) / avg_fps if avg_fps > 0 else 0
                
                print(f"   Segment completed. VRAM: {used_vram}MB/{total_vram}MB ({used_vram/total_vram*100:.1f}%)")
                print(f"   Processing speed: {avg_fps:.1f} fps | ETA: {eta_seconds/60:.1f} minutes")
                
                current_time += current_segment_duration
                segment_index += 1
            
            # Final progress summary
            total_elapsed = time.time() - start_time
            avg_fps_final = total_frames_processed / total_elapsed
            print(f"\n✅ Processing completed!")
            print(f"   Total frames processed: {total_frames_processed}")
            print(f"   Total video seconds: {duration:.1f}s")
            print(f"   Total elapsed time: {total_elapsed:.1f}s")
            print(f"   Average processing speed: {avg_fps_final:.1f} fps")
            
            # Concatenate all segments
            print("\n🔗 Concatenating segments...")
            concat_segments(segment_files, output_file)
            
        finally:
            # Clean up temporary files
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    os.remove(segment_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            
            # Clean up OpenGL resources
            cleanup_gl_resources(vao, vbo, prog)
            ctx.release()
            
    except RuntimeError as e:
        print(f"❌ Processing stopped: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

def concat_segments(segment_files, output_file):
    """Concatenate video segments using ffmpeg"""
    # Create concat file
    concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    
    try:
        for segment_file in segment_files:
            concat_file.write(f"file '{segment_file}'\n")
        concat_file.close()
        
        # Concatenate using ffmpeg (suppress logs)
        (
            ffmpeg
            .input(concat_file.name, format='concat', safe=0)
            .output(output_file, c='copy')
            .overwrite_output()
            .run(pipe_stderr=subprocess.DEVNULL)
        )
        
    finally:
        os.unlink(concat_file.name)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python shader_video.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Processing video: {input_file}")
    process_video_segments(input_file, output_file)
    print(f"Output video saved as: {output_file}")