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

# === VRAM MANAGEMENT CONSTANTS ===
TOTAL_VRAM_MB = 4096  # RTX 3050 total VRAM
KILL_SWITCH_THRESHOLD = 0.95  # 95% VRAM usage threshold - immediate stop
EARLY_ABORT_THRESHOLD = 0.90  # 90% VRAM usage - stop loading more textures (was 0.!)
TARGET_VRAM_USAGE = 0.85      # Target 85% of total VRAM for aggressive usage
SAFETY_BUFFER = 0.85          # Use 85% of available VRAM per segment

# === VRAM BUDGET CONSTANTS ===
BASELINE_VRAM_USAGE_MB = 300  # OpenGL context overhead
DRIVER_OVERHEAD_MB = 500      # Extra buffer for driver unpredictability  
SAFETY_MARGIN_MB = 300        # Additional safety margin
CONTEXT_CREATION_OVERHEAD_MB = 50  # Extra VRAM used during context creation

# === SEGMENT SIZE LIMITS ===
MIN_FRAMES_PER_SEGMENT = 100   # Minimum viable batch size
MAX_FRAMES_PER_SEGMENT = 2000  # Much higher limit - let VRAM be the constraint
SAFETY_FACTOR = 0.90          # Use 90% of calculated segment size (was 0.75)

# === PARALLEL PROCESSING CONSTANTS ===
PARALLEL_BATCH_SIZE = 8       # Number of frames to process simultaneously
PROGRESS_UPDATE_INTERVAL = 100  # Update progress every N frames
FREQUENT_VRAM_CHECK_INTERVAL = 50   # Check VRAM every N frames during loading
CONSOLE_SPAM_REDUCTION = 200  # Update console every N frames for ultra parallel

# === CLEANUP AND TIMING CONSTANTS ===
CLEANUP_SLEEP_TIME = 0.1      # Time to wait after cleanup operations
DOUBLE_GC_CLEANUP = True      # Whether to run garbage collection twice
CONTEXT_RECREATION_SLEEP = 0.02  # Time to wait after VRAM cleanup

# === FRAME VRAM ESTIMATION CONSTANTS ===
MIPMAP_OVERHEAD_FACTOR = 0.33 # Mipmaps add 33% to texture size
GL_OVERHEAD_FACTOR = 0.2      # OpenGL overhead is 20% of texture + framebuffer
BYTES_PER_RGB_PIXEL = 3       # RGB = 3 bytes per pixel

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
        print(f"   Threshold: {KILL_SWITCH_THRESHOLD*100:.0f}%")
        print("Stopping script to prevent system crash...")
        return True
    return False

def estimate_frame_vram_usage(width, height):
    """Estimate VRAM usage per frame in MB with detailed breakdown"""
    # Input texture: width * height * 3 bytes (RGB)
    input_texture = width * height * BYTES_PER_RGB_PIXEL
    
    # Mipmaps: approximately 1.33x original texture size
    mipmap_overhead = input_texture * MIPMAP_OVERHEAD_FACTOR
    
    # Framebuffer: width * height * 3 bytes (RGB output)
    framebuffer = width * height * BYTES_PER_RGB_PIXEL
    
    # OpenGL overhead and temporary buffers
    gl_overhead = (input_texture + framebuffer) * GL_OVERHEAD_FACTOR
    
    total_bytes = input_texture + mipmap_overhead + framebuffer + gl_overhead
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'total_mb': total_mb,
        'input_texture_mb': input_texture / (1024 * 1024),
        'mipmap_mb': mipmap_overhead / (1024 * 1024),
        'framebuffer_mb': framebuffer / (1024 * 1024),
        'overhead_mb': gl_overhead / (1024 * 1024)
    }

def calculate_max_segment_size(width, height):
    """Calculate segment size based on EMPIRICAL measurements"""
    return calculate_empirical_segment_size(width, height)

def quick_vram_based_segment_size(width, height):
    """Quick VRAM-based segment calculation without full empirical measurement"""
    
    # Get current VRAM state
    free_vram, total_vram, used_vram = get_gpu_memory_info()
    
    print(f"🔥 QUICK VRAM CALCULATION:")
    print(f"   Total VRAM: {total_vram}MB")
    print(f"   Currently used: {used_vram}MB")
    print(f"   Currently free: {free_vram}MB")
    
    # Estimate per-frame VRAM (more accurate than old constants)
    bytes_per_pixel = 3  # RGB
    pixels_per_frame = width * height
    
    # Texture + mipmaps + framebuffer + overhead
    texture_mb = (pixels_per_frame * bytes_per_pixel) / (1024 * 1024)
    mipmap_mb = texture_mb * 0.33  # 33% overhead for mipmaps
    framebuffer_mb = texture_mb  # Same size as texture
    overhead_mb = (texture_mb + framebuffer_mb) * 0.2  # 20% OpenGL overhead
    
    total_per_frame = texture_mb + mipmap_mb + framebuffer_mb + overhead_mb
    
    # Use 90% of available VRAM aggressively
    target_vram = total_vram * 0.90
    available_for_frames = target_vram - used_vram
    
    # Calculate max frames
    max_frames = int(available_for_frames / total_per_frame)
    
    # Apply safety factor
    safe_frames = int(max_frames * 0.85)  # 85% of calculated max
    
    # Clamp to reasonable bounds
    final_frames = max(100, min(safe_frames, 1000))  # Between 100-1000 frames
    
    print(f"   VRAM per frame: {total_per_frame:.2f}MB")
    print(f"   Available for frames: {available_for_frames:.1f}MB")
    print(f"   Max frames (calculated): {max_frames}")
    print(f"   Safe frames (85%): {safe_frames}")
    print(f"   FINAL SEGMENT SIZE: {final_frames} frames")
    
    return final_frames

def adaptive_segment_sizing(width, height, failed_frames=None):
    """Adaptively find the sweet spot by learning from failures"""
    if failed_frames:
        print(f"🔧 ADAPTIVE SIZING: Previous attempt failed at {failed_frames} frames")
        adaptive_frames = int(failed_frames * SAFETY_FACTOR)
        print(f"   Trying {adaptive_frames} frames ({SAFETY_FACTOR*100:.0f}% of failed amount)")
        return adaptive_frames
    
    # Use quick VRAM-based calculation for speed
    return quick_vram_based_segment_size(width, height)

def process_frames_batch_with_monitoring(ctx, prog, vao, frames, start_time, fps):
    """Batch process with real-time VRAM monitoring and early abort"""
    
    print(f"   🎯 BATCH MODE: Loading {len(frames)} frames with VRAM monitoring...")
    
    textures = []
    framebuffers = []
    
    start_vram = get_gpu_memory_info()[2]
    
    try:
        height, width = frames[0].shape[:2]
        
        # === PHASE 1: CAREFUL LOADING WITH MONITORING ===
        for frame_idx, frame in enumerate(frames):
            # Check VRAM BEFORE each texture load
            current_vram = get_gpu_memory_info()[2]
            vram_percent = current_vram / TOTAL_VRAM_MB
            
            # Progress updates using constant
            if frame_idx % FREQUENT_VRAM_CHECK_INTERVAL == 0:
                print(f"     Loading texture {frame_idx + 1}/{len(frames)} - VRAM: {current_vram}MB ({vram_percent*100:.1f}%)")
            
            # Early abort using constant threshold
            if vram_percent >= EARLY_ABORT_THRESHOLD:
                print(f"     ⚠️  EARLY ABORT: VRAM at {vram_percent*100:.1f}% >= {EARLY_ABORT_THRESHOLD*100:.0f}% - stopping at frame {frame_idx}")
                frames = frames[:frame_idx]
                break
            
            # Kill switch check
            if check_vram_kill_switch():
                print(f"     🚨 KILL SWITCH: Stopping at frame {frame_idx}")
                frames = frames[:frame_idx]
                break
            
            # Load texture and framebuffer
            tex = ctx.texture((width, height), 3, frame.tobytes())
            tex.build_mipmaps()
            textures.append(tex)
            
            fbo = ctx.simple_framebuffer((width, height))
            framebuffers.append(fbo)
        
        load_end_vram = get_gpu_memory_info()[2]
        vram_loaded = load_end_vram - start_vram
        actual_frames_loaded = len(textures)
        
        print(f"   ✅ Loaded {actual_frames_loaded} textures: {vram_loaded}MB VRAM ({start_vram}MB → {load_end_vram}MB)")
        
        if actual_frames_loaded == 0:
            return []
        
        # === PHASE 2: PROCESS LOADED FRAMES ===
        print(f"   🎬 BATCH MODE: Processing {actual_frames_loaded} frames...")
        
        processed_frames = []
        
        for frame_idx, (tex, fbo) in enumerate(zip(textures, framebuffers)):
            if frame_idx % PROGRESS_UPDATE_INTERVAL == 0:
                current_vram = get_gpu_memory_info()[2]
                print(f"     Processing frame {frame_idx + 1}/{actual_frames_loaded} - VRAM: {current_vram}MB")
            
            # Process this frame
            global_time = start_time + (frame_idx / fps)
            
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0, 0.0)
            tex.use()
            
            prog['tex'].value = 0
            prog['time'].value = global_time
            
            vao.render()
            
            # Read result
            data = fbo.read(components=3, alignment=1)
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            img = np.flip(img, axis=0)
            
            processed_frames.append(img)
        
        process_end_vram = get_gpu_memory_info()[2]
        print(f"   ✅ Processed {len(processed_frames)} frames: VRAM: {process_end_vram}MB")
        
        return processed_frames
        
    finally:
        # === PHASE 3: CLEANUP ===
        print(f"   💥 Cleaning up {len(textures)} textures + {len(framebuffers)} framebuffers...")
        
        cleanup_start_vram = get_gpu_memory_info()[2]
        
        for tex in textures:
            tex.release()
        for fbo in framebuffers:
            fbo.release()
        
        textures.clear()
        framebuffers.clear()
        del textures, framebuffers
        ctx.finish()
        ctx.gc()
        if DOUBLE_GC_CLEANUP:
            ctx.gc()
            gc.collect()
            gc.collect()
        
        time.sleep(CLEANUP_SLEEP_TIME)
        
        cleanup_end_vram = get_gpu_memory_info()[2]
        vram_freed = cleanup_start_vram - cleanup_end_vram
        
        print(f"   ✅ Cleanup: {vram_freed}MB freed ({cleanup_start_vram}MB → {cleanup_end_vram}MB)")

def process_frames_batch_ultra_parallel(ctx, prog, vao, frames, start_time, fps):
    """ULTRA PARALLEL: Queue ALL render commands, let GPU work in parallel, then read ALL results"""
    
    print(f"   🚀 ULTRA PARALLEL MODE: Loading {len(frames)} frames...")
    
    textures = []
    framebuffers = []
    start_vram = get_gpu_memory_info()[2]
    
    try:
        height, width = frames[0].shape[:2]
        
        # === PHASE 1: LOAD ALL TEXTURES ===
        print(f"   📥 Loading ALL {len(frames)} textures to VRAM...")
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx % PROGRESS_UPDATE_INTERVAL == 0:
                current_vram = get_gpu_memory_info()[2]
                print(f"     Loading {frame_idx + 1}/{len(frames)} - VRAM: {current_vram}MB")
                
                if check_vram_kill_switch():
                    print(f"     🚨 KILL SWITCH at frame {frame_idx}")
                    frames = frames[:frame_idx]
                    break
            
            tex = ctx.texture((width, height), 3, frame.tobytes())
            tex.build_mipmaps()
            textures.append(tex)
            
            fbo = ctx.simple_framebuffer((width, height))
            framebuffers.append(fbo)
        
        actual_frames = len(textures)
        load_end_vram = get_gpu_memory_info()[2]
        vram_loaded = load_end_vram - start_vram
        
        print(f"   ✅ ALL textures loaded: {actual_frames} frames, {vram_loaded}MB VRAM ({start_vram}MB → {load_end_vram}MB)")
        
        if actual_frames == 0:
            return []
        
        # === PHASE 2: ULTRA PARALLEL COMMAND QUEUING ===
        print(f"   🔥 ULTRA PARALLEL: Queueing ALL {actual_frames} render commands (NO WAITING)...")
        
        queue_start = time.time()
        
        # Queue ALL render commands without any synchronization
        for frame_idx, (tex, fbo) in enumerate(zip(textures, framebuffers)):
            global_time = start_time + (frame_idx / fps)
            
            # Queue render command - GPU will pipeline/parallelize these
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0, 0.0)
            tex.use()
            prog['tex'].value = 0
            prog['time'].value = global_time
            vao.render()
            
            # NO ctx.finish(), NO fbo.read() - just queue and continue!
            
            # Progress using constant to reduce console spam
            if frame_idx % CONSOLE_SPAM_REDUCTION == 0:
                print(f"     🚀 Queued commands for frame {frame_idx + 1}/{actual_frames}")
        
        queue_time = time.time() - queue_start
        queue_vram = get_gpu_memory_info()[2]
        
        print(f"   ⚡ ALL {actual_frames} commands queued in {queue_time:.3f}s - VRAM: {queue_vram}MB")
        print(f"   🔥 GPU is now working on ALL frames in parallel...")
        
        # === PHASE 3: SINGLE SYNCHRONIZATION POINT ===
        print(f"   ⏱️  Waiting for GPU to complete ALL parallel rendering...")
        
        render_start = time.time()
        ctx.finish()  # This is where ALL the parallel magic happens!
        render_time = time.time() - render_start
        
        render_vram = get_gpu_memory_info()[2]
        effective_fps = actual_frames / render_time if render_time > 0 else 0
        
        print(f"   🚀 GPU COMPLETED ALL rendering in {render_time:.3f}s!")
        print(f"   ⚡ Effective processing speed: {effective_fps:.1f} fps")
        print(f"   💾 Post-render VRAM: {render_vram}MB")
        
        # === PHASE 4: READ ALL RESULTS ===
        print(f"   📖 Reading ALL {actual_frames} results from completed renders...")
        
        processed_frames = []
        read_start = time.time()
        
        for frame_idx, fbo in enumerate(framebuffers):
            # GPU already finished rendering - this is just reading memory
            data = fbo.read(components=3, alignment=1)
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            img = np.flip(img, axis=0)
            processed_frames.append(img)
            
            # Progress using constant
            if frame_idx % CONSOLE_SPAM_REDUCTION == 0:
                print(f"     📖 Read result {frame_idx + 1}/{actual_frames}")
        
        read_time = time.time() - read_start
        final_vram = get_gpu_memory_info()[2]
        
        print(f"   📖 Read ALL results in {read_time:.3f}s")
        
        # === PERFORMANCE SUMMARY ===
        total_time = queue_time + render_time + read_time
        overall_fps = actual_frames / total_time if total_time > 0 else 0
        parallel_efficiency = effective_fps / overall_fps if overall_fps > 0 else 1
        
        print(f"\n   🎯 ULTRA PARALLEL PERFORMANCE SUMMARY:")
        print(f"      Queue time: {queue_time:.3f}s")
        print(f"      Render time: {render_time:.3f}s ({effective_fps:.1f} fps)")
        print(f"      Read time: {read_time:.3f}s")
        print(f"      Total time: {total_time:.3f}s ({overall_fps:.1f} fps overall)")
        print(f"      Parallel efficiency: {parallel_efficiency:.1f}x")
        print(f"      Final VRAM: {final_vram}MB")
        
        return processed_frames
        
    finally:
        # === PHASE 5: NUCLEAR CLEANUP ===
        print(f"   💥 ULTRA CLEANUP: Nuking {len(textures)} textures + {len(framebuffers)} framebuffers...")
        
        cleanup_start_vram = get_gpu_memory_info()[2]
        cleanup_start = time.time()
        
        # Release everything
        for tex in textures:
            tex.release()
        for fbo in framebuffers:
            fbo.release()
        
        textures.clear()
        framebuffers.clear()
        del textures, framebuffers
        
        # Aggressive cleanup using constants
        ctx.finish()
        ctx.gc()
        if DOUBLE_GC_CLEANUP:
            ctx.gc()
            gc.collect()
            gc.collect()
        
        time.sleep(CLEANUP_SLEEP_TIME)
        
        cleanup_time = time.time() - cleanup_start
        cleanup_end_vram = get_gpu_memory_info()[2]
        vram_freed = cleanup_start_vram - cleanup_end_vram
        
        print(f"   ✅ ULTRA CLEANUP: {vram_freed}MB freed in {cleanup_time:.3f}s ({cleanup_start_vram}MB → {cleanup_end_vram}MB)")

def create_fresh_context():
    """Create a completely fresh OpenGL context with VRAM tracking"""
    start_vram = get_gpu_memory_info()[2]  # used_vram
    
    try:
        # Create new context
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
        
        end_vram = get_gpu_memory_info()[2]  # used_vram
        vram_used = end_vram - start_vram
        
        return ctx, prog, vao, vbo, vram_used, start_vram, end_vram
        
    except Exception as e:
        print(f"   ❌ Error creating fresh context: {e}")
        raise

def nuclear_context_cleanup(ctx, vao, vbo, prog):
    """NUCLEAR cleanup - destroy everything synchronously with VRAM tracking"""
    start_vram = get_gpu_memory_info()[2]  # used_vram
    
    try:
        # Clean up resources in reverse order
        if vao:
            vao.release()
        if prog:
            prog.release()
        if vbo:
            vbo.release()
        
        # Force context destruction
        if ctx:
            ctx.finish()  # Wait for all GPU operations to complete
            ctx.gc()      # Force ModernGL garbage collection
            ctx.release() # Destroy context
        
        # Force system cleanup using constants
        if DOUBLE_GC_CLEANUP:
            gc.collect()
            gc.collect()  # Double cleanup
        
        # Wait for driver to process destruction
        time.sleep(CLEANUP_SLEEP_TIME)
        
        end_vram = get_gpu_memory_info()[2]  # used_vram
        vram_freed = start_vram - end_vram
        
        return vram_freed, start_vram, end_vram
        
    except Exception as e:
        print(f"   ⚠️  Error during cleanup: {e}")
        return 0, start_vram, start_vram

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
        # Always clean up GPU resources immediately
        if tex is not None:
            tex.release()
            del tex
        if fbo is not None:
            fbo.release()
            del fbo
        
        # Force garbage collection every frame to prevent accumulation
        gc.collect()

def recreate_gl_context():
    """Recreate OpenGL context to force complete VRAM cleanup"""
    try:
        # Create new context
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
        
        return ctx, prog, vao, vbo
        
    except Exception as e:
        print(f"Error recreating GL context: {e}")
        raise

def force_vram_cleanup(ctx):
    """Force aggressive VRAM cleanup"""
    try:
        # Force ModernGL garbage collection
        ctx.gc()
        ctx.gc()  # Call twice to be sure
        
        # Force Python garbage collection
        gc.collect()
        gc.collect()
        
        # Small delay to allow GPU driver to process cleanup
        time.sleep(CLEANUP_SLEEP_TIME)
        
    except Exception as e:
        print(f"Warning: Error during VRAM cleanup: {e}")

def process_frames_batch(ctx, prog, vao, frames, start_time, fps):
    """Process ALL frames in a segment as a true batch - load all textures, process all, then cleanup all"""
    
    print(f"   🎯 BATCH MODE: Loading {len(frames)} frames to GPU...")
    
    # === PHASE 1: LOAD ALL TEXTURES TO GPU ===
    textures = []
    framebuffers = []
    
    start_vram = get_gpu_memory_info()[2]
    
    try:
        height, width = frames[0].shape[:2]
        
        # Load ALL frame textures to GPU at once
        for frame_idx, frame in enumerate(frames):
            if frame_idx % PROGRESS_UPDATE_INTERVAL == 0:
                current_vram = get_gpu_memory_info()[2]
                print(f"     Loading texture {frame_idx + 1}/{len(frames)} - VRAM: {current_vram}MB")
                
                # Kill switch during loading
                if check_vram_kill_switch():
                    raise RuntimeError(f"VRAM kill switch during texture loading at frame {frame_idx}")
            
            tex = ctx.texture((width, height), 3, frame.tobytes())
            tex.build_mipmaps()
            textures.append(tex)
            
            fbo = ctx.simple_framebuffer((width, height))
            framebuffers.append(fbo)
        
        load_end_vram = get_gpu_memory_info()[2]
        vram_loaded = load_end_vram - start_vram
        print(f"   ✅ All textures loaded: {vram_loaded}MB VRAM used ({start_vram}MB → {load_end_vram}MB)")
        
        # === PHASE 2: PROCESS ALL FRAMES ===
        print(f"   🎬 BATCH MODE: Processing {len(frames)} frames...")
        
        processed_frames = []
        
        for frame_idx, (tex, fbo) in enumerate(zip(textures, framebuffers)):
            if frame_idx % PROGRESS_UPDATE_INTERVAL == 0:
                current_vram = get_gpu_memory_info()[2]
                print(f"     Processing frame {frame_idx + 1}/{len(frames)} - VRAM: {current_vram}MB")
            
            # Process this frame
            global_time = start_time + (frame_idx / fps)
            
            fbo.use()
            ctx.clear(0.0, 0.0, 0.0, 0.0)
            tex.use()
            
            prog['tex'].value = 0
            prog['time'].value = global_time
            
            vao.render()
            
            # Read result
            data = fbo.read(components=3, alignment=1)
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            img = np.flip(img, axis=0)
            
            processed_frames.append(img)
        
        process_end_vram = get_gpu_memory_info()[2]
        print(f"   ✅ All frames processed: VRAM: {process_end_vram}MB")
        
        return processed_frames
        
    finally:
        # === PHASE 3: NUCLEAR CLEANUP OF ALL RESOURCES ===
        print(f"   💥 BATCH MODE: Nuclear cleanup of {len(textures)} textures + {len(framebuffers)} framebuffers...")
        
        cleanup_start_vram = get_gpu_memory_info()[2]
        
        # Release all textures
        for tex in textures:
            tex.release()
        textures.clear()
        
        # Release all framebuffers  
        for fbo in framebuffers:
            fbo.release()
        framebuffers.clear()
        
        # Force aggressive cleanup
        del textures, framebuffers
        ctx.finish()  # Wait for all GPU operations
        ctx.gc()
        ctx.gc()
        gc.collect()
        gc.collect()
        
        # Wait for driver cleanup
        time.sleep(CLEANUP_SLEEP_TIME)
        
        cleanup_end_vram = get_gpu_memory_info()[2]
        vram_freed = cleanup_start_vram - cleanup_end_vram
        
        print(f"   ✅ Batch cleanup completed: {vram_freed}MB freed ({cleanup_start_vram}MB → {cleanup_end_vram}MB)")

def process_video_segments(input_file, output_file):
    """Process video in segments with BATCH processing per segment"""
    try:
        # Get video info
        width, height, fps, duration, total_frames = get_video_info(input_file)
        print(f"\n🎥 Video Info: {width}x{height}, {fps}fps, {duration:.2f}s, {total_frames} frames")
        
        # Calculate MAXIMUM segment size for batch processing
        max_frames_per_segment, vram_per_frame = calculate_max_segment_size(width, height)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        segment_files = []
        
        total_frames_processed = 0
        start_time = time.time()
        
        try:
            current_time = 0
            segment_index = 0
            
            while total_frames_processed < total_frames:
                # === CREATE FRESH CONTEXT FOR EACH SEGMENT ===
                print(f"\n🎬 Segment {segment_index + 1} - FRESH CONTEXT + BATCH PROCESSING")
                
                ctx_start = time.time()
                ctx, prog, vao, vbo, ctx_vram_used, ctx_start_vram, ctx_end_vram = create_fresh_context()
                ctx_time = time.time() - ctx_start
                
                print(f"   🔄 Fresh context: {ctx_time*1000:.1f}ms, VRAM: {ctx_start_vram}MB → {ctx_end_vram}MB")
                
                try:
                    # Calculate segment size
                    remaining_frames = total_frames - total_frames_processed
                    current_segment_frames = min(max_frames_per_segment, remaining_frames)
                    current_segment_duration = current_segment_frames / fps
                    
                    print(f"   📊 Progress: {total_frames_processed/total_frames*100:.1f}% ({total_frames_processed}/{total_frames})")
                    print(f"   🎯 Batch size: {current_segment_frames} frames")
                    
                    # === DECODE ALL FRAMES FOR THIS SEGMENT ===
                    decode_start = time.time()
                    frames = decode_video_segment(input_file, current_time, current_segment_duration)
                    decode_time = time.time() - decode_start
                    
                    if not frames:
                        break
                    
                    actual_frames = len(frames)
                    print(f"   📥 Decoded {actual_frames} frames in {decode_time:.1f}s")
                    
                    # Trim if needed
                    if total_frames_processed + actual_frames > total_frames:
                        frames = frames[:total_frames - total_frames_processed]
                        actual_frames = len(frames)
                    
                    # === BATCH PROCESS ALL FRAMES ===
                    batch_start = time.time()
                    processed_frames = process_frames_batch(ctx, prog, vao, frames, current_time, fps)
                    batch_time = time.time() - batch_start
                    
                    print(f"   ⚡ Batch processed {len(processed_frames)} frames in {batch_time:.1f}s")
                    
                    # === ENCODE BATCH TO VIDEO ===
                    segment_output = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
                    segment_files.append(segment_output)
                    
                    encode_start = time.time()
                    process_out = (
                        ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
                        .output(segment_output, pix_fmt='yuv420p', vcodec='libx264', crf=18, preset='fast')
                        .overwrite_output()
                        .run_async(pipe_stdin=True, pipe_stderr=subprocess.DEVNULL)
                    )
                    
                    for processed_frame in processed_frames:
                        process_out.stdin.write(processed_frame.tobytes())
                    
                    process_out.stdin.close()
                    process_out.wait()
                    encode_time = time.time() - encode_start
                    
                    print(f"   💾 Encoded segment in {encode_time:.1f}s")
                    
                    # Update counters
                    total_frames_processed += actual_frames
                    current_time += current_segment_duration
                    segment_index += 1
                    
                    # Cleanup segment data
                    del frames, processed_frames
                    gc.collect()
                    
                finally:
                    # === NUCLEAR CONTEXT CLEANUP ===
                    cleanup_start = time.time()
                    vram_freed, cleanup_start_vram, cleanup_end_vram = nuclear_context_cleanup(ctx, vao, vbo, prog)
                    cleanup_time = time.time() - cleanup_start
                    
                    print(f"   💥 Context destroyed: {cleanup_time*1000:.1f}ms, VRAM: {cleanup_start_vram}MB → {cleanup_end_vram}MB")
            
            # Concatenate segments
            print("\n🔗 Concatenating segments...")
            concat_segments(segment_files, output_file)
            
        finally:
            # Cleanup temp files
            for segment_file in segment_files:
                try:
                    if os.path.exists(segment_file):
                        os.remove(segment_file)
                except:
                    pass
            try:
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass
    
    except Exception as e:
        print(f"❌ Error: {e}")
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
            .run(quiet=True)  # Use quiet=True instead of pipe_stderr
        )
        
    finally:
        os.unlink(concat_file.name)

# Replace the arbitrary constants section with empirical measurement

# === EMPIRICAL VRAM MEASUREMENT ===
def measure_baseline_vram():
    """Measure ACTUAL baseline VRAM usage before any OpenGL operations"""
    free_vram_before, total_vram, used_vram_before = get_gpu_memory_info()
    return used_vram_before, free_vram_before, total_vram

def measure_context_creation_overhead():
    """Measure ACTUAL VRAM overhead of creating OpenGL context"""
    print("\n🔬 MEASURING: OpenGL context creation overhead...")
    
    # Measure before context creation
    used_before, free_before, total_vram = measure_baseline_vram()
    print(f"   Before context: {used_before}MB used, {free_before}MB free")
    
    # Create minimal context
    ctx = moderngl.create_context(standalone=True, backend='egl')
    
    # Measure after context creation
    used_after, free_after, _ = get_gpu_memory_info()
    actual_context_overhead = used_after - used_before
    
    print(f"   After context: {used_after}MB used, {free_after}MB free")
    print(f"   ACTUAL context overhead: {actual_context_overhead}MB")
    
    # Clean up test context
    ctx.release()
    gc.collect()
    time.sleep(0.1)
    
    # Measure after cleanup
    used_cleanup, free_cleanup, _ = get_gpu_memory_info()
    cleanup_effectiveness = used_after - used_cleanup
    
    print(f"   After cleanup: {used_cleanup}MB used, {free_cleanup}MB free")
    print(f"   Cleanup freed: {cleanup_effectiveness}MB")
    
    return actual_context_overhead, used_before

def measure_single_frame_vram_usage(width, height):
    """Measure ACTUAL VRAM usage for processing one frame"""
    print(f"\n🔬 MEASURING: Single frame VRAM usage for {width}x{height}...")
    
    # Create test context
    ctx = moderngl.create_context(standalone=True, backend='egl')
    
    try:
        # Measure before any textures
        used_before, _, _ = get_gpu_memory_info()
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Create texture
        tex = ctx.texture((width, height), 3, test_frame.tobytes())
        used_after_texture, _, _ = get_gpu_memory_info()
        texture_vram = used_after_texture - used_before
        
        # Add mipmaps
        tex.build_mipmaps()
        used_after_mipmaps, _, _ = get_gpu_memory_info()
        mipmap_vram = used_after_mipmaps - used_after_texture
        
        # Create framebuffer
        fbo = ctx.simple_framebuffer((width, height))
        used_after_fbo, _, _ = get_gpu_memory_info()
        framebuffer_vram = used_after_fbo - used_after_mipmaps
        
        total_frame_vram = used_after_fbo - used_before
        
        print(f"   Texture only: {texture_vram}MB")
        print(f"   Mipmaps add: {mipmap_vram}MB")
        print(f"   Framebuffer: {framebuffer_vram}MB")
        print(f"   TOTAL per frame: {total_frame_vram}MB")
        
        # Cleanup
        tex.release()
        fbo.release()
        
        return {
            'total_mb': total_frame_vram,
            'texture_mb': texture_vram,
            'mipmap_mb': mipmap_vram,
            'framebuffer_mb': framebuffer_vram
        }
        
    finally:
        ctx.release()
        gc.collect()

def empirical_vram_calibration(width, height):
    """Perform complete empirical VRAM measurement and calibration"""
    print(f"\n🔬 EMPIRICAL VRAM CALIBRATION for {width}x{height}")
    print("=" * 60)
    
    # 1. Measure baseline system VRAM usage
    baseline_used, baseline_free, total_vram = measure_baseline_vram()
    print(f"\n📊 System baseline: {baseline_used}MB used, {baseline_free}MB free")
    
    # 2. Measure actual context overhead
    context_overhead, _ = measure_context_creation_overhead()
    
    # 3. Measure per-frame VRAM usage
    frame_vram = measure_single_frame_vram_usage(width, height)
    
    # 4. Calculate AGGRESSIVE available VRAM (you have 4GB!)
    # Only leave 5% headroom since your background usage is tiny
    aggressive_total = total_vram * 0.95  # Use 95% of total VRAM
    available_for_processing = aggressive_total - baseline_used - context_overhead
    
    print(f"   🔥 AGGRESSIVE MODE: Using {aggressive_total}MB of {total_vram}MB VRAM")
    
    # 5. Calculate maximum frames based on ACTUAL measurements
    max_frames_from_vram = int(available_for_processing / frame_vram['total_mb'])
    
    # 6. Apply minimal safety factor (90% instead of 80%)
    safe_max_frames = int(max_frames_from_vram * 0.90)  # 90% of calculated max
    
    print(f"\n✅ AGGRESSIVE EMPIRICAL RESULTS:")
    print(f"   Total VRAM: {total_vram}MB")
    print(f"   Aggressive limit (95%): {aggressive_total:.1f}MB")
    print(f"   System baseline: {baseline_used}MB")
    print(f"   Context overhead: {context_overhead}MB")
    print(f"   Available for frames: {available_for_processing:.1f}MB")
    print(f"   VRAM per frame: {frame_vram['total_mb']:.2f}MB")
    print(f"   Max frames (calculated): {max_frames_from_vram}")
    print(f"   Aggressive max frames (90%): {safe_max_frames}")
    
    return {
        'total_vram': total_vram,
        'baseline_used': baseline_used,
        'context_overhead': context_overhead,
        'frame_vram': frame_vram,
        'available_vram': available_for_processing,
        'max_frames': max_frames_from_vram,
        'safe_max_frames': safe_max_frames
    }

def calculate_empirical_segment_size(width, height):
    """Calculate segment size based on REAL measurements - AGGRESSIVE MODE"""
    calibration = empirical_vram_calibration(width, height)
    
    # Use aggressive max frames with higher limits
    optimal_frames = max(100, min(calibration['safe_max_frames'], 2000))  # Up to 2000 frames
    
    print(f"\n🔥 AGGRESSIVE EMPIRICAL SEGMENT SIZE: {optimal_frames} frames")
    print(f"   Maximizing your 4GB VRAM instead of being overly conservative!")
    
    return optimal_frames, calibration['frame_vram']['total_mb']
def process_video_segments(input_file, output_file):
    """Process video with ULTRA PARALLEL batch processing"""
    try:
        width, height, fps, duration, total_frames = get_video_info(input_file)
        print(f"\n🎥 Video Info: {width}x{height}, {fps}fps, {duration:.2f}s, {total_frames} frames")
        
        # Calculate optimal segment size
        max_frames_per_segment = adaptive_segment_sizing(width, height)
        
        temp_dir = tempfile.mkdtemp()
        segment_files = []
        
        total_frames_processed = 0
        start_time = time.time()
        
        try:
            current_time = 0
            segment_index = 0
            
            while total_frames_processed < total_frames:
                print(f"\n🎬 Segment {segment_index + 1} - ULTRA PARALLEL BATCH")
                
                # Create fresh context
                ctx_start = time.time()
                ctx, prog, vao, vbo, ctx_vram_used, ctx_start_vram, ctx_end_vram = create_fresh_context()
                ctx_time = time.time() - ctx_start
                
                print(f"   🔄 Fresh context: {ctx_time*1000:.1f}ms, VRAM: {ctx_start_vram}MB → {ctx_end_vram}MB")
                
                try:
                    # Calculate segment
                    remaining_frames = total_frames - total_frames_processed
                    current_segment_frames = min(max_frames_per_segment, remaining_frames)
                    current_segment_duration = current_segment_frames / fps
                    
                    progress = total_frames_processed / total_frames * 100
                    print(f"   📊 Progress: {progress:.1f}% ({total_frames_processed}/{total_frames})")
                    print(f"   🎯 Ultra parallel batch: {current_segment_frames} frames")
                    
                    # Decode frames
                    decode_start = time.time()
                    frames = decode_video_segment(input_file, current_time, current_segment_duration)
                    decode_time = time.time() - decode_start
                    
                    if not frames:
                        break
                    
                    actual_decoded = len(frames)
                    print(f"   📥 Decoded {actual_decoded} frames in {decode_time:.1f}s")
                    
                    # Trim if needed
                    if total_frames_processed + actual_decoded > total_frames:
                        frames = frames[:total_frames - total_frames_processed]
                    
                    # === ULTRA PARALLEL PROCESSING ===
                    batch_start = time.time()
                    processed_frames = process_frames_batch_ultra_parallel(ctx, prog, vao, frames, current_time, fps)
                    # Alternative: processed_frames = process_frames_batch_mega_parallel(ctx, prog, vao, frames, current_time, fps)
                    batch_time = time.time() - batch_start
                    
                    actual_processed = len(processed_frames)
                    processing_fps = actual_processed / batch_time if batch_time > 0 else 0
                    
                    print(f"   🚀 ULTRA PARALLEL: {actual_processed} frames in {batch_time:.1f}s ({processing_fps:.1f} fps)")
                    
                    if actual_processed == 0:
                        print("   ⚠️  No frames processed, stopping")
                        break
                    
                    # Encode segment
                    segment_output = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
                    segment_files.append(segment_output)
                    
                    encode_start = time.time()
                    process_out = (
                        ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
                        .output(segment_output, pix_fmt='yuv420p', vcodec='libx264', crf=18, preset='fast')
                        .overwrite_output()
                        .run_async(pipe_stdin=True, pipe_stderr=subprocess.DEVNULL)
                    )
                    
                    for processed_frame in processed_frames:
                        process_out.stdin.write(processed_frame.tobytes())
                    
                    process_out.stdin.close()
                    process_out.wait()
                    encode_time = time.time() - encode_start
                    
                    print(f"   💾 Encoded in {encode_time:.1f}s")
                    
                    # Update counters
                    total_frames_processed += actual_processed
                    current_time += (actual_processed / fps)
                    segment_index += 1
                    
                    # Cleanup
                    del frames, processed_frames
                    gc.collect()
                    
                finally:
                    # Cleanup context
                    cleanup_start = time.time()
                    vram_freed, cleanup_start_vram, cleanup_end_vram = nuclear_context_cleanup(ctx, vao, vbo, prog)
                    cleanup_time = time.time() - cleanup_start
                    
                    print(f"   💥 Context destroyed: {cleanup_time*1000:.1f}ms")
            
            # Concatenate segments
            print("\n🔗 Concatenating segments...")
            concat_segments(segment_files, output_file)
            
        finally:
            # Cleanup temp files
            for segment_file in segment_files:
                try:
                    if os.path.exists(segment_file):
                        os.remove(segment_file)
                except:
                    pass
            try:
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python shader_video.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"🚀 ULTRA PARALLEL VIDEO PROCESSING")
    print(f"   VRAM Configuration:")
    print(f"   - Total VRAM: {TOTAL_VRAM_MB}MB")
    print(f"   - Kill switch: {KILL_SWITCH_THRESHOLD*100:.0f}%")
    print(f"   - Early abort: {EARLY_ABORT_THRESHOLD*100:.0f}%")
    print(f"   - Target usage: {TARGET_VRAM_USAGE*100:.0f}%")
    print(f"   - Safety factor: {SAFETY_FACTOR*100:.0f}%")
    print(f"   - Segment limits: {MIN_FRAMES_PER_SEGMENT}-{MAX_FRAMES_PER_SEGMENT} frames")
    
    start_time = time.time()
    print(f"\nProcessing video: {input_file}")
    process_video_segments(input_file, output_file)
    print(f"Output video saved as: {output_file}")
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")