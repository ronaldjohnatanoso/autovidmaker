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

def get_gpu_memory_info():
    """Get available GPU memory in MB"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free // (1024 * 1024), info.total // (1024 * 1024)
    except:
        # Fallback estimation for RTX 3050 (4GB)
        return 3000, 4096  # Leave 1GB buffer

def estimate_frame_vram_usage(width, height):
    """Estimate VRAM usage per frame in MB"""
    # Texture memory: width * height * 3 bytes (RGB)
    texture_mem = width * height * 3
    
    # Framebuffer memory: width * height * 3 bytes (RGB output)
    framebuffer_mem = width * height * 3
    
    # Additional overhead for mipmaps, temporary buffers, etc. (multiply by 2.5)
    total_per_frame = (texture_mem + framebuffer_mem) * 2.5
    
    return total_per_frame / (1024 * 1024)  # Convert to MB

def calculate_segment_size(width, height, total_frames, fps):
    """Calculate optimal segment size based on VRAM constraints"""
    available_vram, total_vram = get_gpu_memory_info()
    
    # Test with a small batch first for accurate estimation
    test_frames = min(5, total_frames)
    estimated_per_frame = estimate_frame_vram_usage(width, height)
    
    print(f"Estimated VRAM per frame: {estimated_per_frame:.2f} MB")
    print(f"Available VRAM: {available_vram} MB")
    
    # Leave 20% buffer for safety and other GPU processes
    safe_vram = available_vram * 0.8
    
    # Calculate frames per segment
    frames_per_segment = int(safe_vram / estimated_per_frame)
    frames_per_segment = max(1, min(frames_per_segment, total_frames))
    
    # Convert to time segments
    segment_duration = frames_per_segment / fps
    
    print(f"Frames per segment: {frames_per_segment}")
    print(f"Segment duration: {segment_duration:.2f} seconds")
    
    return frames_per_segment, segment_duration

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
        .run_async(pipe_stdout=True)
    )
    
    probe = ffmpeg.probe(input_file)
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
    
    # Clean up GPU resources
    tex.release()
    fbo.release()
    
    return img

def process_video_segments(input_file, output_file):
    """Process video in segments and concatenate"""
    # Get video info
    width, height, fps, duration, total_frames = get_video_info(input_file)
    
    # Calculate optimal segment size
    frames_per_segment, segment_duration = calculate_segment_size(width, height, total_frames, fps)
    
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
    
    try:
        # Process segments
        current_time = 0
        segment_index = 0
        
        while current_time < duration:
            remaining_time = duration - current_time
            current_segment_duration = min(segment_duration, remaining_time)
            
            print(f"Processing segment {segment_index + 1}, time: {current_time:.2f}s - {current_time + current_segment_duration:.2f}s")
            
            # Decode segment frames
            frames = decode_video_segment(input_file, current_time, current_segment_duration)
            
            if not frames:
                break
            
            # Create output for this segment
            segment_output = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
            segment_files.append(segment_output)
            
            # Encode this segment
            process_out = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
                .output(segment_output, pix_fmt='yuv420p', vcodec='libx264', crf=18, preset='fast')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            
            # Process frames in this segment
            for frame_idx, frame in enumerate(frames):
                global_time = current_time + (frame_idx / fps)  # Maintain global time continuity
                out_frame = run_shader_on_frame(ctx, prog, vao, frame, global_time)
                process_out.stdin.write(out_frame.tobytes())
            
            process_out.stdin.close()
            process_out.wait()
            
            # Clean up memory
            del frames
            gc.collect()
            ctx.gc()  # ModernGL garbage collection
            
            current_time += current_segment_duration
            segment_index += 1
        
        # Concatenate all segments
        print("Concatenating segments...")
        concat_segments(segment_files, output_file)
        
    finally:
        # Clean up temporary files
        for segment_file in segment_files:
            if os.path.exists(segment_file):
                os.remove(segment_file)
        os.rmdir(temp_dir)
        
        # Clean up OpenGL resources
        vao.release()
        vbo.release()
        prog.release()
        ctx.release()

def concat_segments(segment_files, output_file):
    """Concatenate video segments using ffmpeg"""
    # Create concat file
    concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    
    try:
        for segment_file in segment_files:
            concat_file.write(f"file '{segment_file}'\n")
        concat_file.close()
        
        # Concatenate using ffmpeg
        (
            ffmpeg
            .input(concat_file.name, format='concat', safe=0)
            .output(output_file, c='copy')
            .overwrite_output()
            .run()
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
