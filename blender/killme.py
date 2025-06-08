import subprocess
import numpy as np
import moderngl
import time
import os

def log_vram_usage():
    """Logs the current VRAM usage and returns the value in MB."""
    try:
        vram_usage = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8').strip().split('\n')
        vram_usage_mb = int(vram_usage[0])  # Assuming single GPU
        print(f"Current VRAM usage: {vram_usage_mb} MB")
        return vram_usage_mb
    except Exception as e:
        print(f"Error logging VRAM usage: {e}")
        return -1  # Return -1 if unable to fetch VRAM usage

def process_segment(segment_index, start_time, duration, input_file, temp_dir):
    WIDTH, HEIGHT = 1920, 1080
    FPS = 30
    FRAME_SIZE = WIDTH * HEIGHT * 4  # RGBA

    # Log VRAM usage before processing
    print(f"Logging VRAM usage before processing segment {segment_index}...")
    vram_before = log_vram_usage()
    if vram_before > 4000:  # Kill switch if VRAM exceeds 4 GB
        print("VRAM usage exceeded 4 GB. Terminating process.")
        exit(1)

    # === Step 1: Extract segment with FFmpeg ===
    print(f"Extracting segment {segment_index}...")
    raw_data = subprocess.check_output([
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', input_file,
        '-vf', f'scale={WIDTH}:{HEIGHT}',
        '-pix_fmt', 'rgba',
        '-f', 'rawvideo',
        '-loglevel', 'quiet',
        '-'
    ])

    frames = np.frombuffer(raw_data, dtype=np.uint8)
    actual_size = frames.size
    N_FRAMES = actual_size // FRAME_SIZE  # Dynamically calculate the number of frames

    # Handle size mismatch
    if actual_size % FRAME_SIZE != 0:
        print(f"Warning: Size mismatch for segment {segment_index}. Actual size {actual_size} is not a multiple of frame size {FRAME_SIZE}.")
        N_FRAMES = actual_size // FRAME_SIZE  # Recalculate N_FRAMES based on actual size

    frames = frames.reshape((N_FRAMES, HEIGHT, WIDTH, 4))

    # === Step 2: Set up moderngl compute context ===
    print("Setting up moderngl compute context...")
    ctx = moderngl.create_standalone_context(backend='egl')

    # === Step 3: Upload frames to 2D texture array ===
    print("Uploading frames to 2D texture array...")
    tex_array = ctx.texture_array((WIDTH, HEIGHT, N_FRAMES), 4, dtype='f1')
    tex_array.write(frames.tobytes())
    tex_array.build_mipmaps()
    tex_array.use(location=0)

    # === Step 4: Create output image for writing processed frames ===
    output_tex = ctx.texture_array((WIDTH, HEIGHT, N_FRAMES), 4, dtype='f1')
    output_tex.bind_to_image(1, read=False, write=True)

    # === Step 5: Time uniform ===
    time_uniform = np.linspace(0, duration, N_FRAMES).astype('f4')

    # === Step 6: Compute Shader ===
    print("Running compute shader...")
    compute_shader = ctx.compute_shader("""
    #version 430
    layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

    layout(binding = 0) uniform sampler2DArray input_frames;
    layout(rgba8, binding = 1) writeonly uniform image2DArray output_frames;
    uniform float time_array[%(N_FRAMES)s];

    void main() {
        ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
        int layer = int(gl_GlobalInvocationID.z);

        if (uv.x >= %(WIDTH)s || uv.y >= %(HEIGHT)s || layer >= %(N_FRAMES)s) return;

        vec4 color = texelFetch(input_frames, ivec3(uv, layer), 0);
        float t = time_array[layer];
        float pulse = 0.5 + 0.5 * sin(6.2831 * t);  // 1Hz
        color.r += pulse * 0.5;
        color = clamp(color, 0.0, 1.0);

        imageStore(output_frames, ivec3(uv, layer), color);
    }
    """ % {'WIDTH': WIDTH, 'HEIGHT': HEIGHT, 'N_FRAMES': N_FRAMES})

    compute_shader['time_array'].write(time_uniform.tobytes())
    compute_shader.run(group_x=(WIDTH // 32), group_y=(HEIGHT // 32), group_z=N_FRAMES)
    ctx.finish()

    # === Step 7: Read all processed frames from GPU ===
    print("Reading frames back from GPU...")
    output_data = output_tex.read()
    output_frames = np.frombuffer(output_data, dtype=np.uint8).reshape((N_FRAMES, HEIGHT, WIDTH, 4))

    # === Step 8: Encode segment with FFmpeg ===
    temp_file = os.path.join(temp_dir, f"temp_segment_{segment_index}.mp4")
    print(f"Encoding segment {segment_index} to {temp_file}...")
    encode = subprocess.Popen([
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgba',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        temp_file
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for frame in output_frames:
        encode.stdin.write(frame.tobytes())

    encode.stdin.close()
    encode.wait()
    print(f"Segment {segment_index} processing completed.")

    # Explicitly delete GPU resources
    print("Releasing GPU resources...")
    tex_array.release()
    output_tex.release()

    # Destroy moderngl context to free VRAM
    print("Destroying moderngl context to free VRAM...")
    ctx.release()

    # Log GPU memory usage
    print("Logging GPU memory usage...")
    subprocess.run(['nvidia-smi'])

    # Wait until VRAM usage drops below 100 MB with a timeout
    print("Waiting for VRAM usage to drop below 100 MB...")
    timeout = 60  # Timeout in seconds
    start_time = time.time()
    while True:
        vram_usage = log_vram_usage()
        if vram_usage < 100:
            print("VRAM usage is below 100 MB. Proceeding to next segment...")
            break
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for VRAM to drop below 100 MB. Proceeding anyway.")
            break
        time.sleep(1)  # Check VRAM usage every second

    # Log VRAM usage after processing
    print(f"Logging VRAM usage after processing segment {segment_index}...")
    vram_after = log_vram_usage()
    if vram_after > 4000:  # Kill switch if VRAM exceeds 4 GB
        print("VRAM usage exceeded 4 GB. Terminating process.")
        exit(1)

def concat_segments(temp_dir, output_file):
    print("Concatenating all segments...")
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        for temp_file in sorted(os.listdir(temp_dir)):
            if temp_file.endswith(".mp4"):
                f.write(f"file '{os.path.join(temp_dir, temp_file)}'\n")

    subprocess.run([
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        output_file
    ])
    print(f"✅ Final output saved to {output_file}")

if __name__ == "__main__":
    input_file = "input.mp4"
    output_file = "output.mp4"
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)

    # Divide video into 10-second segments
    duration = 10
    total_duration = float(subprocess.check_output([
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_file
    ]).strip())
    num_segments = int(total_duration // duration)  # Use floor division to get the number of segments

    for segment_index in range(num_segments):
        start_time = segment_index * duration
        process_segment(segment_index, start_time, duration, input_file, temp_dir)

    # Process the remaining part of the video, if any
    if total_duration % duration > 0:
        process_segment(num_segments, num_segments * duration, total_duration % duration, input_file, temp_dir)

    # Concatenate all segments
    concat_segments(temp_dir, output_file)