import logging
import subprocess
import numpy as np
import moderngl
import psutil
import sys
import time
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import threading

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

# Define pulsing red shader
FRAGMENT_SHADER = '''
#version 330
uniform sampler2D tex;
uniform float u_time;
in vec2 uv;
out vec4 color;

void main() {
    vec4 c = texture(tex, uv);
    
    // Create a pulsing effect using sine wave (same as balls.py)
    float pulse = sin(u_time * 3.14159) * 0.5 + 0.5;  // Changed from 'time' to 'u_time'
    
    // Apply red overlay (additive, not multiplicative)
    vec3 red_overlay = vec3(pulse, 0.0, 0.0);
    vec3 result = mix(c.rgb, c.rgb + red_overlay, 0.3);  // Add red, don't multiply
    
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
        return 0  # Fallback if not using NVIDIA or not found

def setup_ffmpeg_input():
    logging.info("Setting up FFmpeg input process with GPU acceleration.")
    return subprocess.Popen(
        ['ffmpeg', '-hwaccel', 'cuda', '-c:v', 'h264_cuvid', '-i', 'input.mp4',
         '-vf', f'scale={WIDTH}:{HEIGHT}',
         '-f', 'rawvideo',
         '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL  # Suppress FFmpeg output
    )

def setup_ffmpeg_output():
    logging.info("Setting up FFmpeg output process with GPU encoding.")
    return subprocess.Popen(
        ['ffmpeg',
         '-y', '-f', 'rawvideo',
         '-vcodec', 'rawvideo',
         '-pix_fmt', 'rgb24',
         '-s', f'{WIDTH}x{HEIGHT}',
         '-r', str(FRAME_RATE),
         '-i', '-', '-an',
         '-vcodec', 'h264_nvenc',  # Use NVIDIA GPU for encoding
         '-pix_fmt', 'yuv420p',
         'output.mp4'],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL  # Suppress FFmpeg output
    )

def create_preview_window():
    """Create a PyQt5 GUI window for video preview."""
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Video Preview")
    layout = QVBoxLayout()
    label = QLabel()
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)
    window.setLayout(layout)
    window.show()
    return app, window, label

def main():
    logging.info("Initializing OpenGL context.")
    ctx = moderngl.create_standalone_context(backend='egl')

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    prog['u_time'] = 0.0  # Changed from 'time' to 'u_time'

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

    ffmpeg_in = setup_ffmpeg_input()
    ffmpeg_out = setup_ffmpeg_output()

    gpu_usage_data = []  # Store GPU usage data
    timestamps = []  # Store timestamps

    # Create GUI preview window
    app, window, label = create_preview_window()

    def process_frames():
        try:
            logging.info("Starting video processing loop.")
            pbar = tqdm(desc="Processing frames", unit="frame", dynamic_ncols=True)
            frame_count = 0  # Add frame counter

            while True:
                raw_frame = ffmpeg_in.stdout.read(WIDTH * HEIGHT * 3)
                if not raw_frame:
                    logging.info("No more frames to process. Exiting loop.")
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
                tex.write(frame.tobytes())
                fbo.use()
                tex.use(0)
                prog['tex'] = 0
                prog['u_time'] = frame_count / FRAME_RATE  # Changed from 'time' to 'u_time'
                vao.render(moderngl.TRIANGLE_STRIP)

                out_data = fbo.read(components=3, alignment=1)
                ffmpeg_out.stdin.write(out_data)

                # Update progress bar
                pbar.update(1)

                # Update GUI preview
                image = QImage(out_data, WIDTH, HEIGHT, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                label.setPixmap(pixmap)

                # Log GPU usage
                gpu_usage = get_gpu_memory_usage()
                gpu_usage_data.append(gpu_usage * 100)  # Convert to percentage
                timestamps.append(time.time())

                if gpu_usage >= VRAM_THRESHOLD:
                    logging.error("⚠️ VRAM threshold exceeded. Killing process.")
                    break

                frame_count += 1  # Increment frame counter

            pbar.close()

        finally:
            logging.info("Terminating FFmpeg processes.")
            ffmpeg_in.terminate()
            ffmpeg_out.stdin.close()
            ffmpeg_out.wait()
            logging.info("Video processing completed.")

            # Plot VRAM usage graph
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, gpu_usage_data, label="VRAM Usage (%)", color="blue")
            plt.axhline(y=VRAM_THRESHOLD * 100, color="red", linestyle="--", label="Threshold (95%)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("VRAM Usage (%)")
            plt.title("VRAM Usage Over Time")
            plt.legend()
            plt.grid()
            plt.savefig("vram_usage_report.png")  # Save the graph as an image
            # plt.show()

    # Run the frame processing in a separate thread
    threading.Thread(target=process_frames).start()

    # Start the PyQt5 event loop
    app.exec_()

if __name__ == '__main__':
    logging.info("Starting main program.")
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Program completed in {end_time - start_time:.2f} seconds.")