import ffmpeg
import numpy as np
from PIL import Image
import moderngl
import os
from tqdm import tqdm
import subprocess
import tempfile

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec4 color = texture(tex, v_uv);

    // Simple scanline effect
    float scanline = sin(v_uv.y * 800.0) * 0.1;
    color.rgb -= scanline;

    f_color = color;
}
"""

def decode_video_frames(input_file):
    """Decode video frames as raw RGB numpy arrays"""
    probe = ffmpeg.probe(input_file)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    process = (
        ffmpeg
        .input(input_file)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame, width, height

    process.wait()

def encode_video_frames(output_file, width, height, fps):
    """Create an ffmpeg process to pipe frames to for encoding"""
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
        .output(output_file, pix_fmt='yuv420p', vcodec='libx264', crf=18, preset='fast')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

def run_shader_on_frame(ctx, prog, frame):
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

    # Render fullscreen quad
    prog['tex'].value = 0
    vao.render()

    # Read pixels from framebuffer
    data = fbo.read(components=3, alignment=1)
    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    img = np.flip(img, axis=0)  # Flip vertically (OpenGL vs image coords)
    return img

if __name__ == '__main__':
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Probe input for fps
    probe = ffmpeg.probe(input_file)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = eval(video_stream['r_frame_rate'])  # e.g., "30/1" -> 30.0

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

    print(f"Processing video: {input_file}")

    # Open output pipe
    first_frame = True
    process_out = None

    for i, (frame, width, height) in enumerate(decode_video_frames(input_file)):
        if first_frame:
            process_out = encode_video_frames(output_file, width, height, fps)
            first_frame = False

        out_frame = run_shader_on_frame(ctx, prog, frame)
        process_out.stdin.write(out_frame.tobytes())

        if i % 10 == 0:
            print(f"Processed frame {i}")

    process_out.stdin.close()
    process_out.wait()

    print(f"Output video saved as: {output_file}")
