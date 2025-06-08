import moderngl
import numpy as np
import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk

# Resolution constant
RESOLUTION = (1920, 1080)  # Width, Height

# Vertex shader
vertex_shader = '''
#version 330 core
in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
'''

# Fragment shader with pulsing red effect
fragment_shader = '''
#version 330 core
uniform sampler2D u_texture;
uniform float u_time;
in vec2 v_texcoord;
out vec4 fragColor;

void main() {
    vec4 original = texture(u_texture, v_texcoord);
    float pulse = sin(u_time * 3.14159) * 0.5 + 0.5;
    vec3 red_overlay = vec3(pulse, 0.0, 0.0);
    vec3 result = mix(original.rgb, original.rgb + red_overlay, 0.3);
    fragColor = vec4(result, original.a);
}
'''

def process_video_with_preview(input_path, output_path):
    # Create EGL context
    ctx = moderngl.create_context(standalone=True, backend='egl')
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = RESOLUTION
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, RESOLUTION)
    
    # Create shader program
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    
    # Create texture
    texture = ctx.texture(RESOLUTION, 3)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    
    # Create framebuffer
    fbo = ctx.framebuffer(color_attachments=[ctx.texture(RESOLUTION, 3)])
    
    # Vertex data for full-screen quad
    vertices = np.array([
        # Position  # TexCoord
        -1.0, -1.0,  0.0, 1.0,
         1.0, -1.0,  1.0, 1.0,
         1.0,  1.0,  1.0, 0.0,
        -1.0, -1.0,  0.0, 1.0,
         1.0,  1.0,  1.0, 0.0,
        -1.0,  1.0,  0.0, 0.0,
    ], dtype=np.float32)
    
    # Create vertex buffer and vertex array
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(program, [(vbo, '2f 2f', 'in_position', 'in_texcoord')])
    
    # Create GUI window
    root = tk.Tk()
    root.title("Video Preview")
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()
    
    frame_count = 0
    
    def update_preview():
        nonlocal frame_count
        ret, frame = cap.read()
        if not ret:
            root.quit()
            return
            
        # Resize frame to match RESOLUTION
        frame = cv2.resize(frame, RESOLUTION)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate time for pulsing effect
        current_time = frame_count / fps
        
        # Upload frame to texture
        texture.write(frame_rgb.tobytes())
        
        # Render with shader
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        program['u_texture'].value = 0
        program['u_time'].value = current_time
        texture.use(0)
        vao.render()
        
        # Read processed frame
        processed_data = fbo.read(components=3)
        processed_frame = np.frombuffer(processed_data, dtype=np.uint8).reshape((height, width, 3))
        
        # Flip vertically (OpenGL to image coordinates)
        processed_frame = np.flipud(processed_frame)
        
        # Convert back to BGR for OpenCV
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Write to output video
        out.write(processed_frame_bgr)
        
        # Display preview in GUI
        preview_image = Image.fromarray(processed_frame)
        preview_photo = ImageTk.PhotoImage(image=preview_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=preview_photo)
        canvas.image = preview_photo
        
        # Print progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
        
        frame_count += 1
        root.after(int(1000 / fps), update_preview)
    
    update_preview()
    root.mainloop()
    
    # Cleanup
    cap.release()
    out.release()
    ctx.release()
    
    print("Processing completed!")

if __name__ == "__main__":
    input_file = "input.mp4"
    output_file = "output.mp4"
    process_video_with_preview(input_file, output_file)