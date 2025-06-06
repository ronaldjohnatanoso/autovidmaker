import bpy
import os
import sys
import time
from tqdm import tqdm

starttime = time.time()

# --- CONFIGURATION ---
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "../0-project-files/demotivational/images_1080p/")  # Change to your PNG folder
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output.mp4")  # Output video file
FPS = 24  # Frames per second
RES_X = 1920  # Resolution width
RES_Y = 1080  # Resolution height
IMAGE_DURATION = 2 * FPS  # 2 seconds per image

# --- CLEANUP ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- SETUP SCENE ---
scene = bpy.context.scene
scene.sequence_editor_create()

# Use Workbench engine for lightweight rendering
scene.render.engine = 'BLENDER_WORKBENCH'

# Set render settings
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.image_settings.color_mode = 'RGB'  # Enable alpha channel

scene.render.ffmpeg.format = 'QUICKTIME'  # Quicktime container supports alpha
scene.render.ffmpeg.codec = 'QTRLE'       # Quicktime RLE codec supports alpha
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.fps = FPS
scene.render.filepath = OUTPUT_PATH

# --- LOAD IMAGES INTO VSE ---
images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.png')])
if not images:
    print("No PNG images found in", IMAGE_DIR)
    sys.exit(1)

frame_start = 1
for img in images:
    strip = scene.sequence_editor.sequences.new_image(
        name=img,
        filepath=os.path.join(IMAGE_DIR, img),
        channel=1,
        frame_start=frame_start
    )
    strip.frame_final_duration = IMAGE_DURATION
    frame_start += IMAGE_DURATION

scene.frame_start = 1
scene.frame_end = frame_start - 1

# --- SETUP COMPOSITOR NODES FOR SCANLINE EFFECT ---
scene.use_nodes = True
nodes = scene.node_tree.nodes
links = scene.node_tree.links
nodes.clear()

# Input Render Layers node
render_layers = nodes.new(type='CompositorNodeRLayers')
render_layers.location = (0, 0)

# Create a procedural scanline texture using Wave Texture node
wave_tex = nodes.new(type='ShaderNodeTexWave')
wave_tex.location = (-300, 200)
wave_tex.wave_type = 'BANDS'
wave_tex.bands_direction = 'HORIZONTAL'
wave_tex.inputs['Scale'].default_value = 50  # Adjust density of scanlines

# Convert Wave Texture to a format usable in compositor
texture_node = nodes.new(type='CompositorNodeTexture')
texture_node.location = (-100, 200)
texture_node.texture = bpy.data.textures.new("ScanlineTex", type='WAVE')
texture_node.texture.wave_type = 'BANDS'
texture_node.texture.bands_direction = 'HORIZONTAL'
texture_node.texture.scale = 50

# Mix the scanline texture with original footage
mix_node = nodes.new(type='CompositorNodeMixRGB')
mix_node.blend_type = 'MULTIPLY'
mix_node.inputs['Fac'].default_value = 0.3  # Scanline intensity
mix_node.location = (200, 0)

# Composite output node
composite_out = nodes.new(type='CompositorNodeComposite')
composite_out.location = (400, 0)

# Viewer node (optional, for preview)
viewer_node = nodes.new(type='CompositorNodeViewer')
viewer_node.location = (400, -200)

# Link nodes
links.new(render_layers.outputs['Image'], mix_node.inputs[1])
links.new(texture_node.outputs['Color'], mix_node.inputs[2])
links.new(mix_node.outputs['Image'], composite_out.inputs['Image'])
links.new(mix_node.outputs['Image'], viewer_node.inputs['Image'])

# --- RENDER WITH PROGRESS BAR ---
print("Starting render...")

import logging
logging.getLogger("blender").setLevel(logging.ERROR)

import io
from contextlib import redirect_stdout

total_frames = scene.frame_end - scene.frame_start + 1
progress_bar = tqdm(total=total_frames, desc="Rendering", unit="frames")

def frame_change_handler(scene):
    progress_bar.update(1)

bpy.app.handlers.frame_change_pre.append(frame_change_handler)

with redirect_stdout(io.StringIO()):
    bpy.ops.render.render(animation=True)

progress_bar.close()
print("Video rendered to", OUTPUT_PATH)

endtime = time.time()
print("Total time taken:", endtime - starttime, "seconds")
