import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
# COMPLETELY DISABLE ImageMagick
import moviepy.config as config
config.IMAGEMAGICK_BINARY = None

import sys
import json
import argparse
import time
import math
import threading
import subprocess
import tempfile
import shutil
import traceback
from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip, AudioFileClip
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Constants
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 24
SEGMENT_DURATION = 15  # seconds per segment

# Subtitle styling constants (updated for brain rot)
SUBTITLE_FONTSIZE = 100  # Bigger text
SUBTITLE_COLOR = "#FFFFFF"  # Will be overridden by rainbow if enabled
SUBTITLE_FONT = 'Komika-Axis'  # Custom thick font
SUBTITLE_STROKE_COLOR = 'black'
SUBTITLE_STROKE_WIDTH = 8  # Thicker stroke
SUBTITLE_POS_Y_RATIO = 0.7  # Lower position for more impact
SUBTITLE_POSITION_Y = int(TARGET_HEIGHT * SUBTITLE_POS_Y_RATIO)

# Text effect constants (more aggressive)
FADE_IN_DURATION = 0.1  # Faster fade
FADE_OUT_DURATION = 0.1
SCALE_EFFECT = True
SCALE_FACTOR = 1.2  # More dramatic scale. caption enlarge
SCALE_DURATION = 0.3  # Faster animation

# Motion effect constants
ENABLE_IMAGE_MOTION = True
MOTION_TYPE = 'ken_burns'  # Options: '30ken_burns', 'parallax', 'zoom_pan', 'drift'
MOTION_INTENSITY = 0.2  # How much movement (0.0 to 0.5)
ZOOM_RANGE = (1.0, 1.2)  # Min and max zoom levels
PAN_SPEED = 0.05  # How fast to pan across the image

# Background music constants
BACKGROUND_MUSIC_FILE = "inspirational.mp3"  # Name of the background music file
BACKGROUND_MUSIC_VOLUME = 0.5  # Volume level for background music (0.0 to 1.0)

# Narration audio constants
NARRATION_VOLUME = 2  # Volume level for narration audio (1.0 = normal, 1.5 = 50% louder, 2.0 = double volume)

# Watermark constants
WATERMARK_TEXT = ""  # The watermark text to display
WATERMARK_FONT = 'Arial-Bold'  # Font for watermark
WATERMARK_FONTSIZE = 24  # Font size for watermark
WATERMARK_HEIGHT = 50  # Fixed height for watermark (None = auto from fontsize)
WATERMARK_MAX_WIDTH = 300  # Maximum width for watermark (prevents too wide)
WATERMARK_SCALE = 1.0  # Overall scale multiplier (1.0 = normal, 1.5 = 150% size)
WATERMARK_COLOR = 'white'  # Watermark text color
WATERMARK_OPACITY = 0.8  # Overall watermark opacity (0.0 to 1.0)
WATERMARK_STYLE = 'minimal'  # Options: 'glow', 'shadow', 'outline', 'gradient', 'minimal'
WATERMARK_MARGIN = 0  # Distance from bottom-right corner (now actually works)
WATERMARK_BG_STYLE = 'transparent'  # Options: 'black', 'dark_gray', 'transparent', 'custom'
WATERMARK_BG_COLOR = (0, 0, 0)  # Custom RGB color for background (if using 'custom')

# Brain rot effect constants
ENABLE_BREATHING_EFFECT = False  # Disable breathing effect
BREATHING_INTENSITY = 0.01  # Reduced even more
BREATHING_SPEED = 0.5  # Much slower

ENABLE_ROTATION_DRIFT = True  # Enable rotation drift
ROTATION_RANGE = 5.0  # INCREASED rotation for visibility

# New brain rot effects
ENABLE_SHAKE_EFFECT = False  # DISABLE SHAKE - this was causing vibration
SHAKE_INTENSITY = 2.0  # Reduced shake
SHAKE_FREQUENCY = 3.0  # Much less frequent

ENABLE_ZOOM_PULSE = True  # Disable zoom pulse
ZOOM_PULSE_INTENSITY = 0.1  # 
ZOOM_PULSE_SPEED = 4.0  # 

ENABLE_GLITCH_EFFECT = False  # Disable glitch
GLITCH_PROBABILITY = 0.001  # Much lower chance
GLITCH_INTENSITY = 5  # Reduced intensity

# NEW: Retro TV Effects (optimized)
ENABLE_TV_EFFECTS = True
ENABLE_VIGNETTE = True
VIGNETTE_STRENGTH = 0.6  # 0.0 to 1.0
ENABLE_SCANLINES = True
SCANLINE_INTENSITY = 0.8  # 0.0 to 1.0
SCANLINE_COUNT = 540  # Number of scanlines (half of height for performance)
ENABLE_TV_STATIC = True
TV_STATIC_INTENSITY = 0.3  # 0.0 to 1.0
TV_STATIC_FREQUENCY = 0.02  # How often static appears (0.0 to 1.0)
ENABLE_RGB_SHIFT = True
RGB_SHIFT_INTENSITY = 2  # Pixel offset for chromatic aberration
ENABLE_TV_FLICKER = False
TV_FLICKER_INTENSITY = 0.000001  # Brightness variation

# Corner cover constants
ENABLE_CORNER_COVER = True  # Enable corner cover to hide other watermarks
CORNER_COVER_TYPE = 'semi_circle'  # Options: 'semi_circle', 'rectangle' 
CORNER_COVER_SIZE = 0.20  # Size as fraction of image (0.15 = 15% of image size for radius)
CORNER_COVER_DARKNESS = 0.0  # How dark to make it (0.0 = pure black, 1.0 = no change)
CORNER_COVER_OFFSET_X = 85  # Pixels to move circle center left from right edge (positive = left)
CORNER_COVER_OFFSET_Y = 20  # Pixels to move circle center up from bottom edge (positive = up)

class GlobalMotionState:
    """Manages motion state across the ENTIRE video timeline - shared between all threads"""
    
    def __init__(self, img_timestamps, total_duration):
        self.image_configs = {}
        self.total_duration = total_duration
        self._precompute_motion_configs(img_timestamps)
    
    def _precompute_motion_configs(self, img_timestamps):
        """Pre-calculate motion parameters for each image across ENTIRE timeline"""
        for entry in img_timestamps:
            tag = entry['tag']
            start_time = float(entry['start'])
            end_time = float(entry['end_adjusted'])
            duration = end_time - start_time
            
            # Use tag as seed for consistency
            seed_value = hash(tag) % 2147483647
            np.random.seed(seed_value)
            
            config = {
                'tag': tag,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'seed': seed_value
            }
            
            # Pre-compute motion parameters based on type
            if MOTION_TYPE == 'ken_burns':
                config.update({
                    'start_zoom': np.random.uniform(ZOOM_RANGE[0], ZOOM_RANGE[0] + 0.05),
                    'end_zoom': np.random.uniform(ZOOM_RANGE[1] - 0.05, ZOOM_RANGE[1]),
                    # REMOVE X/Y movement - keep centered
                    'start_x': 0,  # No horizontal movement
                    'end_x': 0,    # No horizontal movement
                    'start_y': 0,  # No vertical movement
                    'end_y': 0     # No vertical movement
                })
            elif MOTION_TYPE == 'zoom_pan':
                config.update({
                    'zoom_in': np.random.choice([True, False]),
                    'pan_x': 0  # Remove panning for center zoom
                })
            elif MOTION_TYPE == 'drift':
                config.update({
                    'drift_angle': 0,  # No drift
                    'drift_distance': 0  # No drift distance
                })
            
            # Pre-compute brain rot parameters
            if ENABLE_ROTATION_DRIFT:
                config['rotation_direction'] = np.random.choice([-1, 1])
            
            self.image_configs[tag] = config
    
    def get_motion_transform(self, tag, absolute_time):
        """Get motion transform for given tag at absolute video time (thread-safe)"""
        if tag not in self.image_configs:
            return {'zoom': 1.0, 'position': ('center', 'center'), 'rotation': 0}
        
        config = self.image_configs[tag]
        
        # If we're outside the image's time range, return default
        if absolute_time < config['start_time'] or absolute_time > config['end_time']:
            return {'zoom': 1.0, 'position': ('center', 'center'), 'rotation': 0}
        
        local_time = absolute_time - config['start_time']
        progress = local_time / config['duration'] if config['duration'] > 0 else 0
        progress = np.clip(progress, 0.0, 1.0)
        
        transform = {'zoom': 1.0, 'position': ('center', 'center'), 'rotation': 0}
        
        # Apply motion based on type - CENTER ZOOM ONLY
        if MOTION_TYPE == 'ken_burns':
            zoom = config['start_zoom'] + (config['end_zoom'] - config['start_zoom']) * progress
            # NO X/Y movement - keep perfectly centered
            transform.update({
                'zoom': zoom,
                'position': ('center', 'center')  # Always center
            })
        
        elif MOTION_TYPE == 'zoom_pan':
            start_zoom = ZOOM_RANGE[1] if config['zoom_in'] else ZOOM_RANGE[0]
            end_zoom = ZOOM_RANGE[0] if config['zoom_in'] else ZOOM_RANGE[1]
            zoom = start_zoom + (end_zoom - start_zoom) * progress
            # NO panning - keep centered
            transform.update({
                'zoom': zoom,
                'position': ('center', 'center')  # Always center
            })
        
        elif MOTION_TYPE == 'drift':
            # NO drift - just zoom
            zoom = 1.0 + (0.1 * progress)  # Slight zoom
            transform.update({
                'zoom': zoom,
                'position': ('center', 'center')  # Always center
            })
        
        # Add subtle breathing effect if enabled (but keep centered)
        if ENABLE_BREATHING_EFFECT:
            breathing_cycle = np.sin(absolute_time * 2 * np.pi * BREATHING_SPEED)
            transform['zoom'] *= (1.0 + breathing_cycle * BREATHING_INTENSITY)
        
        # FIXED rotation - make it more visible
        if ENABLE_ROTATION_DRIFT:
            rotation = config['rotation_direction'] * ROTATION_RANGE * progress  # REMOVED the 0.1 multiplier
            transform['rotation'] = rotation
        
        return transform

class SegmentProcessor:
    """Processes individual video segments with global motion continuity"""
    
    def __init__(self, global_motion_state, project_name, base_dir, upscaled_images_dir, subtitles):
        self.global_motion_state = global_motion_state
        self.project_name = project_name
        self.base_dir = base_dir
        self.upscaled_images_dir = upscaled_images_dir
        self.subtitles = subtitles
        
        # OPTIMIZED: Pre-process subtitles for faster lookups
        self._preprocess_subtitles()
        
        # OPTIMIZED: Pre-compute TV effect masks for reuse
        if ENABLE_TV_EFFECTS:
            self._precompute_tv_effects()
    
    def _preprocess_subtitles(self):
        """OPTIMIZED: Pre-sort subtitles and create lookup maps"""
        if not self.subtitles:
            self.subtitle_lookup = {}
            return
            
        # Sort subtitles by start time for binary search
        self.subtitles.sort(key=lambda x: x['start'])
        
        # Create lookup map: subtitle -> next subtitle
        self.subtitle_lookup = {}
        for i, sub in enumerate(self.subtitles):
            if i + 1 < len(self.subtitles):
                self.subtitle_lookup[id(sub)] = self.subtitles[i + 1]
            else:
                self.subtitle_lookup[id(sub)] = None

    def _precompute_tv_effects(self):
        """Pre-compute TV effect masks for optimal performance"""
        # Pre-compute vignette mask
        if ENABLE_VIGNETTE:
            self.vignette_mask = self._create_vignette_mask()
        
        # Pre-compute scanline pattern
        if ENABLE_SCANLINES:
            self.scanline_pattern = self._create_scanline_pattern()
    
    def _create_vignette_mask(self):
        """Create optimized vignette mask using NumPy"""
        y, x = np.ogrid[:TARGET_HEIGHT, :TARGET_WIDTH]
        center_x, center_y = TARGET_WIDTH // 2, TARGET_HEIGHT // 2
        
        # Calculate distance from center (normalized)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        # Create smooth vignette falloff
        vignette = 1 - (dist * VIGNETTE_STRENGTH)
        vignette = np.clip(vignette, 0, 1)
        
        # Convert to 3-channel for RGB
        return np.stack([vignette] * 3, axis=2)
    
    def _create_scanline_pattern(self):
        """Create optimized scanline pattern"""
        scanlines = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.float32)
        
        # Create scanline effect (every other line dimmed)
        step = TARGET_HEIGHT // SCANLINE_COUNT
        for i in range(0, TARGET_HEIGHT, step * 2):
            if i < TARGET_HEIGHT:
                end_line = min(i + step, TARGET_HEIGHT)
                scanlines[i:end_line] *= (1 - SCANLINE_INTENSITY)
        
        return scanlines
    
    def _apply_tv_effects(self, frame, t, absolute_time):
        """Apply optimized TV effects to a frame - FIXED broadcasting"""
        if not ENABLE_TV_EFFECTS:
            return frame
        
        # Get actual frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply vignette (create on-demand for actual frame size)
        if ENABLE_VIGNETTE:
            vignette_mask = self._create_vignette_mask_for_size(frame_width, frame_height)
            frame_float *= vignette_mask
        
        # Apply scanlines (create on-demand for actual frame size)
        if ENABLE_SCANLINES:
            scanline_pattern = self._create_scanline_pattern_for_size(frame_width, frame_height)
            frame_float *= scanline_pattern
        
        # Apply TV static (optimized noise)
        if ENABLE_TV_STATIC:
            # Use time-seeded noise for consistency
            static_seed = int(absolute_time * 1000) % 10000
            np.random.seed(static_seed)
            
            if np.random.random() < TV_STATIC_FREQUENCY:
                # Generate noise for actual frame size
                noise = np.random.random((frame_height, frame_width, 3)) * TV_STATIC_INTENSITY
                frame_float = np.clip(frame_float + noise, 0, 1)
        
        # Apply RGB shift (chromatic aberration)
        if ENABLE_RGB_SHIFT:
            shift = int(RGB_SHIFT_INTENSITY)
            if shift > 0 and shift < frame_width:
                # Shift red channel right, blue channel left
                frame_shifted = frame_float.copy()
                
                # Red channel shift
                frame_shifted[:, shift:, 0] = frame_float[:, :-shift, 0]
                
                # Blue channel shift  
                frame_shifted[:, :-shift, 2] = frame_float[:, shift:, 2]
                
                frame_float = frame_shifted
        
        # Apply TV flicker
        if ENABLE_TV_FLICKER:
            # Use sine wave for smooth flicker
            flicker = 1 + TV_FLICKER_INTENSITY * np.sin(absolute_time * 15)
            frame_float *= flicker
        
        # Convert back to uint8
        frame_output = np.clip(frame_float * 255, 0, 255).astype(np.uint8)
        return frame_output
    
    def _create_vignette_mask_for_size(self, width, height):
        """Create vignette mask for specific frame size"""
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        # Calculate distance from center (normalized)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        # Create smooth vignette falloff
        vignette = 1 - (dist * VIGNETTE_STRENGTH)
        vignette = np.clip(vignette, 0, 1)
        
        # Convert to 3-channel for RGB
        return np.stack([vignette] * 3, axis=2)
    
    def _create_scanline_pattern_for_size(self, width, height):
        """Create scanline pattern for specific frame size"""
        scanlines = np.ones((height, width, 3), dtype=np.float32)
        
        # Create scanline effect (every other line dimmed)
        step = max(1, height // SCANLINE_COUNT)
        for i in range(0, height, step * 2):
            if i < height:
                end_line = min(i + step, height)
                scanlines[i:end_line] *= (1 - SCANLINE_INTENSITY)
        
        return scanlines

    def _create_text_image(self, text, fontsize, color, stroke_color=None, stroke_width=0, width=None):
        """Create text image using PIL instead of MoviePy TextClip"""
        try:
            # Use default font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", fontsize)
                except:
                    font = ImageFont.load_default()
            
            # Get text dimensions - FIXED to account for descenders
            dummy_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            
            # Get proper text metrics including ascent and descent
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # IMPORTANT FIX: Add extra height for descenders and proper vertical spacing
            extra_height = int(fontsize * 0.3)  # Add 30% extra height for descenders
            actual_text_height = text_height + extra_height
            
            # Add padding for stroke
            padding = stroke_width * 2 if stroke_width > 0 else 20  # Increased base padding
            img_width = text_width + padding * 2
            img_height = actual_text_height + padding * 2
            
            # Limit width if specified
            if width and img_width > width:
                img_width = width
            
            # Create image
            img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Calculate text position (centered horizontally, with proper vertical spacing)
            x = (img_width - text_width) // 2
            y = padding  # Start from padding, not centered vertically
            
            # Draw stroke if enabled
            if stroke_width > 0 and stroke_color:
                # Convert color string to RGB
                if isinstance(stroke_color, str):
                    if stroke_color == 'black':
                        stroke_rgb = (0, 0, 0)
                    elif stroke_color == 'white':
                        stroke_rgb = (255, 255, 255)
                    else:
                        stroke_rgb = (0, 0, 0)  # Default to black
                else:
                    stroke_rgb = stroke_color
                
                # Draw text multiple times for stroke effect
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), text, font=font, fill=stroke_rgb)
            
            # Draw main text
            if isinstance(color, str):
                if color == 'white' or color == '#FFFFFF':
                    text_color = (255, 255, 255)
                elif color == 'black':
                    text_color = (0, 0, 0)
                else:
                    text_color = (255, 255, 255)  # Default to white
            else:
                text_color = color
            
            draw.text((x, y), text, font=font, fill=text_color)
            
            # Convert PIL image to numpy array
            img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            print(f"Error creating text image: {e}")
            # Fallback: create simple colored rectangle with text dimensions
            fallback_width = len(text) * fontsize // 2
            fallback_height = int(fontsize * 1.5)  # Increased fallback height
            fallback_img = np.zeros((fallback_height, fallback_width, 4), dtype=np.uint8)
            fallback_img[:, :, 3] = 255  # Full alpha
            return fallback_img

    def _create_text_with_blur_bg(self, text, fontsize, color, stroke_color=None, stroke_width=0):
        """Create text with configurable background and size control"""
        try:
            # Calculate effective font size based on scale
            effective_fontsize = int(fontsize * WATERMARK_SCALE)
            
            # Get font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            dummy_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Apply height constraint if specified
            if WATERMARK_HEIGHT is not None:
                # Calculate scale to achieve target height
                height_scale = WATERMARK_HEIGHT / text_height
                effective_fontsize = int(effective_fontsize * height_scale)
                
                # Recreate font with adjusted size
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                # Recalculate dimensions
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Apply width constraint if specified
            if WATERMARK_MAX_WIDTH is not None and text_width > WATERMARK_MAX_WIDTH:
                # Calculate scale to fit within max width
                width_scale = WATERMARK_MAX_WIDTH / text_width
                effective_fontsize = int(effective_fontsize * width_scale)
                
                # Recreate font with adjusted size
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                # Recalculate dimensions
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Minimal padding for tight fit
            padding = max(6, int(8 * WATERMARK_SCALE))  # Scale padding too
            
            canvas_width = text_width + padding * 2
            canvas_height = text_height + padding * 2
            
            # Create image
            img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Choose background based on style
            radius = max(4, int(6 * WATERMARK_SCALE))  # Scale radius
            
            if WATERMARK_BG_STYLE == 'transparent':
                # No background - just text
                pass
            elif WATERMARK_BG_STYLE == 'black':
                # Pure black background
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(0, 0, 0, 255)  # Pure black
                )
            elif WATERMARK_BG_STYLE == 'dark_gray':
                # Dark gray background
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(40, 40, 40, 255)  # Dark gray
                )
            elif WATERMARK_BG_STYLE == 'custom':
                # Custom color background
                r, g, b = WATERMARK_BG_COLOR
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(r, g, b, 255)  # Custom color
                )
            
            # Calculate text position (centered)
            text_x = padding
            text_y = padding
            
            # Draw stroke if enabled
            if stroke_width > 0 and stroke_color:
                scaled_stroke_width = max(1, int(stroke_width * WATERMARK_SCALE))
                stroke_rgb = (0, 0, 0) if stroke_color == 'black' else (255, 255, 255)
                
                for dx in range(-scaled_stroke_width, scaled_stroke_width + 1):
                    for dy in range(-scaled_stroke_width, scaled_stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), text, font=font, fill=stroke_rgb)
            
            # Draw main text
            text_color = (255, 255, 255) if color == 'white' else (255, 255, 255)
            draw.text((text_x, text_y), text, font=font, fill=text_color)
            
            return np.array(img)
            
        except Exception as e:
            print(f"Error creating text with background: {e}")
            return self._create_text_image(text, fontsize, color, stroke_color, stroke_width)

    def _create_glow_text_with_blur(self, text, fontsize, color):
        """Create glowing text with configurable background and size control"""
        try:
            # Calculate effective font size based on scale
            effective_fontsize = int(fontsize * WATERMARK_SCALE)
            
            # Get font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            dummy_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Apply height constraint if specified
            if WATERMARK_HEIGHT is not None:
                height_scale = WATERMARK_HEIGHT / text_height
                effective_fontsize = int(effective_fontsize * height_scale)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Apply width constraint if specified
            if WATERMARK_MAX_WIDTH is not None and text_width > WATERMARK_MAX_WIDTH:
                width_scale = WATERMARK_MAX_WIDTH / text_width
                effective_fontsize = int(effective_fontsize * width_scale)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Account for glow effect
            glow_radius = max(6, int(8 * WATERMARK_SCALE))
            padding = max(4, int(6 * WATERMARK_SCALE))
            canvas_width = text_width + (glow_radius + padding) * 2
            canvas_height = text_height + (glow_radius + padding) * 2
            
            # Create base image
            img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Choose background based on style
            radius = max(6, int(8 * WATERMARK_SCALE))
            
            if WATERMARK_BG_STYLE == 'transparent':
                # No background
                pass
            elif WATERMARK_BG_STYLE == 'black':
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(0, 0, 0, 255)  # Pure black
                )
            elif WATERMARK_BG_STYLE == 'dark_gray':
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(40, 40, 40, 255)  # Dark gray
                )
            elif WATERMARK_BG_STYLE == 'custom':
                r, g, b = WATERMARK_BG_COLOR
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(r, g, b, 255)  # Custom color
                )
            
            text_x = glow_radius + padding
            text_y = glow_radius + padding
            
            # Create glow layers
            from PIL import ImageFilter
            glow_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_img)
            
            # Draw glow
            glow_layers = max(2, int(3 * WATERMARK_SCALE))
            for i in range(glow_layers):
                offset = i + 1
                alpha = max(30, 100 - (i * 25))
                
                for dx in range(-offset, offset + 1):
                    for dy in range(-offset, offset + 1):
                        if dx != 0 or dy != 0:
                            glow_draw.text((text_x + dx, text_y + dy), text, 
                                         font=font, fill=(255, 255, 255, alpha))
            
            # Blur the glow
            blur_radius = max(2, int(3 * WATERMARK_SCALE))
            glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Composite glow with main image
            img = Image.alpha_composite(img, glow_img)
            
            # Draw main text on top
            draw = ImageDraw.Draw(img)
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
            
            return np.array(img)
            
        except Exception as e:
            print(f"Error creating glow text: {e}")
            return self._create_glow_text(text, fontsize, color)

    def _create_shadow_text_with_blur(self, text, fontsize, color):
        """Create shadow text with configurable background and size control"""
        try:
            # Calculate effective font size based on scale
            effective_fontsize = int(fontsize * WATERMARK_SCALE)
            
            # Get font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            dummy_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Apply height constraint if specified
            if WATERMARK_HEIGHT is not None:
                height_scale = WATERMARK_HEIGHT / text_height
                effective_fontsize = int(effective_fontsize * height_scale)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Apply width constraint if specified
            if WATERMARK_MAX_WIDTH is not None and text_width > WATERMARK_MAX_WIDTH:
                width_scale = WATERMARK_MAX_WIDTH / text_width
                effective_fontsize = int(effective_fontsize * width_scale)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Account for shadow
            shadow_offset = max(2, int(3 * WATERMARK_SCALE))
            padding = max(4, int(6 * WATERMARK_SCALE))
            canvas_width = text_width + shadow_offset + padding * 2
            canvas_height = text_height + shadow_offset + padding * 2
            
            img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Choose background based on style
            radius = max(4, int(6 * WATERMARK_SCALE))
            
            if WATERMARK_BG_STYLE == 'transparent':
                # No background
                pass
            elif WATERMARK_BG_STYLE == 'black':
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(0, 0, 0, 255)  # Pure black
                )
            elif WATERMARK_BG_STYLE == 'dark_gray':
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(40, 40, 40, 255)  # Dark gray
                )
            elif WATERMARK_BG_STYLE == 'custom':
                r, g, b = WATERMARK_BG_COLOR
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(r, g, b, 255)  # Custom color
                )
            
            text_x = padding
            text_y = padding
            
            # Draw shadow
            draw.text((text_x + shadow_offset, text_y + shadow_offset), text, 
                     font=font, fill=(0, 0, 0, 180))  # Semi-transparent black shadow
            
            # Draw main text
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
            
            return np.array(img)
            
        except Exception as e:
            print(f"Error creating shadow text: {e}")
            return self._create_shadow_text(text, fontsize, color)

    def _create_gradient_text_with_blur(self, text, fontsize):
        """Create text with configurable background and size control"""
        try:
            # Calculate effective font size based on scale
            effective_fontsize = int(fontsize * WATERMARK_SCALE)
            
            # Get font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size
            dummy_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Apply height constraint if specified
            if WATERMARK_HEIGHT is not None:
                height_scale = WATERMARK_HEIGHT / text_height
                effective_fontsize = int(effective_fontsize * height_scale)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Apply width constraint if specified
            if WATERMARK_MAX_WIDTH is not None and text_width > WATERMARK_MAX_WIDTH:
                width_scale = WATERMARK_MAX_WIDTH / text_width
                effective_fontsize = int(effective_fontsize * width_scale)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", effective_fontsize)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            padding = max(6, int(8 * WATERMARK_SCALE))
            canvas_width = text_width + padding * 2
            canvas_height = text_height + padding * 2
            
            # Create image
            img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Choose background based on style
            radius = max(6, int(8 * WATERMARK_SCALE))
            
            if WATERMARK_BG_STYLE == 'transparent':
                # No background
                pass
            elif WATERMARK_BG_STYLE == 'black':
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(0, 0, 0, 255)  # Pure black
                )
            elif WATERMARK_BG_STYLE == 'dark_gray':
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(40, 40, 40, 255)  # Dark gray
                )
            elif WATERMARK_BG_STYLE == 'custom':
                r, g, b = WATERMARK_BG_COLOR
                draw.rounded_rectangle(
                    [0, 0, canvas_width, canvas_height],
                    radius=radius,
                    fill=(r, g, b, 255)  # Custom color
                )
            
            # Draw text
            draw.text((padding, padding), text, font=font, fill=(255, 255, 255, 255))
            
            return np.array(img)
            
        except Exception as e:
            print(f"Error creating text: {e}")
            return self._create_gradient_text(text, fontsize)

    def _create_segment_watermark(self, duration):
        """Create watermark with PROPER positioning - actually sticks to corner"""
        try:
            if WATERMARK_STYLE == 'glow':
                watermark_img = self._create_glow_text_with_blur(
                    WATERMARK_TEXT,
                    WATERMARK_FONTSIZE,
                    WATERMARK_COLOR
                )
            elif WATERMARK_STYLE == 'shadow':
                watermark_img = self._create_shadow_text_with_blur(
                    WATERMARK_TEXT,
                    WATERMARK_FONTSIZE,
                    WATERMARK_COLOR
                )
            elif WATERMARK_STYLE == 'outline':
                watermark_img = self._create_text_with_blur_bg(
                    WATERMARK_TEXT,
                    WATERMARK_FONTSIZE,
                    WATERMARK_COLOR,
                    'black',
                    4  # Thick outline
                )
            elif WATERMARK_STYLE == 'minimal':
                watermark_img = self._create_text_with_blur_bg(
                    WATERMARK_TEXT,
                    WATERMARK_FONTSIZE,
                    WATERMARK_COLOR
                )
            else:  # gradient
                watermark_img = self._create_gradient_text_with_blur(
                    WATERMARK_TEXT,
                    WATERMARK_FONTSIZE
                )
            
            # Convert to ImageClip
            txt_clip = ImageClip(watermark_img, duration=duration, transparent=True)
            
            # Apply overall opacity
            if WATERMARK_OPACITY < 1.0:
                txt_clip = txt_clip.set_opacity(WATERMARK_OPACITY)
            
            # FIXED POSITIONING - Calculate EXACT position for bottom-right
            txt_height, txt_width = watermark_img.shape[:2]
            
            # Position from bottom-right corner with margin
            watermark_x = TARGET_WIDTH - txt_width - WATERMARK_MARGIN
            watermark_y = TARGET_HEIGHT - txt_height - WATERMARK_MARGIN
            
            # Ensure it doesn't go off screen
            watermark_x = max(0, watermark_x)
            watermark_y = max(0, watermark_y)
            
            return txt_clip.set_position((watermark_x, watermark_y))
            
        except Exception as e:
            print(f"Error creating watermark: {e}")
            return None

    def _create_segment_subtitle_clip(self, sub, segment_start):
        """Create subtitle clip using PIL instead of MoviePy TextClip"""
        try:
            clip_start = sub['segment_start']
            clip_end = sub['segment_end']
            duration = clip_end - clip_start
            
            if duration <= 0:
                return None
            
            # Create text image using PIL
            text_img = self._create_text_image(
                sub['text'],
                SUBTITLE_FONTSIZE,
                SUBTITLE_COLOR,
                SUBTITLE_STROKE_COLOR,
                SUBTITLE_STROKE_WIDTH,
                TARGET_WIDTH
            )
            
            # Convert to ImageClip
            txt_clip = ImageClip(text_img, duration=duration, transparent=True)
            txt_clip = txt_clip.set_position(('center', SUBTITLE_POSITION_Y))
            
            # SMART SCALE EFFECT - Only for emphasis words
            if SCALE_EFFECT:
                should_emphasize = self._should_emphasize_word(sub, segment_start)
                
                if should_emphasize:
                    def optimized_scale_function(t):
                        if t < SCALE_DURATION:
                            progress = t / SCALE_DURATION
                            if progress < 0.6:
                                scale = 1.0 + (SCALE_FACTOR - 1.0) * (progress / 0.6)
                            else:
                                overshoot = 1.3
                                bounce_progress = (progress - 0.6) / 0.4
                                scale = SCALE_FACTOR * overshoot - (SCALE_FACTOR * (overshoot - 1.0)) * bounce_progress
                            return scale
                        return 1.0
                    
                    txt_clip = txt_clip.resize(optimized_scale_function)
            
            # Set timing within segment
            txt_clip = txt_clip.set_start(clip_start).set_end(clip_end)
            return txt_clip
            
        except Exception as e:
            print(f"Error creating subtitle '{sub['text']}': {e}")
            return None

    def process_segment(self, segment_start, segment_end, segment_index, img_timestamps):
        """Process a single segment with continuous motion"""
        print(f"Processing segment {segment_index}: {segment_start:.2f}s - {segment_end:.2f}s")
        
        segment_duration = segment_end - segment_start
        all_clips = []
        
        try:
            # Filter images for this segment
            segment_images = []
            for entry in img_timestamps:
                img_start = float(entry['start'])
                img_end = float(entry['end_adjusted'])
                
                # Check if image overlaps with this segment
                if img_start < segment_end and img_end > segment_start:
                    # Adjust timing relative to segment
                    adjusted_entry = entry.copy()
                    adjusted_entry['segment_start'] = max(0, img_start - segment_start)
                    adjusted_entry['segment_end'] = min(segment_duration, img_end - segment_start)
                    adjusted_entry['absolute_start'] = img_start
                    segment_images.append(adjusted_entry)
            
            # Create image clips for this segment
            for i, entry in enumerate(segment_images):
                clip = self._create_segment_image_clip(entry, segment_start, segment_duration)
                if clip:
                    all_clips.append(clip)
            
            # Filter subtitles for this segment
            segment_subtitles = []
            for sub in self.subtitles:
                if sub['start'] < segment_end and sub['end'] > segment_start:
                    adjusted_sub = sub.copy()
                    adjusted_sub['segment_start'] = max(0, sub['start'] - segment_start)
                    adjusted_sub['segment_end'] = min(segment_duration, sub['end'] - segment_start)
                    adjusted_sub['absolute_start'] = sub['start']
                    segment_subtitles.append(adjusted_sub)
            
            # Create subtitle clips for this segment
            for i, sub in enumerate(segment_subtitles):
                clip = self._create_segment_subtitle_clip(sub, segment_start)
                if clip:
                    all_clips.append(clip)
            
            # Create watermark for this segment
            watermark_clip = self._create_segment_watermark(segment_duration)
            if watermark_clip:
                all_clips.append(watermark_clip)
            
            # Compose segment video
            if not all_clips:
                # Create black segment if no content
                bg = ColorClip(size=(TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0), duration=segment_duration)
                segment_video = bg
            else:
                # Create black background first, then add all clips on top
                bg = ColorClip(size=(TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0), duration=segment_duration)
                all_clips.insert(0, bg)  # Add background as first layer
                segment_video = CompositeVideoClip(all_clips, size=(TARGET_WIDTH, TARGET_HEIGHT))
                segment_video = segment_video.set_duration(segment_duration)
            
            # Render segment to file with REAL MoviePy progress
            temp_dir = tempfile.mkdtemp()
            segment_path = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
            
            print(f"Rendering segment {segment_index} to file...")
            
            # Use MoviePy's built-in progress display
            segment_video.write_videofile(
                segment_path,
                fps=FPS,
                codec='libx264',
                audio=False,
                ffmpeg_params=['-preset', 'ultrafast', '-crf', '18'],
                verbose=True,  # Enable MoviePy's own progress
                logger='bar'   # Use MoviePy's progress bar
            )
            
            print(f"Segment {segment_index} rendered: {segment_path}")
            return segment_path, segment_index
            
        except Exception as e:
            print(f"Error rendering segment {segment_index}: {e}")
            return None, segment_index

    def _resize_clip_optimized(self, img_clip):
        """FIXED resize - PROPERLY fill entire screen with no gaps"""
        original_width = img_clip.w
        original_height = img_clip.h
        aspect_ratio = original_width / original_height
        target_aspect = TARGET_WIDTH / TARGET_HEIGHT
        
        # Calculate dimensions to OVERFILL the screen (crop excess)
        if aspect_ratio > target_aspect:
            # Image is wider - scale by height and crop width
            new_height = TARGET_HEIGHT
            new_width = int(TARGET_HEIGHT * aspect_ratio)
        else:
            # Image is taller - scale by width and crop height
            new_width = TARGET_WIDTH
            new_height = int(TARGET_WIDTH / aspect_ratio)
        
        # ENSURE we at least fill the target size
        new_width = max(TARGET_WIDTH, new_width)
        new_height = max(TARGET_HEIGHT, new_height)
        
        # Resize the clip
        resized_clip = img_clip.resize((new_width, new_height))
        
        # Position at (0,0) so top-left corner touches screen corner
        # Then center it properly
        x_offset = -(new_width - TARGET_WIDTH) // 2
        y_offset = -(new_height - TARGET_HEIGHT) // 2
        
        # Set position to ensure image fills screen completely
        resized_clip = resized_clip.set_position((x_offset, y_offset))
        
        return resized_clip

    def _apply_corner_cover_to_image(self, get_frame, t):
        """Apply quarter-circle corner cover with adjustable center offset"""
        if not ENABLE_CORNER_COVER:
            return get_frame(t)
        
        frame = get_frame(t)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate cover size based on CORNER_COVER_SIZE
        cover_radius = int(min(frame_width, frame_height) * CORNER_COVER_SIZE)
        
        if CORNER_COVER_TYPE == 'semi_circle':
            # Create quarter-circle mask in bottom-right corner with offset
            y_indices, x_indices = np.ogrid[:frame_height, :frame_width]
            
            # Center of the quarter circle with offset from corner
            center_x = frame_width - CORNER_COVER_OFFSET_X   # Move left from right edge
            center_y = frame_height - CORNER_COVER_OFFSET_Y  # Move up from bottom edge
            
            # Create circular mask - only the quarter that's inside the frame
            distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            mask = distances <= cover_radius
            
            # Apply cover only to the quarter circle area
            if CORNER_COVER_DARKNESS == 0.0:
                # Pure black
                frame[mask] = [0, 0, 0]
            else:
                # Darken by specified amount
                frame[mask] = frame[mask] * CORNER_COVER_DARKNESS
                
        elif CORNER_COVER_TYPE == 'rectangle':
            # Simple rectangular cover in bottom-right corner with offset
            cover_width = int(frame_width * CORNER_COVER_SIZE)
            cover_height = int(frame_height * CORNER_COVER_SIZE)
            
            # Calculate bottom-right area with offset
            start_x = frame_width - cover_width - CORNER_COVER_OFFSET_X
            start_y = frame_height - cover_height - CORNER_COVER_OFFSET_Y
            
            # Ensure we don't go outside frame bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(frame_width, start_x + cover_width)
            end_y = min(frame_height, start_y + cover_height)
            
            # Apply cover
            if CORNER_COVER_DARKNESS == 0.0:
                # Pure black
                frame[start_y:end_y, start_x:end_x] = [0, 0, 0]
            else:
                # Darken by specified amount
                frame[start_y:end_y, start_x:end_x] = frame[start_y:end_y, start_x:end_x] * CORNER_COVER_DARKNESS
        
        return frame

    def _create_segment_image_clip(self, entry, segment_start, segment_duration):
        """Create image clip for segment with global motion continuity and corner cover"""
        tag = entry['tag']
        clip_start = entry['segment_start']
        clip_end = entry['segment_end']
        clip_duration = clip_end - clip_start
        
        if clip_duration <= 0:
            return None
        
        image_filename = f"{tag}.png"
        image_path = os.path.join(self.upscaled_images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image '{image_path}' not found. Skipping.")
            return None
        
        try:
            # Create base image clip
            img_clip = ImageClip(image_path).set_duration(clip_duration)
            
            # APPLY CORNER COVER FIRST (before any motion)
            img_clip = img_clip.fl(self._apply_corner_cover_to_image)
            
            # Resize to COMPLETELY FILL screen
            img_clip = self._resize_clip_optimized(img_clip)
            
            # Apply ONLY center zoom motion
            if ENABLE_IMAGE_MOTION:
                def continuous_resize_func(t):
                    absolute_time = segment_start + clip_start + t
                    transform = self.global_motion_state.get_motion_transform(tag, absolute_time)
                    return transform.get('zoom', 1.0)
                
                def continuous_position_func(t):
                    # ALWAYS return center - no movement
                    return ('center', 'center')
                
                img_clip = img_clip.resize(continuous_resize_func)
                img_clip = img_clip.set_position(continuous_position_func)
            else:
                # Keep image centered
                img_clip = img_clip.set_position('center')
            
            # FIXED rotation - make it visible
            if ENABLE_ROTATION_DRIFT:
                def continuous_rotation_func(t):
                    absolute_time = segment_start + clip_start + t
                    transform = self.global_motion_state.get_motion_transform(tag, absolute_time)
                    return transform.get('rotation', 0)
                
                img_clip = img_clip.rotate(continuous_rotation_func)
            
            # APPLY TV EFFECTS to the image clip
            if ENABLE_TV_EFFECTS:
                def tv_effect_func(get_frame, t):
                    frame = get_frame(t)
                    absolute_time = segment_start + clip_start + t
                    return self._apply_tv_effects(frame, t, absolute_time)
                
                img_clip = img_clip.fl(tv_effect_func)
            
            # Set timing within segment
            final_clip = img_clip.set_start(clip_start).set_end(clip_end)
            
            return final_clip
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def _should_emphasize_word(self, current_sub, segment_start):
        """OPTIMIZED: Determine emphasis using ONLY timing criteria"""
        current_duration = current_sub['end'] - current_sub['start']
        
        # OPTIMIZED: O(1) lookup for next subtitle
        next_sub = self.subtitle_lookup.get(id(current_sub))
        
        # Calculate gap to next word
        gap_to_next = next_sub['start'] - current_sub['end'] if next_sub else 0.0
        
        # ONLY timing-based emphasis criteria:
        has_dramatic_pause = gap_to_next > 0.3  # Gap to next word > 0.3 seconds
        has_long_duration = current_duration > 0.4  # Word duration > 0.4 seconds
        
        should_emphasize = has_dramatic_pause or has_long_duration
        
        return should_emphasize

def create_threaded_video(project_name, base_dir, upscaled_images_dir, img_timestamps, subtitles):
    """Create video using threaded segment processing with continuous motion"""
    
    # Calculate total duration
    total_duration = max(float(entry['end_adjusted']) for entry in img_timestamps)
    
    # Initialize global motion state
    print("Initializing motion state...")
    global_motion_state = GlobalMotionState(img_timestamps, total_duration)
    
    # Calculate segments
    segments = []
    num_segments = math.ceil(total_duration / SEGMENT_DURATION)
    
    for i in range(num_segments):  # FIXED: Added range()
        segment_start = i * SEGMENT_DURATION
        segment_end = min((i + 1) * SEGMENT_DURATION, total_duration)
        segments.append((segment_start, segment_end, i))
    
    print(f"Processing {len(segments)} segments with {psutil.cpu_count()} threads...")
    
    # Process segments in parallel
    segment_files = []
    processor = SegmentProcessor(global_motion_state, project_name, base_dir, upscaled_images_dir, subtitles)
    
    # Simple counter for completed segments
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=min(len(segments), psutil.cpu_count())) as executor:
        # FIXED: Store future to segment mapping correctly
        future_to_segment = {}
        for segment_start, segment_end, segment_index in segments:
            future = executor.submit(
                processor.process_segment, 
                segment_start, segment_end, segment_index, img_timestamps
            )
            future_to_segment[future] = segment_index
        
        # FIXED: Collect results properly
        for future in as_completed(future_to_segment.keys()):
            segment_path, returned_index = future.result()
            if segment_path:
                segment_files.append((returned_index, segment_path))
            
            completed_count += 1
            print(f"\n=== Completed {completed_count}/{len(segments)} segments ===\n")
    
    # Sort segments by index
    segment_files.sort(key=lambda x: x[0])
    
    if not segment_files:
        print("No segments were created successfully!")
        return None
    
    # Stitch segments with FFmpeg
    print("Stitching segments with FFmpeg...")
    temp_dir = os.path.dirname(segment_files[0][1])
    filelist_path = os.path.join(temp_dir, "segments.txt")
    
    with open(filelist_path, 'w') as f:
        for _, segment_path in segment_files:
            f.write(f"file '{segment_path}'\n")
    
    output_path = os.path.join(base_dir, f"{project_name}_stitched.mp4")
    
    # FFmpeg concat with exact timing preservation
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', filelist_path,
        '-c', 'copy',  # Copy streams without re-encoding
        '-avoid_negative_ts', 'make_zero',
        output_path
    ]
    
    print("Running FFmpeg concatenation...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup temp files
    print("Cleaning up temporary files...")
    for _, segment_path in segment_files:
        if os.path.exists(segment_path):
            os.unlink(segment_path)
    
    shutil.rmtree(temp_dir)
    
    if result.returncode != 0:
        print(f"FFmpeg stitching failed: {result.stderr}")
        return None
    
    print(f"Video stitched successfully: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create threaded video with continuous motion')
    parser.add_argument('project_name', help='Name of the project')
    parser.add_argument('--render', action='store_true', help='Render the final video instead of preview')
    
    args = parser.parse_args()
    project_name = args.project_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "0-project-files", project_name))

    if not os.path.exists(base_dir):
        print(f"Project directory '{base_dir}' does not exist.")
        sys.exit(1)

    upscaled_images_dir = os.path.join(base_dir, "upscaled_images")
    if not os.path.exists(upscaled_images_dir):
        print(f"Upscaled images directory '{upscaled_images_dir}' does not exist.")
        sys.exit(1)

    img_timestamps_file_path = os.path.join(base_dir, f'{project_name}_img_prompts.json')
    if not os.path.exists(img_timestamps_file_path):
        print(f"Image timestamps file '{img_timestamps_file_path}' does not exist.")
        sys.exit(1)

    # Load data with progress
    print("Loading image timestamps...")
    with open(img_timestamps_file_path, 'r') as f:
        img_timestamps = json.load(f)
    print(f"Loaded {len(img_timestamps)} image entries")

    # Load subtitles
    srt_path = os.path.join(base_dir, f"{project_name}_wordlevel.srt")
    subtitles = []
    if os.path.exists(srt_path):
        subtitles = parse_srt(srt_path)  # Now has built-in progress bar
    else:
        print("No subtitle file found")

    if args.render:
        print("Creating threaded video with continuous motion...")
        
        # Create the video using threading
        stitched_video_path = create_threaded_video(
            project_name, base_dir, upscaled_images_dir, img_timestamps, subtitles
        )
        
        if not stitched_video_path:
            print("Failed to create video!")
            sys.exit(1)
        
        # Add audio with FFmpeg
        audio_path = os.path.join(base_dir, f"{project_name}.wav")
        background_music_path = os.path.join(script_dir, "..", "common_assets", BACKGROUND_MUSIC_FILE)
        final_output_path = os.path.join(base_dir, f"{project_name}_final.mp4")
        
        if add_audio_with_ffmpeg(stitched_video_path, audio_path, background_music_path, final_output_path):
            print(f"Final video with audio: {final_output_path}")
            # Clean up intermediate file
            if os.path.exists(stitched_video_path):
                os.unlink(stitched_video_path)
        
    else:
        # Preview mode (first 30 seconds)
        print("Creating preview...")
        preview_duration = min(5, max(float(entry['end_adjusted']) for entry in img_timestamps))
        
        # Filter for preview
        preview_timestamps = []
        for entry in img_timestamps:
            if float(entry['start']) < preview_duration:
                preview_entry = entry.copy()
                preview_entry['end_adjusted'] = str(min(float(entry['end_adjusted']), preview_duration))
                preview_timestamps.append(preview_entry)
        
        preview_subtitles = [sub for sub in subtitles if sub['start'] < preview_duration]
        
        preview_path = create_threaded_video(
            f"{project_name}_preview", base_dir, upscaled_images_dir, 
            preview_timestamps, preview_subtitles
        )
        
        if preview_path:
            subprocess.run(['ffplay', '-autoexit', preview_path])

def parse_srt(srt_path):
    """OPTIMIZED: Parse SRT file with regex and progress bar"""
    import re
    
    subtitles = []
    
    print(f"Loading subtitles from {srt_path}...")
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # OPTIMIZED: Use regex for faster timestamp parsing
    pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})'
    )
    
    blocks = content.split('\n\n')
    
    # Parse with progress bar
    parse_progress = tqdm(blocks, desc="Parsing subtitles", unit="subtitle")
    
    for block in parse_progress:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # OPTIMIZED: Regex parsing instead of string splits
            match = pattern.search(lines[1])
            if match:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                
                # OPTIMIZED: Direct calculation instead of function calls
                start_time = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                end_time = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                
                text = ' '.join(lines[2:])
                
                subtitles.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
    
    parse_progress.close()
    print(f"Loaded {len(subtitles)} subtitles")
    return subtitles

def add_audio_with_ffmpeg(video_path, audio_path, background_music_path, output_path):
    """Use FFmpeg to add audio to the concatenated video with two-pass compression"""
    
    # STEP 1: Add audio (keeping fast video encoding)
    temp_video_with_audio = output_path.replace('.mp4', '_temp_with_audio.mp4')
    
    cmd = ['ffmpeg', '-y', '-i', video_path]
    
    filter_complex = []
    audio_inputs = 1
    
    # Add main audio if available
    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-i', audio_path])
        audio_inputs += 1
        main_audio_index = audio_inputs - 1
        filter_complex.append(f"[{main_audio_index}:a]volume={NARRATION_VOLUME}[main_audio]")
    
    # Add background music if available
    if background_music_path and os.path.exists(background_music_path):
        cmd.extend(['-i', background_music_path])
        audio_inputs += 1
        bg_music_index = audio_inputs - 1
        
        # Loop background music and set volume
        if filter_complex:
            filter_complex.append(f"[{bg_music_index}:a]aloop=loop=-1:size=2e+09,volume={BACKGROUND_MUSIC_VOLUME}[bg_music]")
            filter_complex.append("[main_audio][bg_music]amix=inputs=2:duration=first:dropout_transition=2[audio_out]")
            audio_map = "[audio_out]"
        else:
            filter_complex.append(f"[{bg_music_index}:a]aloop=loop=-1:size=2e+09,volume={BACKGROUND_MUSIC_VOLUME}[audio_out]")
            audio_map = "[audio_out]"
    else:
        audio_map = "[main_audio]" if filter_complex else None
    
    # Add filter complex if we have audio processing
    if filter_complex:
        cmd.extend(['-filter_complex', ';'.join(filter_complex)])
        cmd.extend(['-map', '0:v', '-map', audio_map])
    elif audio_path and os.path.exists(audio_path):
        # Simple case: just copy main audio
        cmd.extend(['-map', '0:v', '-map', '1:a'])
    else:
        # No audio
        cmd.extend(['-map', '0:v'])
    
    # STEP 1: Create temp file with audio but keep fast video encoding
    cmd.extend([
        '-c:v', 'copy',  # Keep the ultrafast-encoded video as-is
        '-c:a', 'aac',   # Encode audio as AAC
        '-shortest',     # End when shortest stream ends
        temp_video_with_audio
    ])
    
    print(f"Step 1: Adding audio (keeping fast video encoding)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg audio error: {result.stderr}")
        return False
    
    print(f"Step 1 complete: {temp_video_with_audio}")
    
    # STEP 2: Re-compress the video with much better compression settings
    print("Step 2: Re-compressing video for much smaller file size...")
    print("(This will take longer but reduce file size by 60-80%)")
    
    compress_cmd = [
        'ffmpeg', '-y',
        '-i', temp_video_with_audio,
        '-c:v', 'libx264',
        '-preset', 'medium',    # Much better compression than 'ultrafast'
        '-crf', '23',          # Higher CRF = smaller file (was 18)
        '-c:a', 'copy',        # Don't re-encode the audio we just added
        '-movflags', '+faststart',  # Optimize for web streaming
        output_path
    ]
    
    print("Running FFmpeg compression (this is the slow part)...")
    result = subprocess.run(compress_cmd, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(temp_video_with_audio):
        os.unlink(temp_video_with_audio)
        print("Cleaned up temporary file")
    
    if result.returncode != 0:
        print(f"FFmpeg compression error: {result.stderr}")
        return False
    
    print(f"Successfully compressed video: {output_path}")
    print("File size should now be 60-80% smaller than before!")
    return True

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")