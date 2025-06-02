import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
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
from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip, AudioFileClip, TextClip
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Add progress bar

# Constants
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 24
SEGMENT_DURATION = 20  # seconds per segment

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
MOTION_TYPE = 'ken_burns'  # Options: 'ken_burns', 'parallax', 'zoom_pan', 'drift'
MOTION_INTENSITY = 0.2  # How much movement (0.0 to 0.5)
ZOOM_RANGE = (1.0, 1.2)  # Min and max zoom levels
PAN_SPEED = 0.05  # How fast to pan across the image

# Background music constants
BACKGROUND_MUSIC_FILE = "curious.mp3"  # Name of the background music file
BACKGROUND_MUSIC_VOLUME = 0.5  # Volume level for background music (0.0 to 1.0)

# Watermark constants
WATERMARK_TEXT = "@unhingedWizard"  # The watermark text to display
WATERMARK_FONTSIZE = 40  # Size of watermark text
WATERMARK_COLOR = "white"  # White text color (use color names for better compatibility)
WATERMARK_BG_COLOR = (0, 0, 0)  # Black background color as RGB tuple
WATERMARK_FONT = 'Arial-Bold'  # Font for watermark
WATERMARK_PADDING = 10  # Padding around the text
WATERMARK_MARGIN = 20  # Distance from bottom-right corner

# Brain rot effect constants
ENABLE_BREATHING_EFFECT = False  # Disable breathing effect
BREATHING_INTENSITY = 0.01  # Reduced even more
BREATHING_SPEED = 0.5  # Much slower

ENABLE_ROTATION_DRIFT = True  # Enable rotation drift
ROTATION_RANGE = 3.0  # INCREASED rotation for visibility

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
        
        # OPTIMIZED: Pre-compile emphasis words set for O(1) lookup
        self._emphasis_words = {
            'wow', 'amazing', 'incredible', 'unbelievable', 'whoa', 'damn', 
            'shit', 'fuck', 'holy', 'god', 'jesus', 'what', 'why', 'how', 
            'really', 'seriously', 'literally', 'actually', 'definitely', 
            'absolutely', 'completely', 'totally', 'perfect', 'insane', 
            'crazy', 'wild', 'epic', 'legendary'
        }
    
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

    def _create_segment_image_clip(self, entry, segment_start, segment_duration):
        """Create image clip for segment with global motion continuity"""
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
                    return transform.get('rotation', 0)  # REMOVED the 0.1 multiplier
                
                img_clip = img_clip.rotate(continuous_rotation_func)
            
            # Set timing within segment
            final_clip = img_clip.set_start(clip_start).set_end(clip_end)
            
            return final_clip
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def _create_segment_subtitle_clip(self, sub, segment_start):
        """Create subtitle clip for segment with global continuity"""
        try:
            clip_start = sub['segment_start']
            clip_end = sub['segment_end']
            duration = clip_end - clip_start
            
            if duration <= 0:
                return None
            
            # Create base text clip
            txt_clip = TextClip(
                sub['text'],
                fontsize=SUBTITLE_FONTSIZE,
                color=SUBTITLE_COLOR,
                font=SUBTITLE_FONT,
                stroke_color=SUBTITLE_STROKE_COLOR,
                stroke_width=SUBTITLE_STROKE_WIDTH
            ).set_duration(duration).set_position(('center', SUBTITLE_POSITION_Y))
            
            # SMART SCALE EFFECT - Only for emphasis words
            if SCALE_EFFECT:
                # Calculate if this word should be emphasized
                should_emphasize = self._should_emphasize_word(sub, segment_start)
                
                if should_emphasize:
                    def optimized_scale_function(t):
                        # Scale effect happens at the beginning of EMPHASIZED subtitles only
                        if t < SCALE_DURATION:
                            progress = t / SCALE_DURATION
                            if progress < 0.6:
                                scale = 1.0 + (SCALE_FACTOR - 1.0) * (progress / 0.6)
                            else:
                                overshoot = 1.3  # MORE overshoot for emphasis
                                bounce_progress = (progress - 0.6) / 0.4
                                scale = SCALE_FACTOR * overshoot - (SCALE_FACTOR * (overshoot - 1.0)) * bounce_progress
                            return scale
                        return 1.0
                    
                    txt_clip = txt_clip.resize(optimized_scale_function)
            
            # Apply shake using absolute timeline
            if ENABLE_SHAKE_EFFECT:
                def optimized_subtitle_shake(t):
                    absolute_time = segment_start + clip_start + t
                    shake_seed = int(absolute_time * SHAKE_FREQUENCY * 1000) % 1000
                    np.random.seed(shake_seed)
                    
                    base_y = SUBTITLE_POSITION_Y
                    y_shake = np.random.uniform(-SHAKE_INTENSITY/2, SHAKE_INTENSITY/2)
                    return ('center', base_y + y_shake)
                
                txt_clip = txt_clip.set_position(optimized_subtitle_shake)
            
            # Set timing within segment
            txt_clip = txt_clip.set_start(clip_start).set_end(clip_end)
            return txt_clip
            
        except Exception as e:
            print(f"Error creating subtitle '{sub['text']}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _should_emphasize_word(self, current_sub, segment_start):
        """OPTIMIZED: Determine emphasis using pre-computed lookups"""
        current_duration = current_sub['end'] - current_sub['start']
        
        # OPTIMIZED: O(1) lookup for next subtitle
        next_sub = self.subtitle_lookup.get(id(current_sub))
        
        # Calculate gap to next word
        gap_to_next = next_sub['start'] - current_sub['end'] if next_sub else 0.0
        
        # OPTIMIZED: Simple boolean checks (no list iterations)
        has_dramatic_pause = gap_to_next > 0.3
        has_long_duration = current_duration > 0.4
        has_emphasis_text = self._has_text_emphasis_fast(current_sub['text'])
        
        should_emphasize = has_dramatic_pause or has_long_duration or has_emphasis_text
        
        return should_emphasize
    
    def _has_text_emphasis_fast(self, text):
        """OPTIMIZED: Fast text emphasis check with early returns"""
        text = text.strip()
        
        # OPTIMIZED: Early returns for fastest checks first
        if len(text) <= 1:
            return False
            
        # Check punctuation (fastest)
        if text[-1] in '!?':
            return True
            
        if text.endswith('...'):
            return True
            
        # Check caps (fast)
        if text.isupper():
            return True
            
        # Check asterisks (fast)
        if text.startswith('*') and text.endswith('*'):
            return True
            
        # Check length (fast)
        if len(text) > 8:
            return True
            
        # OPTIMIZED: O(1) set lookup instead of O(n) list search
        if text.lower() in self._emphasis_words:
            return True
        
        return False

    def _create_segment_watermark(self, duration):
        """Create watermark for segment"""
        try:
            txt_clip = TextClip(
                WATERMARK_TEXT,
                fontsize=WATERMARK_FONTSIZE,
                color=WATERMARK_COLOR,
                font=WATERMARK_FONT
            ).set_duration(duration)
            
            # Get dimensions efficiently
            test_frame = txt_clip.get_frame(0)
            txt_height, txt_width = test_frame.shape[:2]
            
            # Create background
            bg_width = txt_width + (WATERMARK_PADDING * 2) + WATERMARK_MARGIN
            bg_height = txt_height + (WATERMARK_PADDING * 2) + WATERMARK_MARGIN
            
            if isinstance(WATERMARK_BG_COLOR, str):
                if WATERMARK_BG_COLOR.startswith('#'):
                    hex_color = WATERMARK_BG_COLOR[1:]
                    bg_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                else:
                    bg_color = (0, 0, 0)
            else:
                bg_color = WATERMARK_BG_COLOR
            
            bg_clip = ColorClip(size=(bg_width, bg_height), color=bg_color, duration=duration)
            txt_positioned = txt_clip.set_position((WATERMARK_PADDING, WATERMARK_PADDING))
            
            watermark_composite = CompositeVideoClip([bg_clip, txt_positioned], size=(bg_width, bg_height))
            
            watermark_x = TARGET_WIDTH - bg_width
            watermark_y = TARGET_HEIGHT - bg_height
            
            return watermark_composite.set_position((watermark_x, watermark_y))
            
        except Exception as e:
            print(f"Error creating watermark: {e}")
            return None

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
    
    for i in range(num_segments):
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
        preview_duration = min(30, max(float(entry['end_adjusted']) for entry in img_timestamps))
        
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
    """Use FFmpeg to add audio to the concatenated video"""
    cmd = ['ffmpeg', '-y', '-i', video_path]
    
    filter_complex = []
    audio_inputs = 1
    
    # Add main audio if available
    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-i', audio_path])
        audio_inputs += 1
        main_audio_index = audio_inputs - 1
        filter_complex.append(f"[{main_audio_index}:a]volume=1.0[main_audio]")
    
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
    
    # Output settings
    cmd.extend([
        '-c:v', 'copy',  # Copy video stream
        '-c:a', 'aac',   # Encode audio as AAC
        '-shortest',     # End when shortest stream ends
        output_path
    ])
    
    print(f"Adding audio to video...")
    
    # Create a simple progress indicator for FFmpeg
    print("Running FFmpeg audio processing...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg audio error: {result.stderr}")
        return False
    
    print(f"Successfully added audio: {output_path}")
    return True

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
