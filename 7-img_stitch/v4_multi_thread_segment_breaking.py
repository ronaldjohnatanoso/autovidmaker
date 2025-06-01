import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import sys
import json
import argparse
import time
import math
import random
import threading
import subprocess
import tempfile
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip, AudioFileClip, TextClip, CompositeAudioClip
from moviepy.audio.fx.all import audio_loop
import psutil

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
ENABLE_BREATHING_EFFECT = True  # Subtle scale pulsing
BREATHING_INTENSITY = 0.05  # How much to scale (5% up and down)
BREATHING_SPEED = 3.0  # Cycles per second (faster = more chaotic)

ENABLE_ROTATION_DRIFT = True  # Very subtle rotation
ROTATION_RANGE = 2.0  # Max rotation in degrees (increase for more chaos)

# New brain rot effects
ENABLE_SHAKE_EFFECT = True  # Random camera shake
SHAKE_INTENSITY = 8.0  # How much shake (pixels)
SHAKE_FREQUENCY = 12.0  # Shakes per second

ENABLE_ZOOM_PULSE = True  # Rapid zoom in/out
ZOOM_PULSE_INTENSITY = 0.1  # How much to zoom
ZOOM_PULSE_SPEED = 4.0  # Pulses per second

ENABLE_GLITCH_EFFECT = True  # Random glitch jumps
GLITCH_PROBABILITY = 0.01  # Chance per frame (10%)
GLITCH_INTENSITY = 20  # How far to jump (pixels)

ENABLE_RAINBOW_TEXT = False  # Cycling subtitle colors
RAINBOW_SPEED = 2.0  # Color cycles per second

class VideoSegmentProcessor:
    """Handles processing of individual video segments"""
    
    def __init__(self, project_name, base_dir, upscaled_images_dir, subtitles=None):
        self.project_name = project_name
        self.base_dir = base_dir
        self.upscaled_images_dir = upscaled_images_dir
        self.subtitles = subtitles or []
        
    def process_segment(self, segment_id, start_time, end_time, img_timestamps, temp_dir):
        """Process a single video segment"""
        print(f"Thread {threading.current_thread().name}: Processing segment {segment_id} ({start_time:.2f}s - {end_time:.2f}s)")
        
        try:
            # Filter images and subtitles for this segment
            segment_images = self._filter_images_for_segment(img_timestamps, start_time, end_time)
            segment_subtitles = self._filter_subtitles_for_segment(start_time, end_time)
            
            if not segment_images:
                print(f"No images found for segment {segment_id}")
                return None
            
            # Create image clips for this segment
            image_clips = []
            for entry in segment_images:
                clip = self._create_image_clip(entry)
                if clip:
                    # Adjust timing relative to segment start
                    clip_start = max(0, float(entry['start']) - start_time)
                    clip_end = min(end_time - start_time, float(entry['end_adjusted']) - start_time)
                    clip = clip.set_start(clip_start).set_end(clip_end)
                    image_clips.append(clip)
            
            # Create subtitle clips for this segment
            subtitle_clips = []
            for sub in segment_subtitles:
                clip = self._create_subtitle_clip(sub)
                if clip:
                    # Adjust timing relative to segment start
                    clip_start = max(0, sub['start'] - start_time)
                    clip_end = min(end_time - start_time, sub['end'] - start_time)
                    clip = clip.set_start(clip_start).set_end(clip_end)
                    subtitle_clips.append(clip)
            
            # Create watermark for this segment
            segment_duration = end_time - start_time
            watermark_clip = create_watermark_clip(segment_duration)
            
            # Compose segment
            all_clips = image_clips + subtitle_clips
            if watermark_clip:
                all_clips.append(watermark_clip)
            
            if not all_clips:
                print(f"No clips created for segment {segment_id}")
                return None
            
            segment_video = CompositeVideoClip(all_clips, size=(TARGET_WIDTH, TARGET_HEIGHT)).set_duration(segment_duration)
            
            # Render segment to temporary file
            segment_filename = f"segment_{segment_id:03d}.mp4"
            segment_path = os.path.join(temp_dir, segment_filename)
            
            print(f"Rendering segment {segment_id} to {segment_path}")
            
            # Try GPU encoding first, fallback to CPU if it fails
            encoding_configs = [
                {
                    'name': 'NVIDIA GPU',
                    'codec': 'h264_nvenc',
                    'ffmpeg_params': [
                        '-preset', 'fast',
                        '-rc', 'vbr',
                        '-cq', '23',
                        '-gpu', '0'
                    ]
                },
                {
                    'name': 'CPU (fast)',
                    'codec': 'libx264',
                    'ffmpeg_params': [
                        '-preset', 'fast',
                        '-crf', '23'
                    ]
                },
                {
                    'name': 'CPU (ultrafast)',
                    'codec': 'libx264',
                    'ffmpeg_params': [
                        '-preset', 'ultrafast',
                        '-crf', '28'
                    ]
                }
            ]
            
            success = False
            for config in encoding_configs:
                try:
                    print(f"Trying {config['name']} encoding for segment {segment_id}")
                    segment_video.write_videofile(
                        segment_path,
                        fps=FPS,
                        codec=config['codec'],
                        audio=False,  # We'll add audio later with FFmpeg
                        ffmpeg_params=config['ffmpeg_params'],
                        verbose=False,
                        logger=None
                    )
                    print(f"Segment {segment_id} completed with {config['name']}: {segment_path}")
                    success = True
                    break
                except Exception as e:
                    print(f"{config['name']} encoding failed for segment {segment_id}: {e}")
                    # Clean up partial file if it exists
                    if os.path.exists(segment_path):
                        os.unlink(segment_path)
                    continue
            
            if not success:
                print(f"All encoding methods failed for segment {segment_id}")
                return None
            
            return segment_path
            
        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            traceback.print_exc()
            return None
    
    def _filter_images_for_segment(self, img_timestamps, start_time, end_time):
        """Filter images that appear in the given time segment"""
        segment_images = []
        for entry in img_timestamps:
            img_start = float(entry['start'])
            img_end = float(entry['end_adjusted'])
            
            # Check if image overlaps with segment
            if img_start < end_time and img_end > start_time:
                segment_images.append(entry)
        
        return segment_images
    
    def _filter_subtitles_for_segment(self, start_time, end_time):
        """Filter subtitles that appear in the given time segment"""
        segment_subtitles = []
        for sub in self.subtitles:
            # Check if subtitle overlaps with segment
            if sub['start'] < end_time and sub['end'] > start_time:
                segment_subtitles.append(sub)
        
        return segment_subtitles
    
    def _create_image_clip(self, entry):
        """Create image clip with motion effects"""
        return create_fitted_image_clip_threaded(entry, self.upscaled_images_dir)
    
    def _create_subtitle_clip(self, sub):
        """Create subtitle clip with brain rot effects"""
        return create_subtitle_clip_threaded(sub)

def calculate_video_segments(img_timestamps, segment_duration=SEGMENT_DURATION):
    """Calculate video segments based on total duration"""
    if not img_timestamps:
        return []
    
    # Find total video duration
    max_end_time = max(float(entry['end_adjusted']) for entry in img_timestamps)
    
    # Calculate segments
    segments = []
    current_time = 0
    segment_id = 0
    
    while current_time < max_end_time:
        end_time = min(current_time + segment_duration, max_end_time)
        segments.append({
            'id': segment_id,
            'start': current_time,
            'end': end_time
        })
        current_time = end_time
        segment_id += 1
    
    return segments

def concat_video_segments(segment_paths, output_path):
    """Use FFmpeg to concatenate video segments"""
    if not segment_paths:
        raise ValueError("No segments to concatenate")
    
    # Create temporary file list for FFmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for segment_path in segment_paths:
            f.write(f"file '{os.path.abspath(segment_path)}'\n")
        filelist_path = f.name
    
    try:
        # FFmpeg concat command
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',  # Copy streams without re-encoding
            output_path
        ]
        
        print(f"Concatenating {len(segment_paths)} segments...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg concat error: {result.stderr}")
            return False
        
        print(f"Successfully concatenated video: {output_path}")
        return True
        
    finally:
        # Clean up temporary file list
        if os.path.exists(filelist_path):
            os.unlink(filelist_path)

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg audio error: {result.stderr}")
        return False
    
    print(f"Successfully added audio: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create video from images with timestamps')
    parser.add_argument('project_name', help='Name of the project')
    parser.add_argument('--render', action='store_true', help='Render the final video instead of preview')
    parser.add_argument('--segment-duration', type=int, default=SEGMENT_DURATION, 
                       help=f'Duration of each segment in seconds (default: {SEGMENT_DURATION})')
    
    args = parser.parse_args()
    project_name = args.project_name
    segment_duration = args.segment_duration

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

    # Load image timestamps
    with open(img_timestamps_file_path, 'r') as f:
        img_timestamps = json.load(f)

    # Load subtitles if available
    srt_path = os.path.join(base_dir, f"{project_name}_wordlevel.srt")
    subtitles = []
    if os.path.exists(srt_path):
        print(f"Loading subtitles from {srt_path}")
        subtitles = parse_srt(srt_path)

    # Calculate video segments
    segments = calculate_video_segments(img_timestamps, segment_duration)
    print(f"Video will be processed in {len(segments)} segments of {segment_duration}s each")

    if args.render:
        # Create temporary directory for segments
        temp_dir = tempfile.mkdtemp(prefix=f"{project_name}_segments_")
        print(f"Using temporary directory: {temp_dir}")
        
        try:
            # Create segment processor
            processor = VideoSegmentProcessor(project_name, base_dir, upscaled_images_dir, subtitles)
            
            # Process segments in parallel
            optimal_threads = min(get_optimal_thread_count(), len(segments))
            print(f"Processing {len(segments)} segments using {optimal_threads} threads...")
            
            segment_paths = []
            
            with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                # Submit all segment processing tasks
                future_to_segment = {
                    executor.submit(
                        processor.process_segment,
                        seg['id'], seg['start'], seg['end'], 
                        img_timestamps, temp_dir
                    ): seg for seg in segments
                }
                
                # Collect results as they complete
                completed_segments = {}
                for future in as_completed(future_to_segment):
                    segment = future_to_segment[future]
                    result = future.result()
                    if result:
                        completed_segments[segment['id']] = result
            
            # Sort segments by ID to maintain order
            for i in range(len(segments)):
                if i in completed_segments:
                    segment_paths.append(completed_segments[i])
            
            if not segment_paths:
                print("No segments were successfully created!")
                sys.exit(1)
            
            print(f"Successfully created {len(segment_paths)} video segments")
            
            # Concatenate video segments
            video_no_audio_path = os.path.join(base_dir, f"{project_name}_no_audio.mp4")
            if not concat_video_segments(segment_paths, video_no_audio_path):
                print("Failed to concatenate video segments!")
                sys.exit(1)
            
            # Add audio to the final video
            output_path = os.path.join(base_dir, f"{project_name}.mp4")
            audio_path = os.path.join(base_dir, f"{project_name}.wav")
            background_music_path = os.path.join(script_dir, "..", "common_assets", BACKGROUND_MUSIC_FILE)
            
            if not add_audio_with_ffmpeg(video_no_audio_path, audio_path, background_music_path, output_path):
                print("Failed to add audio to video!")
                sys.exit(1)
            
            # Clean up temporary video file
            if os.path.exists(video_no_audio_path):
                os.unlink(video_no_audio_path)
            
            print(f"Final video created: {output_path}")
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
    
    else:
        # Preview mode - create a short preview using existing logic
        print("Preview mode - creating short preview...")
        preview_segment = segments[0] if segments else {'start': 0, 'end': min(30, max(float(entry['end_adjusted']) for entry in img_timestamps))}
        
        processor = VideoSegmentProcessor(project_name, base_dir, upscaled_images_dir, subtitles)
        
        # Create a temporary preview
        with tempfile.TemporaryDirectory() as temp_dir:
            preview_path = processor.process_segment(
                0, preview_segment['start'], preview_segment['end'], 
                img_timestamps, temp_dir
            )
            
            if preview_path:
                # Use ffplay to preview the video
                subprocess.run(['ffplay', '-autoexit', preview_path])

def parse_srt(srt_path):
    """Parse SRT file and return list of subtitle entries"""
    subtitles = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    blocks = content.split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Parse timestamp line (format: 00:00:00,000 --> 00:00:00,000)
            timestamp_line = lines[1]
            start_str, end_str = timestamp_line.split(' --> ')
            
            # Convert timestamp to seconds
            def time_to_seconds(time_str):
                time_str = time_str.replace(',', '.')
                parts = time_str.split(':')
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            
            start_time = time_to_seconds(start_str)
            end_time = time_to_seconds(end_str)
            
            # Join text lines (in case subtitle spans multiple lines)
            text = ' '.join(lines[2:])
            
            subtitles.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })
    
    return subtitles

def get_optimal_thread_count():
    """Calculate optimal thread count based on system resources"""
    cpu_count = os.cpu_count()
    
    # Get current CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    # Conservative approach: use 75% of available cores, but adjust based on load
    if cpu_usage > 80 or memory_usage > 85:
        # System is busy, use fewer threads
        optimal_threads = max(2, cpu_count // 4)
    elif cpu_usage > 50 or memory_usage > 70:
        # Moderate load, use half the cores
        optimal_threads = max(4, cpu_count // 2)
    else:
        # Low load, use most cores but leave some headroom
        optimal_threads = max(4, int(cpu_count * 0.75))
    
    print(f"System info: {cpu_count} cores, {cpu_usage:.1f}% CPU, {memory_usage:.1f}% memory")
    print(f"Selected {optimal_threads} threads for processing")
    
    return optimal_threads

def create_fitted_image_clip_threaded(entry, upscaled_images_dir):
    """Thread-safe version of image clip creation with motion effects"""
    tag = entry['tag']
    start = float(entry['start'])
    end = float(entry['end_adjusted'])
    image_filename = f"{tag}.png"
    image_path = os.path.join(upscaled_images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Warning: Image '{image_path}' not found. Skipping.")
        return None

    try:
        duration = end - start
        img_clip = ImageClip(image_path).set_duration(duration)

        # Debug: Print original dimensions
        print(f"Processing {image_filename}: {img_clip.w}x{img_clip.h}, duration: {duration:.2f}s")

        # Ensure we have valid dimensions
        if img_clip.w <= 0 or img_clip.h <= 0:
            print(f"Error: Invalid image dimensions for {image_filename}: {img_clip.w}x{img_clip.h}")
            return None

        # Resize preserving aspect ratio to fit inside 1920x1080
        aspect_ratio = img_clip.w / img_clip.h
        target_aspect = TARGET_WIDTH / TARGET_HEIGHT

        if aspect_ratio < target_aspect:
            # Image is taller, fit to height
            new_height = TARGET_HEIGHT
            new_width = int(TARGET_HEIGHT * aspect_ratio)
        else:
            # Image is wider, fit to width
            new_width = TARGET_WIDTH
            new_height = int(TARGET_WIDTH / aspect_ratio)

        # Ensure minimum dimensions
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        print(f"Resizing {image_filename} to {new_width}x{new_height}")
        img_clip = img_clip.resize((new_width, new_height))

        # Apply motion effects if enabled (only for longer clips to avoid issues)
        if ENABLE_IMAGE_MOTION and duration > 0.5:
            img_clip = apply_motion_effect(img_clip, duration)

        # Black background clip
        bg = ColorClip(size=(TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0), duration=duration)

        # Composite with centered image on black background
        # Don't set start/end here - that will be handled in the segment processor
        final = CompositeVideoClip([bg, img_clip.set_position("center")])
        return final
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
        return None

def apply_motion_effect(img_clip, duration):
    """Apply various motion effects to image clips"""
    import random
    
    # Apply primary motion effect FIRST
    if MOTION_TYPE == 'ken_burns':
        img_clip = apply_ken_burns_effect(img_clip, duration)
    elif MOTION_TYPE == 'parallax':
        img_clip = apply_parallax_effect(img_clip, duration)
    elif MOTION_TYPE == 'zoom_pan':
        img_clip = apply_zoom_pan_effect(img_clip, duration)
    elif MOTION_TYPE == 'drift':
        img_clip = apply_drift_effect(img_clip, duration)
    
    # Apply brain rot effects (avoiding conflicting resize operations)
    if ENABLE_BREATHING_EFFECT and not ENABLE_ZOOM_PULSE:
        img_clip = apply_breathing_effect(img_clip, duration)
    
    if ENABLE_ROTATION_DRIFT:
        img_clip = apply_rotation_drift(img_clip, duration)
        
    if ENABLE_SHAKE_EFFECT:
        img_clip = apply_shake_effect(img_clip, duration)
        
    # Only apply ONE zoom-based effect to avoid conflicts
    if ENABLE_ZOOM_PULSE and not ENABLE_BREATHING_EFFECT:
        img_clip = apply_zoom_pulse_effect(img_clip, duration)
        
    if ENABLE_GLITCH_EFFECT:
        img_clip = apply_glitch_effect(img_clip, duration)
    
    return img_clip

def apply_zoom_pulse_effect(img_clip, duration):
    """Rapid zoom pulse effect"""
    import math
    
    def zoom_pulse_func(t):
        pulse = math.sin(t * ZOOM_PULSE_SPEED * 2 * math.pi)
        zoom_variation = pulse * ZOOM_PULSE_INTENSITY
        return 1.0 + zoom_variation
    
    # Don't chain with existing resize - apply directly
    return img_clip.resize(zoom_pulse_func)

def apply_breathing_effect(img_clip, duration):
    """Subtle breathing/pulsing scale effect"""
    import math
    
    def breathing_scale(t):
        # Create a sine wave for breathing effect
        cycle = math.sin(t * 2 * math.pi * BREATHING_SPEED)
        scale_variation = cycle * BREATHING_INTENSITY
        return 1.0 + scale_variation
    
    return img_clip.resize(breathing_scale)

def apply_rotation_drift(img_clip, duration):
    """Very subtle rotation drift"""
    import random
    import math
    
    max_rotation = ROTATION_RANGE
    rotation_direction = random.choice([-1, 1])
    
    def rotation_func(t):
        progress = t / duration
        # Smooth easing
        eased_progress = 0.5 * (1 - math.cos(progress * math.pi))
        return rotation_direction * max_rotation * eased_progress
    
    return img_clip.rotate(rotation_func)

def apply_ken_burns_effect(img_clip, duration):
    """Classic Ken Burns effect - slow zoom and pan"""
    import random
    
    # Random start and end zoom levels
    start_zoom = random.uniform(ZOOM_RANGE[0], ZOOM_RANGE[0] + 0.1)
    end_zoom = random.uniform(ZOOM_RANGE[1] - 0.1, ZOOM_RANGE[1])
    
    # Random pan direction
    start_x = random.uniform(-MOTION_INTENSITY, MOTION_INTENSITY)
    end_x = random.uniform(-MOTION_INTENSITY, MOTION_INTENSITY)
    start_y = random.uniform(-MOTION_INTENSITY, MOTION_INTENSITY)
    end_y = random.uniform(-MOTION_INTENSITY, MOTION_INTENSITY)
    
    def ken_burns_transform(t):
        progress = t / duration
        
        # Interpolate zoom
        zoom = start_zoom + (end_zoom - start_zoom) * progress
        
        # Interpolate position
        x_offset = (start_x + (end_x - start_x) * progress) * TARGET_WIDTH
        y_offset = (start_y + (end_y - start_y) * progress) * TARGET_HEIGHT
        
        return zoom, x_offset, y_offset
    
    def resize_func(t):
        zoom, _, _ = ken_burns_transform(t)
        return zoom
    
    def position_func(t):
        _, x_offset, y_offset = ken_burns_transform(t)
        return ('center', 'center')  # Keep centered for now, adjust if needed
    
    return img_clip.resize(resize_func)

def apply_parallax_effect(img_clip, duration):
    """Subtle horizontal parallax movement"""
    import math
    
    # Calculate movement range
    max_offset = MOTION_INTENSITY * TARGET_WIDTH
    
    def parallax_position(t):
        # Smooth sinusoidal movement
        progress = t / duration
        x_offset = math.sin(progress * math.pi * 2) * max_offset
        return (TARGET_WIDTH // 2 + x_offset, 'center')
    
    return img_clip.set_position(parallax_position)

def apply_zoom_pan_effect(img_clip, duration):
    """Slow zoom with subtle pan"""
    import random
    
    # Random zoom direction (in or out)
    zoom_in = random.choice([True, False])
    start_zoom = ZOOM_RANGE[1] if zoom_in else ZOOM_RANGE[0]
    end_zoom = ZOOM_RANGE[0] if zoom_in else ZOOM_RANGE[1]
    
    # Random pan direction
    pan_x = random.uniform(-MOTION_INTENSITY, MOTION_INTENSITY) * TARGET_WIDTH
    
    def zoom_pan_transform(t):
        progress = t / duration
        
        # Smooth zoom
        zoom = start_zoom + (end_zoom - start_zoom) * progress
        
        # Linear pan
        x_offset = pan_x * progress
        
        return zoom, x_offset
    
    def resize_func(t):
        zoom, _ = zoom_pan_transform(t)
        return zoom
    
    def position_func(t):
        _, x_offset = zoom_pan_transform(t)
        return (TARGET_WIDTH // 2 + x_offset, 'center')
    
    return img_clip.resize(resize_func).set_position(position_func)

def apply_drift_effect(img_clip, duration):
    """Gentle drifting movement in random direction"""
    import random
    import math
    
    # Random drift direction and speed
    drift_angle = random.uniform(0, 2 * math.pi)
    drift_distance = MOTION_INTENSITY * min(TARGET_WIDTH, TARGET_HEIGHT)
    
    def drift_position(t):
        progress = t / duration
        
        # Smooth easing function
        eased_progress = 0.5 * (1 - math.cos(progress * math.pi))
        
        x_offset = math.cos(drift_angle) * drift_distance * eased_progress
        y_offset = math.sin(drift_angle) * drift_distance * eased_progress
        
        return (TARGET_WIDTH // 2 + x_offset, TARGET_HEIGHT // 2 + y_offset)
    
    return img_clip.set_position(drift_position)

def apply_shake_effect(img_clip, duration):
    """Chaotic camera shake effect"""
    import random
    import math
    
    def shake_position(t):
        # High frequency random shake
        x_shake = random.uniform(-SHAKE_INTENSITY, SHAKE_INTENSITY) * math.sin(t * SHAKE_FREQUENCY * 2 * math.pi)
        y_shake = random.uniform(-SHAKE_INTENSITY, SHAKE_INTENSITY) * math.cos(t * SHAKE_FREQUENCY * 1.7 * math.pi)
        
        return (TARGET_WIDTH // 2 + x_shake, TARGET_HEIGHT // 2 + y_shake)
    
    return img_clip.set_position(shake_position)

def apply_glitch_effect(img_clip, duration):
    """Random glitch jumps"""
    import random
    
    def glitch_position(t):
        # Random chance for glitch
        if random.random() < GLITCH_PROBABILITY:
            x_glitch = random.uniform(-GLITCH_INTENSITY, GLITCH_INTENSITY)
            y_glitch = random.uniform(-GLITCH_INTENSITY, GLITCH_INTENSITY)
            return (TARGET_WIDTH // 2 + x_glitch, TARGET_HEIGHT // 2 + y_glitch)
        return ('center', 'center')
    
    return img_clip.set_position(glitch_position)

def create_subtitle_clip_threaded(sub):
    """Thread-safe version of subtitle creation with brain rot effects"""
    try:
        duration = sub['end'] - sub['start']
        
        # Rainbow text color if enabled
        if ENABLE_RAINBOW_TEXT:
            def get_rainbow_color(t):
                import math
                hue = (t * RAINBOW_SPEED) % 1.0
                # Convert HSV to RGB (simplified)
                r = int(255 * (1 + math.sin(hue * 6.28)) / 2)
                g = int(255 * (1 + math.sin(hue * 6.28 + 2.09)) / 2)
                b = int(255 * (1 + math.sin(hue * 6.28 + 4.18)) / 2)
                return f"#{r:02x}{g:02x}{b:02x}"
            
            # Create text clip with rainbow effect (simplified - use base color for now)
            txt_clip = TextClip(
                sub['text'],
                fontsize=SUBTITLE_FONTSIZE,
                color=SUBTITLE_COLOR,
                font=SUBTITLE_FONT,
                stroke_color=SUBTITLE_STROKE_COLOR,
                stroke_width=SUBTITLE_STROKE_WIDTH
            ).set_duration(duration).set_position(('center', SUBTITLE_POSITION_Y))
        else:
            txt_clip = TextClip(
                sub['text'],
                fontsize=SUBTITLE_FONTSIZE,
                color=SUBTITLE_COLOR,
                font=SUBTITLE_FONT,
                stroke_color=SUBTITLE_STROKE_COLOR,
                stroke_width=SUBTITLE_STROKE_WIDTH
            ).set_duration(duration).set_position(('center', SUBTITLE_POSITION_Y))
        
        # Remove fade effects - instant text transitions
        
        # Enhanced scale animation with overshoot
        if SCALE_EFFECT and duration > SCALE_DURATION:
            def brain_rot_scale_function(t):
                import math
                if t < SCALE_DURATION:
                    progress = t / SCALE_DURATION
                    # Bouncy overshoot effect
                    if progress < 0.6:
                        scale = 1.0 + (SCALE_FACTOR - 1.0) * (progress / 0.6)
                    else:
                        overshoot = 1.2  # Overshoot multiplier
                        bounce_progress = (progress - 0.6) / 0.4
                        scale = SCALE_FACTOR * overshoot - (SCALE_FACTOR * (overshoot - 1.0)) * bounce_progress
                    return scale
                return 1.0
            
            txt_clip = txt_clip.resize(brain_rot_scale_function)
        
        # Add shake to subtitles too
        if ENABLE_SHAKE_EFFECT:
            import random
            import math
            def subtitle_shake_position(t):
                base_y = SUBTITLE_POSITION_Y
                x_shake = random.uniform(-SHAKE_INTENSITY/2, SHAKE_INTENSITY/2)
                y_shake = random.uniform(-SHAKE_INTENSITY/2, SHAKE_INTENSITY/2)
                return ('center', base_y + y_shake)
            
            txt_clip = txt_clip.set_position(subtitle_shake_position)
        
        # Set timing
        txt_clip = txt_clip.set_start(sub['start']).set_end(sub['end'])
        return txt_clip
    except Exception as e:
        print(f"Error creating subtitle '{sub['text']}': {e}")
        return None

def create_watermark_clip(duration):
    """Create a watermark text clip with solid background that extends to screen edges"""
    try:
        # Create the text clip with explicit RGB format
        txt_clip = TextClip(
            WATERMARK_TEXT,
            fontsize=WATERMARK_FONTSIZE,
            color=WATERMARK_COLOR,
            font=WATERMARK_FONT
        ).set_duration(duration)
        
        # Ensure text clip has proper RGB channels
        if hasattr(txt_clip, 'mask'):
            txt_clip = txt_clip.set_mask(None)  # Remove any mask issues
        
        # Get text dimensions - force evaluation to get actual size
        test_frame = txt_clip.get_frame(0)
        txt_height, txt_width = test_frame.shape[:2]
        
        # Create background that extends to screen edges
        # Width: from text start to right edge of screen
        # Height: from text start to bottom edge of screen
        bg_width = txt_width + (WATERMARK_PADDING * 2) + WATERMARK_MARGIN
        bg_height = txt_height + (WATERMARK_PADDING * 2) + WATERMARK_MARGIN
        
        # Ensure background is RGB (3 channels)
        if isinstance(WATERMARK_BG_COLOR, str):
            # Convert hex color to RGB tuple
            if WATERMARK_BG_COLOR.startswith('#'):
                hex_color = WATERMARK_BG_COLOR[1:]
                bg_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            else:
                bg_color = (0, 0, 0)  # Default to black
        else:
            bg_color = WATERMARK_BG_COLOR
        
        bg_clip = ColorClip(
            size=(bg_width, bg_height),
            color=bg_color,
            duration=duration
        )
        
        # Position text with padding from the edges of the background
        txt_positioned = txt_clip.set_position((WATERMARK_PADDING, WATERMARK_PADDING))
        
        # Composite text on background
        watermark_composite = CompositeVideoClip([bg_clip, txt_positioned], size=(bg_width, bg_height))
        
        # Position watermark so it touches the right and bottom edges
        watermark_x = TARGET_WIDTH - bg_width
        watermark_y = TARGET_HEIGHT - bg_height
        
        return watermark_composite.set_position((watermark_x, watermark_y))
        
    except Exception as e:
        print(f"Error creating watermark: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
