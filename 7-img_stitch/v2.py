import os
import sys
import json
import argparse
from moviepy.editor import ImageClip, CompositeVideoClip, ColorClip, AudioFileClip, TextClip

# Constants
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 24

# Subtitle styling constants
SUBTITLE_FONTSIZE = 80
SUBTITLE_COLOR = '#90EE90'  # Light green
SUBTITLE_FONT = 'Komika-Axis'  # Custom thick font
SUBTITLE_STROKE_COLOR = 'white'
SUBTITLE_STROKE_WIDTH = 5
SUBTITLE_POS_Y_RATIO = 0.7 # 1 = bottom of the screen, 0 = top of the screen
SUBTITLE_POSITION_Y = int(TARGET_HEIGHT * SUBTITLE_POS_Y_RATIO)  # Pixel position from top (540 = center of 1080p screen)

# Text effect constants
FADE_IN_DURATION = 0.3  # Fade in time in seconds
FADE_OUT_DURATION = 0.3  # Fade out time in seconds
SCALE_EFFECT = True  # Enable scale animation
SCALE_FACTOR = 1.2  # How much to scale up (1.2 = 20% bigger)
SCALE_DURATION = 0.5  # Duration of scale effect

def create_fitted_image_clip(image_path, start, end):
    duration = end - start
    img_clip = ImageClip(image_path).set_duration(duration)

    # Resize preserving aspect ratio to fit inside 1920x1080
    if img_clip.w / img_clip.h < TARGET_WIDTH / TARGET_HEIGHT:
        img_clip = img_clip.resize(height=TARGET_HEIGHT)
    else:
        img_clip = img_clip.resize(width=TARGET_WIDTH)

    # Black background clip
    bg = ColorClip(size=(TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0), duration=duration)

    # Composite with centered image on black background
    final = CompositeVideoClip([bg, img_clip.set_position("center")]).set_start(start).set_end(end)
    return final

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

def create_subtitle_clips(subtitles):
    """Create text clips with fancy effects"""
    subtitle_clips = []
    
    for sub in subtitles:
        duration = sub['end'] - sub['start']
        
        # Create text clip with enhanced styling
        txt_clip = TextClip(
            sub['text'],
            fontsize=SUBTITLE_FONTSIZE,
            color=SUBTITLE_COLOR,
            font=SUBTITLE_FONT,
            stroke_color=SUBTITLE_STROKE_COLOR,
            stroke_width=SUBTITLE_STROKE_WIDTH
        ).set_duration(duration).set_position(('center', SUBTITLE_POSITION_Y))
        
        # Add fade in/out effects
        if duration > FADE_IN_DURATION + FADE_OUT_DURATION:
            txt_clip = txt_clip.fadein(FADE_IN_DURATION).fadeout(FADE_OUT_DURATION)
        
        # Add scale animation effect
        if SCALE_EFFECT and duration > SCALE_DURATION:
            def scale_function(t):
                if t < SCALE_DURATION:
                    # Scale up from 1.0 to SCALE_FACTOR and back to 1.0
                    progress = t / SCALE_DURATION
                    if progress < 0.5:
                        # Scale up
                        scale = 1.0 + (SCALE_FACTOR - 1.0) * (progress * 2)
                    else:
                        # Scale down
                        scale = SCALE_FACTOR - (SCALE_FACTOR - 1.0) * ((progress - 0.5) * 2)
                    return scale
                return 1.0
            
            txt_clip = txt_clip.resize(scale_function)
        
        # Set timing
        txt_clip = txt_clip.set_start(sub['start']).set_end(sub['end'])
        
        subtitle_clips.append(txt_clip)
    
    return subtitle_clips

def main():
    parser = argparse.ArgumentParser(description='Create video from images with timestamps')
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

    # Check for audio file
    audio_path = os.path.join(base_dir, "ocean_debate.wav")
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file '{audio_path}' not found. Video will have no audio.")
        audio_clip = None
    else:
        audio_clip = AudioFileClip(audio_path)

    # Check for SRT subtitle file
    srt_path = os.path.join(base_dir, f"{project_name}_wordlevel.srt")
    if not os.path.exists(srt_path):
        print(f"Warning: SRT file '{srt_path}' not found. Video will have no subtitles.")
        subtitle_clips = []
    else:
        print(f"Loading subtitles from {srt_path}")
        subtitles = parse_srt(srt_path)
        subtitle_clips = create_subtitle_clips(subtitles)

    with open(img_timestamps_file_path, 'r') as f:
        img_timestamps = json.load(f)

    clips = []

    for entry in img_timestamps:
        tag = entry['tag']
        start = float(entry['start'])
        end = float(entry['end_adjusted'])
        image_filename = f"{tag}.png"
        image_path = os.path.join(upscaled_images_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image '{image_path}' not found. Skipping.")
            continue

        clip = create_fitted_image_clip(image_path, start, end)
        clips.append(clip)

    if not clips:
        print("No valid image clips found. Exiting.")
        sys.exit(1)

    # Compose all clips in timeline, size fixed to 16:9
    all_clips = clips + subtitle_clips
    final_video = CompositeVideoClip(all_clips, size=(TARGET_WIDTH, TARGET_HEIGHT)).set_duration(clips[-1].end)
    
    # Add audio if available
    if audio_clip:
        final_video = final_video.set_audio(audio_clip)

    if args.render:
        output_path = os.path.join(base_dir, f"{project_name}.mp4")
        print(f"Rendering final video to {output_path} ...")
        final_video.write_videofile(output_path, fps=FPS, codec='libx264', audio_codec='aac' if audio_clip else None)
    else:
        print("Launching preview...")
        final_video.preview(fps=FPS)

if __name__ == "__main__":
    main()
