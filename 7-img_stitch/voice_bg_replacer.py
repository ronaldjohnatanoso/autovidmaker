import os
import sys
import argparse
import subprocess
import json

def get_config_background_music(project_folder):
    """Get background music file from project config"""
    config_path = os.path.join(project_folder, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("stages", {}).get("img_stitch", {}).get("background_music_file", "curious.mp3")
        except:
            pass
    return "curious.mp3"

def replace_video_audio(video_path, narration_path, background_music_file, output_path, narration_volume=2.0, bg_volume=0.4):
    """Replace video audio with narration and optional background music"""
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct background music path
    background_music_path = os.path.join(script_dir, "..", "common_assets", background_music_file)
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-y', '-i', video_path]
    
    filter_complex = []
    audio_inputs = 1
    
    # Add narration audio if available
    if narration_path and os.path.exists(narration_path):
        cmd.extend(['-i', narration_path])
        audio_inputs += 1
        narration_index = audio_inputs - 1
        filter_complex.append(f"[{narration_index}:a]volume={narration_volume}[narration_audio]")
    
    # Add background music if available and file exists
    if background_music_file and os.path.exists(background_music_path):
        cmd.extend(['-i', background_music_path])
        audio_inputs += 1
        bg_music_index = audio_inputs - 1
        
        # Loop background music and set volume with fade transitions
        if filter_complex:
            # Mix narration + background music with crossfade loop
            filter_complex.append(f"[{bg_music_index}:a]aloop=loop=-1:size=2e+09,volume={bg_volume}[bg_music]")
            filter_complex.append("[narration_audio][bg_music]amix=inputs=2:duration=first:dropout_transition=2[audio_out]")
            audio_map = "[audio_out]"
        else:
            # Only background music with fade loop
            filter_complex.append(f"[{bg_music_index}:a]aloop=loop=-1:size=2e+09,volume={bg_volume}[audio_out]")
            audio_map = "[audio_out]"
    else:
        # Only narration audio
        audio_map = "[narration_audio]" if filter_complex else None
    
    # Add filter complex if we have audio processing
    if filter_complex:
        cmd.extend(['-filter_complex', ';'.join(filter_complex)])
        cmd.extend(['-map', '0:v', '-map', audio_map])
    elif narration_path and os.path.exists(narration_path):
        # Simple case: just copy narration audio
        cmd.extend(['-map', '0:v', '-map', '1:a'])
    else:
        # No audio - strip all audio
        cmd.extend(['-map', '0:v'])
    
    # Output settings
    cmd.extend([
        '-c:v', 'copy',  # Copy video stream without re-encoding
        '-c:a', 'aac',   # Encode audio as AAC
        '-shortest',     # End when shortest stream ends
        output_path
    ])
    
    print(f"Replacing video audio...")
    print(f"Video: {video_path}")
    print(f"Narration: {narration_path if narration_path and os.path.exists(narration_path) else 'None'} (volume: {narration_volume})")
    print(f"Background Music: {background_music_file if background_music_file and os.path.exists(background_music_path) else 'None'} (volume: {bg_volume})")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False
    
    print(f"Successfully replaced audio: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Replace video audio with narration and optional background music")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("--bg-music", help="Background music file name (from common_assets folder)")
    parser.add_argument("--no-bg", action="store_true", help="Don't add background music")
    parser.add_argument("--narration-volume", type=float, default=2.0, help="Narration volume multiplier (default: 2.0)")
    parser.add_argument("--bg-volume", type=float, default=0.4, help="Background music volume multiplier (default: 0.4)")
    
    args = parser.parse_args()
    
    # Get the base directory (autovidmaker)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # Project folder path
    project_folder = os.path.join(base_dir, "0-project-files", args.project_name)
    if not os.path.exists(project_folder):
        print(f"Error: Project folder not found: {project_folder}")
        sys.exit(1)
    
    # Input video path
    video_path = os.path.join(project_folder, f"{args.project_name}_final.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Narration audio path
    narration_path = os.path.join(project_folder, f"{args.project_name}.wav")
    
    # Determine background music
    if args.no_bg:
        background_music_file = None
    elif args.bg_music:
        background_music_file = args.bg_music
    else:
        # Get from config
        background_music_file = get_config_background_music(project_folder)
    
    # Output path
    output_path = os.path.join(project_folder, f"{args.project_name}_audio_replaced.mp4")
    
    # Verify narration exists
    if not os.path.exists(narration_path):
        print(f"Warning: Narration file not found: {narration_path}")
        print("Proceeding without narration audio...")
        narration_path = None
    
    # Replace audio
    if replace_video_audio(video_path, narration_path, background_music_file, output_path, 
                          narration_volume=args.narration_volume, bg_volume=args.bg_volume):
        print(f"\nSuccess! Output saved to: {output_path}")
    else:
        print("Failed to replace audio!")
        sys.exit(1)

if __name__ == "__main__":
    main()