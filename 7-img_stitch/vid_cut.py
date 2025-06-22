import subprocess
import argparse
import os
import sys

# Default values
DEFAULT_START_TIME = 0
DEFAULT_DURATION = 60

def cut_video(input_file, output_file=None, start_time=DEFAULT_START_TIME, duration=DEFAULT_DURATION):
    """
    Cut video using ffmpeg
    
    Args:
        input_file (str): Path to input video file
        output_file (str): Path to output video file (optional)
        start_time (int/float): Start time in seconds
        duration (int/float): Duration in seconds
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_cut_{start_time}s-{start_time + duration}s{ext}"
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # Copy streams without re-encoding for speed
        '-avoid_negative_ts', 'make_zero',
        output_file,
        '-y'  # Overwrite output file if it exists
    ]
    
    try:
        print(f"Cutting video from {start_time}s to {start_time + duration}s...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Video cut successfully!")
            return True
        else:
            print(f"Error cutting video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Cut video using ffmpeg')
    parser.add_argument('input_file', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-s', '--start', type=float, default=DEFAULT_START_TIME, 
                       help=f'Start time in seconds (default: {DEFAULT_START_TIME})')
    parser.add_argument('-d', '--duration', type=float, default=DEFAULT_DURATION,
                       help=f'Duration in seconds (default: {DEFAULT_DURATION})')
    parser.add_argument('-e', '--end', type=float, help='End time in seconds (overrides duration)')
    
    args = parser.parse_args()
    
    # Calculate duration if end time is provided
    if args.end is not None:
        if args.end <= args.start:
            print("Error: End time must be greater than start time.")
            sys.exit(1)
        duration = args.end - args.start
    else:
        duration = args.duration
    
    # Cut the video
    success = cut_video(args.input_file, args.output, args.start, duration)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()