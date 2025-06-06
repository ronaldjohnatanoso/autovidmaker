import subprocess
import os

def create_tv_distortion_effect(input_file="input.mp4", output_file="tv_distortion.mp4", effect_type="classic"):
    """Create classic TV distortion effects"""
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return False
    
    print(f"Creating TV distortion effect: {effect_type}")
    
    if effect_type == "classic":
        # Classic TV interference with moving scanlines
        filter_str = "noise=alls=50:allf=t,rgbashift=rh=5:gh=-3:bh=2"
        
    elif effect_type == "glitch":
        # Glitch effect with RGB shift and noise
        filter_str = "noise=alls=30:allf=t+u,rgbashift=rh=10:gh=-8:bh=6"
        
    elif effect_type == "scanlines":
        # Moving scanlines effect
        filter_str = "noise=alls=20,geq=lum='lum(X,Y)+20*sin(6.28*Y/4)'"
        
    elif effect_type == "interference":
        # TV interference pattern
        filter_str = "noise=alls=40:allf=t,geq=r='r(X,Y)+30*sin(6.28*Y/8)':g='g(X,Y)+25*sin(6.28*Y/6)':b='b(X,Y)+20*sin(6.28*Y/10)'"
        
    elif effect_type == "heavy_glitch":
        # Heavy glitch with displacement
        filter_str = "noise=alls=60:allf=t+u,rgbashift=rh=20:gh=-15:bh=12,geq=r='r(X+(Y%50>25?15:0),Y)':g='g(X,Y)':b='b(X-(Y%40>20?10:0),Y)'"
        
    else:  # "analog_tv"
        # Analog TV distortion
        filter_str = "noise=alls=35:allf=t,geq=r='r(X,Y)+40*sin(6.28*Y/3)':g='g(X,Y)+35*sin(6.28*Y/4)':b='b(X,Y)+30*sin(6.28*Y/5)',rgbashift=rh=3:gh=-2:bh=4"
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-vf', filter_str,
        '-c:a', 'copy',
        output_file
    ]
    
    print(f"Filter: {filter_str}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ TV distortion effect created: {output_file}")
        return True
    else:
        print(f"✗ Failed to create TV distortion")
        print(f"Error: {result.stderr}")
        return False

def create_multiple_tv_effects(input_file="input.mp4"):
    """Create multiple TV distortion effects to test"""
    
    effects = [
        ("classic", "Classic TV interference"),
        ("glitch", "Digital glitch effect"), 
        ("scanlines", "Moving scanlines"),
        ("interference", "TV interference pattern"),
        ("heavy_glitch", "Heavy glitch distortion"),
        ("analog_tv", "Analog TV distortion")
    ]
    
    print("Creating TV distortion effects...")
    print("=" * 50)
    
    working_effects = []
    
    for effect_type, description in effects:
        output_file = f"tv_{effect_type}.mp4"
        print(f"Creating {description}...")
        
        if create_tv_distortion_effect(input_file, output_file, effect_type):
            working_effects.append((effect_type, output_file))
        
        print()
    
    print("=" * 50)
    print("CREATED TV EFFECTS:")
    for effect_type, output_file in working_effects:
        print(f"✓ {output_file} - {dict(effects)[effect_type]}")
    
    return working_effects

def create_animated_tv_glitch(input_file="input.mp4", output_file="animated_tv_glitch.mp4"):
    """Create animated TV glitch by varying the distortion intensity"""
    
    print("Creating animated TV glitch effect...")
    
    # Create multiple intensity levels
    temp_files = []
    intensities = [20, 40, 60, 80, 60, 40]  # Pulsing intensity
    
    for i, intensity in enumerate(intensities):
        temp_file = f"temp_glitch_{i}.mp4"
        temp_files.append(temp_file)
        
        # Vary the noise and RGB shift intensity
        filter_str = f"noise=alls={intensity}:allf=t+u,rgbashift=rh={intensity//4}:gh=-{intensity//5}:bh={intensity//6}"
        
        cmd = [
            'ffmpeg', '-y', '-i', input_file,
            '-vf', filter_str,
            '-t', '1',  # 1 second per intensity
            '-c:a', 'copy',
            temp_file
        ]
        
        print(f"  Creating intensity level {i+1}/{len(intensities)}")
        subprocess.run(cmd, capture_output=True, text=True)
    
    # Concatenate all intensity levels
    concat_file = "glitch_concat.txt"
    with open(concat_file, 'w') as f:
        for temp_file in temp_files:
            f.write(f"file '{temp_file}'\n")
    
    # Create final animated glitch
    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c', 'copy', output_file
    ]
    
    result = subprocess.run(concat_cmd, capture_output=True, text=True)
    
    # Clean up
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    if os.path.exists(concat_file):
        os.unlink(concat_file)
    
    if result.returncode == 0:
        print(f"✓ Animated TV glitch created: {output_file}")
        return True
    else:
        print(f"✗ Failed to create animated glitch")
        return False

if __name__ == "__main__":
    # Create all TV distortion effects
    working_effects = create_multiple_tv_effects()
    
    # Create animated version
    create_animated_tv_glitch()
    
    print("\n" + "=" * 60)
    print("TV DISTORTION EFFECTS READY!")
    print("=" * 60)
    print("Play the generated files to see classic TV interference effects")
    print("These include noise, scanlines, RGB shift, and glitch distortions")