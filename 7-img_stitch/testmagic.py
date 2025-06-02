from moviepy.editor import *
from moviepy.video.fx import resize
import numpy as np

# Create a test image using ImageMagick through MoviePy
def create_test_image_with_imagemagick():
    # Create a simple colored clip
    clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=1)
    
    # Use ImageMagick to add text overlay
    txt_clip = TextClip("Hello ImageMagick!", 
                       fontsize=50, 
                       color='white',
                       font='Arial-Bold')
    
    # Position the text in the center
    txt_clip = txt_clip.set_position('center').set_duration(1)
    
    # Composite the text over the colored background
    final_clip = CompositeVideoClip([clip, txt_clip])
    
    # Save as image (first frame)
    final_clip.save_frame("test_imagemagick_output.png", t=0)
    
    print("Test image created: test_imagemagick_output.png")

# Create an image with ImageMagick effects
def create_image_with_effects():
    # Load or create an image
    clip = ColorClip(size=(400, 300), color=(0, 100, 200), duration=1)
    
    # Add multiple text elements
    title = TextClip("MoviePy + ImageMagick", 
                    fontsize=30, 
                    color='yellow',
                    font='Arial-Bold').set_position(('center', 50)).set_duration(1)
    
    subtitle = TextClip("Test Image", 
                       fontsize=20, 
                       color='white',
                       font='Arial').set_position(('center', 200)).set_duration(1)
    
    # Composite all elements
    final = CompositeVideoClip([clip, title, subtitle])
    
    # Apply resize effect
    final = final.resize(height=400)
    
    # Save the frame
    final.save_frame("test_effects_output.png", t=0)
    print("Effects image created: test_effects_output.png")

# Advanced ImageMagick text effects
def create_advanced_text_effects():
    # Create gradient background
    clip = ColorClip(size=(800, 600), color=(50, 50, 100), duration=1)
    
    # Text with stroke/outline
    main_text = TextClip("EPIC TEXT", 
                        fontsize=80, 
                        color='white',
                        font='Arial-Bold',
                        stroke_color='black',
                        stroke_width=3).set_position('center').set_duration(1)
    
    # Shadow text (positioned slightly offset)
    shadow_text = TextClip("EPIC TEXT", 
                          fontsize=80, 
                          color='gray',
                          font='Arial-Bold').set_position((402, 302)).set_duration(1)
    
    # Subtitle with different styling
    subtitle = TextClip("With Cool Effects", 
                       fontsize=30, 
                       color='cyan',
                       font='Arial',
                       interline=-5).set_position(('center', 400)).set_duration(1)
    
    # Composite with shadow first, then main text
    final = CompositeVideoClip([clip, shadow_text, main_text, subtitle])
    final.save_frame("advanced_text_effects.png", t=0)
    print("Advanced text effects image created: advanced_text_effects.png")

# Create image with rotated and scaled elements
def create_complex_composition():
    # Base background
    bg = ColorClip(size=(1000, 800), color=(20, 20, 40), duration=1)
    
    # Multiple text elements with different positions and rotations
    elements = []
    
    # Main title
    title = TextClip("COMPLEX\nCOMPOSITION", 
                    fontsize=60, 
                    color='orange',
                    font='Arial-Bold',
                    method='caption',
                    size=(400, None)).set_position((50, 100)).set_duration(1)
    
    # Rotated side text
    side_text = TextClip("ROTATED TEXT", 
                        fontsize=40, 
                        color='lime',
                        font='Arial-Bold').set_position((700, 200)).set_duration(1).rotate(45)
    
    # Small info texts
    for i, text in enumerate(["INFO 1", "INFO 2", "INFO 3"]):
        info = TextClip(text, 
                       fontsize=25, 
                       color='white',
                       font='Arial').set_position((100 + i*150, 600)).set_duration(1)
        elements.append(info)
    
    # Add all elements
    elements = [bg, title, side_text] + elements
    final = CompositeVideoClip(elements)
    
    final.save_frame("complex_composition.png", t=0)
    print("Complex composition image created: complex_composition.png")

# Create a collage-style image
def create_text_collage():
    # Large canvas
    canvas = ColorClip(size=(1200, 900), color=(10, 10, 20), duration=1)
    
    texts = [
        ("MOVIEPY", 70, 'red', (100, 50)),
        ("IMAGEMAGICK", 50, 'blue', (400, 150)),
        ("PYTHON", 60, 'green', (200, 300)),
        ("VIDEO", 45, 'yellow', (600, 400)),
        ("PROCESSING", 40, 'magenta', (150, 500)),
        ("AUTOMATION", 35, 'cyan', (700, 600)),
        ("CREATIVITY", 55, 'orange', (300, 700))
    ]
    
    clips = [canvas]
    
    for text, size, color, pos in texts:
        txt = TextClip(text, 
                      fontsize=size, 
                      color=color,
                      font='Arial-Bold').set_position(pos).set_duration(1)
        clips.append(txt)
    
    final = CompositeVideoClip(clips)
    final.save_frame("text_collage.png", t=0)
    print("Text collage image created: text_collage.png")

# Test different fonts and styles
def test_font_variations():
    bg = ColorClip(size=(800, 1000), color=(30, 30, 50), duration=1)
    
    font_tests = [
        ("Arial", "Arial Font Test", 40, 'white', 100),
        ("Arial-Bold", "Arial Bold Test", 40, 'yellow', 180),
        ("Times-Roman", "Times Roman Test", 40, 'lightblue', 260),
        ("Courier", "Courier Font Test", 35, 'lightgreen', 340),
        ("Helvetica", "Helvetica Test", 40, 'pink', 420)
    ]
    
    clips = [bg]
    
    for font, text, size, color, y_pos in font_tests:
        try:
            txt = TextClip(text, 
                          fontsize=size, 
                          color=color,
                          font=font).set_position(('center', y_pos)).set_duration(1)
            clips.append(txt)
        except:
            # Fallback to Arial if font not available
            txt = TextClip(f"{text} (Arial fallback)", 
                          fontsize=size, 
                          color='gray',
                          font='Arial').set_position(('center', y_pos)).set_duration(1)
            clips.append(txt)
    
    final = CompositeVideoClip(clips)
    final.save_frame("font_variations.png", t=0)
    print("Font variations image created: font_variations.png")

# Create a basic test video with ImageMagick text
def create_test_video_with_imagemagick():
    # Create a colored background that changes over time
    clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=5)
    
    # Add animated text overlay
    txt_clip = TextClip("Hello ImageMagick Video!", 
                       fontsize=50, 
                       color='white',
                       font='Arial-Bold')
    
    # Position and animate the text
    txt_clip = txt_clip.set_position('center').set_duration(5)
    
    # Composite the text over the colored background
    final_clip = CompositeVideoClip([clip, txt_clip])
    
    # Write the video
    final_clip.write_videofile("test_imagemagick_video.mp4", fps=24)
    print("Test video created: test_imagemagick_video.mp4")

# Create video with moving text effects
def create_animated_text_video():
    # Create gradient background
    bg = ColorClip(size=(800, 600), color=(50, 50, 100), duration=8)
    
    # Moving title that slides in from left
    title = TextClip("ANIMATED TEXT", 
                    fontsize=60, 
                    color='yellow',
                    font='Arial-Bold',
                    stroke_color='black',
                    stroke_width=2)
    
    title = title.set_position(lambda t: (max(-300, -300 + 100*t), 200)).set_duration(8)
    
    # Subtitle that fades in
    subtitle = TextClip("With Cool Effects", 
                       fontsize=30, 
                       color='cyan',
                       font='Arial')
    
    subtitle = subtitle.set_position('center').set_duration(6).set_start(2).fadeout(1)
    
    # Rotating text element
    rotating_text = TextClip("SPIN!", 
                            fontsize=40, 
                            color='orange',
                            font='Arial-Bold')
    
    rotating_text = rotating_text.set_position((600, 100)).set_duration(8).rotate(lambda t: 45*t)
    
    # Composite all elements
    final = CompositeVideoClip([bg, title, subtitle, rotating_text])
    final.write_videofile("animated_text_video.mp4", fps=24)
    print("Animated text video created: animated_text_video.mp4")

# Create video with scaling and color effects
def create_effects_showcase_video():
    # Base background
    bg = ColorClip(size=(1000, 800), color=(20, 20, 40), duration=10)
    
    # Text that grows and shrinks
    growing_text = TextClip("GROWING TEXT", 
                           fontsize=50, 
                           color='lime',
                           font='Arial-Bold')
    
    growing_text = (growing_text.set_position('center')
                   .set_duration(10)
                   .resize(lambda t: 0.5 + 0.5*np.sin(t)))
    
    # Text that changes position in a circle
    circular_text = TextClip("CIRCULAR", 
                            fontsize=30, 
                            color='magenta',
                            font='Arial-Bold')
    
    def circular_position(t):
        x = 500 + 200 * np.cos(t)
        y = 400 + 200 * np.sin(t)
        return (x, y)
    
    circular_text = circular_text.set_position(circular_position).set_duration(10)
    
    # Pulsing text with opacity changes
    pulsing_text = TextClip("PULSING", 
                           fontsize=40, 
                           color='red',
                           font='Arial-Bold')
    
    pulsing_text = (pulsing_text.set_position((100, 100))
                   .set_duration(10)
                   .set_opacity(lambda t: 0.3 + 0.7*abs(np.sin(2*t))))
    
    # Composite all effects
    final = CompositeVideoClip([bg, growing_text, circular_text, pulsing_text])
    final.write_videofile("effects_showcase_video.mp4", fps=24)
    print("Effects showcase video created: effects_showcase_video.mp4")

# Create a multi-scene video with transitions
def create_multi_scene_video():
    scenes = []
    
    # Scene 1: Title introduction
    bg1 = ColorClip(size=(800, 600), color=(100, 0, 100), duration=3)
    title1 = TextClip("SCENE 1", 
                     fontsize=80, 
                     color='white',
                     font='Arial-Bold').set_position('center').set_duration(3)
    scene1 = CompositeVideoClip([bg1, title1]).fadein(0.5).fadeout(0.5)
    scenes.append(scene1)
    
    # Scene 2: Multi-text animation
    bg2 = ColorClip(size=(800, 600), color=(0, 100, 100), duration=4)
    
    texts = ["MULTIPLE", "ANIMATED", "TEXTS"]
    text_clips = []
    
    for i, text in enumerate(texts):
        txt = TextClip(text, 
                      fontsize=50, 
                      color='yellow',
                      font='Arial-Bold')
        txt = txt.set_position(('center', 150 + i*100)).set_start(i*0.5).set_duration(4-i*0.5)
        text_clips.append(txt)
    
    scene2 = CompositeVideoClip([bg2] + text_clips).fadein(0.5).fadeout(0.5)
    scenes.append(scene2)
    
    # Scene 3: Final credits
    bg3 = ColorClip(size=(800, 600), color=(100, 100, 0), duration=3)
    credits = TextClip("THE END\nMoviePy + ImageMagick", 
                      fontsize=40, 
                      color='black',
                      font='Arial-Bold',
                      method='caption').set_position('center').set_duration(3)
    scene3 = CompositeVideoClip([bg3, credits]).fadein(0.5).fadeout(0.5)
    scenes.append(scene3)
    
    # Concatenate all scenes
    final_video = concatenate_videoclips(scenes)
    final_video.write_videofile("multi_scene_video.mp4", fps=24)
    print("Multi-scene video created: multi_scene_video.mp4")

# Create a video with complex text animations
def create_complex_animation_video():
    # Background
    bg = ColorClip(size=(1200, 900), color=(10, 10, 30), duration=12)
    
    # Flying text from different directions
    fly_texts = [
        ("FLYING", 'red', (-200, 100), (1200, 100), 0, 3),
        ("TEXT", 'green', (1200, 300), (-200, 300), 2, 3),
        ("EFFECTS", 'blue', (600, -100), (600, 900), 4, 3),
        ("DEMO", 'orange', (600, 900), (600, -100), 6, 3)
    ]
    
    text_clips = []
    
    for text, color, start_pos, end_pos, start_time, duration in fly_texts:
        txt = TextClip(text, 
                      fontsize=60, 
                      color=color,
                      font='Arial-Bold',
                      stroke_color='white',
                      stroke_width=2)
        
        # Animate position from start to end
        def make_position_func(start_pos, end_pos, start_time, duration):
            def position_func(t):
                if t < start_time:
                    return start_pos
                elif t > start_time + duration:
                    return end_pos
                else:
                    progress = (t - start_time) / duration
                    x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                    y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                    return (x, y)
            return position_func
        
        txt = txt.set_position(make_position_func(start_pos, end_pos, start_time, duration)).set_duration(12)
        text_clips.append(txt)
    
    # Center text that appears and rotates
    center_text = TextClip("CENTER\nSTAGE", 
                          fontsize=80, 
                          color='yellow',
                          font='Arial-Bold',
                          method='caption')
    
    center_text = (center_text.set_position('center')
                  .set_start(8)
                  .set_duration(4)
                  .rotate(lambda t: 360*(t-8)/4 if t > 8 else 0)
                  .fadein(1)
                  .fadeout(1))
    
    # Composite everything
    all_clips = [bg] + text_clips + [center_text]
    final = CompositeVideoClip(all_clips)
    final.write_videofile("complex_animation_video.mp4", fps=24)
    print("Complex animation video created: complex_animation_video.mp4")

# Create a video with different text styles showcase
def create_styles_showcase_video():
    bg = ColorClip(size=(800, 600), color=(25, 25, 25), duration=15)
    
    # Different text styles appearing one by one
    styles = [
        ("BOLD", 'Arial-Bold', 'white', 1),
        ("ITALIC", 'Arial-Italic', 'cyan', 3),
        ("OUTLINED", 'Arial-Bold', 'red', 5),
        ("SHADOWED", 'Arial-Bold', 'yellow', 7),
        ("COLORFUL", 'Arial-Bold', 'lime', 9),
        ("FINAL", 'Arial-Bold', 'magenta', 11)
    ]
    
    text_clips = []
    
    for i, (text, font, color, start_time) in enumerate(styles):
        txt = TextClip(text, 
                      fontsize=50, 
                      color=color,
                      font=font)
        
        if text == "OUTLINED":
            txt = TextClip(text, 
                          fontsize=50, 
                          color=color,
                          font=font,
                          stroke_color='white',
                          stroke_width=3)
        elif text == "SHADOWED":
            # Create shadow effect with two text clips
            shadow = TextClip(text, 
                             fontsize=50, 
                             color='gray',
                             font=font).set_position((402, 302))
            main = TextClip(text, 
                           fontsize=50, 
                           color=color,
                           font=font).set_position('center')
            txt = CompositeVideoClip([shadow, main], size=(800, 600))
        
        if isinstance(txt, TextClip):
            txt = txt.set_position('center')
        
        txt = (txt.set_start(start_time)
              .set_duration(4)
              .fadein(0.5)
              .fadeout(0.5))
        
        text_clips.append(txt)
    
    # Final composite
    final = CompositeVideoClip([bg] + text_clips)
    final.write_videofile("styles_showcase_video.mp4", fps=24)
    print("Styles showcase video created: styles_showcase_video.mp4")

if __name__ == "__main__":
    try:
        print("Creating test images and videos with ImageMagick effects...")
        create_test_image_with_imagemagick()
        create_image_with_effects()
        create_advanced_text_effects()
        create_complex_composition()
        create_text_collage()
        test_font_variations()
        create_test_video_with_imagemagick()
        create_animated_text_video()
        create_effects_showcase_video()
        create_multi_scene_video()
        create_complex_animation_video()
        create_styles_showcase_video()
        print("All test images and videos created successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure ImageMagick is installed: sudo apt-get install imagemagick")
        print("You may also need additional fonts: sudo apt-get install fonts-liberation")
        print("Also ensure you have ffmpeg installed: sudo apt-get install ffmpeg")