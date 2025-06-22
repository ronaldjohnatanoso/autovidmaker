import os
import sys
import re
import shutil
import json
from datetime import datetime

# ------------------------ Rule Check Functions ------------------------

def check_duration_line(lines):
    if not lines:
        return False, "Script is empty"
    if not lines[0].startswith("duration_estimation_milliseconds:"):
        return False, "First line must start with 'duration_estimation_milliseconds:'"
    try:
        duration_part = lines[0].split(":", 1)[1].strip()
        millis = int(duration_part)
        if millis <= 0:
            return False, "Estimated milliseconds must be a positive integer"
    except (ValueError, IndexError):
        return False, "Invalid duration format. Expected 'duration_estimation_milliseconds: <number>'"
    return True, None

def check_voice_instruction(lines):
    if len(lines) < 2:
        return False, "Missing voice instruction after duration"
    if not lines[1].startswith("voice_instruction:"):
        return False, "Second line must start with 'voice_instruction:'"
    instruction = lines[1].split(":", 1)[1].strip()
    if not instruction:
        return False, "Voice instruction cannot be empty"
    return True, None

def check_title_line(lines):
    if len(lines) < 3:
        return False, "Missing title after voice instruction"
    if not lines[2].startswith("title:"):
        return False, "Third line must start with 'title:'"
    title = lines[2].split(":", 1)[1].strip()
    if not title:
        return False, "Title cannot be empty"
    return True, None

def check_speaker_line_format(line):
    if not re.match(r"^Speaker [12]:", line):
        return False, f"Invalid speaker line format: '{line}'. Expected 'Speaker 1:' or 'Speaker 2:'"
    return True, None

def validate_script_lines(lines):
    if len(lines) < 4:  # Now need at least 4 lines: duration, voice, title, content
        return False, "No content after title"
    
    # Join all lines into a single text block for multi-line parsing
    full_text = '\n'.join(lines[3:])  # Skip duration, voice instruction, and title
    
    # Split by Speaker lines while preserving the speaker markers
    speaker_sections = re.split(r'(^Speaker [12]:)', full_text, flags=re.MULTILINE)
    
    # Remove empty strings and organize into pairs
    sections = [s for s in speaker_sections if s.strip()]
    
    if not sections:
        return False, "No speaker content found"
    
    # Process each speaker section
    i = 0
    while i < len(sections):
        if not re.match(r'^Speaker [12]:$', sections[i]):
            return False, f"Expected speaker line, got: '{sections[i]}'"
        
        if i + 1 >= len(sections):
            return False, "Speaker line found but no content follows"
        
        content = sections[i + 1].strip()
        
        # Check if this content contains proper paragraph tags
        if not validate_paragraph_content(content):
            return False, f"Invalid paragraph content after {sections[i]}: content must be wrapped in proper <p#> tags with image_prompt"
        
        i += 2
    
    return True, None

def validate_paragraph_content(content):
    # Find all paragraph tags in the content
    paragraph_pattern = re.compile(r'<p(\d+)\s+image_prompt="[^"]*">(.*?)</p\1>', re.DOTALL)
    matches = paragraph_pattern.findall(content)
    
    if not matches:
        return False
    
    # Remove all valid paragraph tags and check if anything substantial remains
    cleaned_content = paragraph_pattern.sub('', content).strip()
    
    # Allow only whitespace and newlines to remain
    if cleaned_content and not re.match(r'^\s*$', cleaned_content):
        return False
    
    return True

def validate_script(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]  # Keep empty lines but remove trailing whitespace

    # Filter out completely empty lines for validation but preserve structure
    non_empty_lines = [line for line in lines if line.strip()]

    rules = [
        lambda l: check_duration_line(l),
        lambda l: check_voice_instruction(l),
        lambda l: check_title_line(l),  # Add title validation
        lambda l: validate_script_lines(l)
    ]

    for rule in rules:
        valid, error = rule(non_empty_lines)
        if not valid:
            return False, error

    return True, non_empty_lines

# ------------------------ Folder Move and Raw Generation ------------------------

def initialize_status_file(target_dir):
    now_iso = datetime.utcnow().isoformat() + "Z"
    status = {
        "created_at": now_iso,
        "current_stage": "voice_gen",
        "stages": {}
    }

    for stage in ["voice_gen", "captions", "img_prompts", "image_gen", "upscale_img", "img_stitch", "vid_edit"]:
        status["stages"][stage] = {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "duration": None,
            "error": None,
            "metadata": {}
        }

    status_path = os.path.join(target_dir, ".status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    return status_path

def write_config_file(target_dir, estimated_millis):
    config = {
        "stages": {
            "voice_gen": {
                "estimated_duration_ms": estimated_millis,
                "speaker1" : "Sadachbia",
                "speaker2" : "Gacrux",
                "voice_effect": "none"
            },
            "captions": {},
            "img_prompts": {},
            "image_gen": {},
            "upscale_img": {},
            "img_stitch": {
                "background_music_file": "choir.mp3",
                "shader" : "holy.glsl",
            },
            "vid_edit": {}
        }
    }

    config_path = os.path.join(target_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config_path

def move_to_project_folder(filepath):
    filename = os.path.basename(filepath)
    name_only, _ = os.path.splitext(filename)

    staging_dir = os.path.dirname(filepath)
    parent_dir = os.path.abspath(os.path.join(staging_dir, ".."))
    project_root = os.path.join(parent_dir, "0-project-files")
    target_dir = os.path.join(project_root, name_only)

    os.makedirs(target_dir, exist_ok=True)

    dest_path = os.path.join(target_dir, filename)
    shutil.move(filepath, dest_path)
    return dest_path, target_dir

def strip_tags_preserve_text(lines):
    # Extract voice instruction (line 1)
    voice_instruction = lines[1].split(":", 1)[1].strip()
    stripped_lines = [voice_instruction]  # Keep the voice instruction

    # Join all content lines after title for proper multi-line processing
    full_text = '\n'.join(lines[3:])  # Skip duration, voice instruction, and title
    
    # Split by Speaker lines while preserving the speaker markers
    speaker_sections = re.split(r'(^Speaker [12]:)', full_text, flags=re.MULTILINE)
    
    # Remove empty strings
    sections = [s for s in speaker_sections if s.strip()]
    
    i = 0
    while i < len(sections):
        if re.match(r'^Speaker [12]:$', sections[i]):
            speaker_line = sections[i].strip()
            if i + 1 < len(sections):
                content = sections[i + 1].strip()
                # Remove paragraph tags with DOTALL flag for multi-line content
                text_only = re.sub(r'<p\d+\s+image_prompt="[^"]*">(.*?)</p\d+>', r'\1', content, flags=re.DOTALL)
                # Clean up extra whitespace but preserve line breaks
                text_only = re.sub(r'\n\s*\n', '\n', text_only.strip())
                stripped_lines.append(f"{speaker_line} {text_only}")
            i += 2
        else:
            i += 1

    return stripped_lines


def write_raw_version(original_lines, raw_path):
    raw_lines = strip_tags_preserve_text(original_lines)
    with open(raw_path, "w", encoding="utf-8") as f:
        for line in raw_lines:
            f.write(line + "\n")

# ------------------------ Main ------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_and_move.py <script_file.txt>")
        sys.exit(1)

    script_path = sys.argv[1]
    if not os.path.isfile(script_path):
        print(f"[ERROR] File not found: {script_path}")
        sys.exit(1)

    is_valid, result = validate_script(script_path)
    if not is_valid:
        print(f"[ERROR] Script '{os.path.basename(script_path)}' rejected: {result}")
        sys.exit(1)

    original_lines = result
    # Extract duration from the new format
    estimated_millis = int(original_lines[0].split(":", 1)[1].strip())
    moved_path, target_dir = move_to_project_folder(script_path)

    name_only, _ = os.path.splitext(os.path.basename(script_path))
    raw_path = os.path.join(target_dir, f"{name_only}_raw.txt")

    write_raw_version(original_lines, raw_path)
    status_path = initialize_status_file(target_dir)
    config_path = write_config_file(target_dir, estimated_millis)

    print(f"Status file created: {status_path}")
    print(f"Config file created: {config_path}")
    print(f"[OK] Script '{os.path.basename(script_path)}' accepted.")
    print(f"Moved to: {target_dir}")
    print(f"Raw version created: {raw_path}")
