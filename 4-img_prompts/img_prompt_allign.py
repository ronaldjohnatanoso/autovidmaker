import sys
import os
import re
import json
from difflib import SequenceMatcher

def parse_text_file(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(r"<(p\d+)\s+image_prompt=\"([^\"]+)\">(.*?)</\1>", re.DOTALL)
    prompts = []
    for match in pattern.finditer(text):
        tag, image_prompt, spoken_text = match.groups()
        prompts.append({
            "tag": tag,
            "image_prompt": image_prompt.strip(),
            "spoken_text": " ".join(spoken_text.strip().split())
        })
    return prompts

def parse_srt_file(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            i += 1
            if i >= len(lines): break
            time_line = lines[i].strip()
            start_str, end_str = time_line.split(" --> ")
            start = srt_time_to_seconds(start_str)
            end = srt_time_to_seconds(end_str)
            i += 1
            if i >= len(lines): break
            word = lines[i].strip()
            entries.append({
                "start": start,
                "end": end,
                "word": word
            })
            while i < len(lines) and lines[i].strip() != "":
                i += 1
        i += 1
    return entries

def srt_time_to_seconds(timestamp):
    hrs, mins, secs_ms = timestamp.split(":")
    secs, ms = secs_ms.split(",")
    return int(hrs) * 3600 + int(mins) * 60 + int(secs) + int(ms) / 1000

def fuzzy_match_window(prompt_text, srt_entries, threshold=0.85):
    words = [e['word'].lower() for e in srt_entries]
    prompt_words = prompt_text.lower().split()
    best_score = 0
    best_window = None

    for i in range(len(words) - len(prompt_words) + 1):
        window = words[i:i + len(prompt_words)]
        window_text = " ".join(window)
        prompt_text_joined = " ".join(prompt_words)
        score = SequenceMatcher(None, prompt_text_joined, window_text).ratio()
        if score > best_score:
            best_score = score
            best_window = i

    if best_score >= threshold and best_window is not None:
        start_time = srt_entries[best_window]["start"]
        end_time = srt_entries[best_window + len(prompt_words) - 1]["end"]
        matched_text = " ".join([e["word"] for e in srt_entries[best_window:best_window + len(prompt_words)]])
        return start_time, end_time, matched_text
    return None, None, None

def main():
    if len(sys.argv) != 2:
        print("Usage: python img_prompt_allign.py <project_name>")
        sys.exit(1)

    project_name = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "0-project-files", project_name))

    text_file = os.path.join(base_dir, f"{project_name}.txt")
    srt_file = os.path.join(base_dir, f"{project_name}_wordlevel.srt")
    output_json = os.path.join(base_dir, f"{project_name}_img_prompts.json")

    if not os.path.isfile(text_file):
        print(f"[ERROR] Missing text file: {text_file}")
        sys.exit(1)
    if not os.path.isfile(srt_file):
        print(f"[ERROR] Missing SRT file: {srt_file}")
        sys.exit(1)

    prompts = parse_text_file(text_file)
    srt_entries = parse_srt_file(srt_file)
    aligned = []

    for prompt in prompts:
        start, end, matched_text = fuzzy_match_window(prompt["spoken_text"], srt_entries)
        if start is None or end is None:
            print(f"[WARN] Could not align prompt: {prompt['tag']}")
            continue
        aligned.append({
            "gen-status" : 'pending',
            "gen-start": None,
            "gen-end": None,
            "gen-duration": None,
            "tag": prompt["tag"],
            "image_prompt": prompt["image_prompt"],
            "spoken_text": prompt["spoken_text"],
            "matched_text": matched_text,
            "start": round(start, 3),
            "end": round(end, 3)
        })

    aligned.sort(key=lambda x: x["start"])
    
    # forces the first prompt to start at 0.0
    if aligned:
    # Force first prompt's start time to zero
        aligned[0]["start"] = 0.0

    
    for i in range(len(aligned) - 1):
        if aligned[i]["end"] > aligned[i+1]["start"]:
            print(f"[WARNING] Overlap between {aligned[i]['tag']} and {aligned[i+1]['tag']}")
        aligned[i]["end_adjusted"] = round(aligned[i+1]["start"], 3)
        aligned[i]["duration"] = round(aligned[i]["end"] - aligned[i]["start"], 3)
        aligned[i]["duration_adjusted"] = round(aligned[i]["end_adjusted"] - aligned[i]["start"], 3)

    if aligned:
        aligned[-1]["end_adjusted"] = aligned[-1]["end"]
        aligned[-1]["duration"] = round(aligned[-1]["end"] - aligned[-1]["start"], 3)
        aligned[-1]["duration_adjusted"] = aligned[-1]["duration"]

    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(aligned, out_f, indent=2)

    print(f"✅ Image prompts with timings saved to: {output_json}")

    #update the project .status.json , to done
    status_path = os.path.join(base_dir, f"{project_name}.status.json")
    if os.path.isfile(status_path):
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
        status["stages"]["img_prompts"]["status"] = "done"

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        print(f"✅ Updated project status: {status_path}")
    else:
        print(f"[ERROR] Status file not found: {status_path}")


if __name__ == "__main__":
    main()
