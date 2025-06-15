import whisperx
import torch
import sys
import os
import re
from difflib import SequenceMatcher

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

def clean_text_for_alignment(text):
    """Minimal cleaning - only normalize spaces and case for comparison."""
    # Only remove extra whitespace and convert to lowercase for matching
    # Keep all punctuation and special characters
    cleaned = re.sub(r'\s+', ' ', text.lower().strip())
    return cleaned

def align_ground_truth_with_whisper(ground_truth_text, whisper_words):
    """
    Conservative alignment that preserves original ground truth text structure.
    Only replaces words that WhisperX clearly got wrong.
    """
    # Split ground truth into words while preserving original text
    gt_words_original = ground_truth_text.split()
    
    # Create cleaned versions for comparison only
    gt_words_clean = [clean_text_for_alignment(word) for word in gt_words_original]
    whisper_words_clean = [clean_text_for_alignment(word_info["word"]) for word_info in whisper_words]
    
    print(f"[INFO] Ground truth words: {len(gt_words_original)}")
    print(f"[INFO] WhisperX words: {len(whisper_words)}")
    
    # Use SequenceMatcher on cleaned versions
    matcher = SequenceMatcher(None, whisper_words_clean, gt_words_clean)
    
    aligned_segments = []
    gt_index = 0
    whisper_index = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match - use original ground truth text with WhisperX timing
            for k in range(i2 - i1):
                if whisper_index + k < len(whisper_words) and gt_index + k < len(gt_words_original):
                    original_segment = whisper_words[whisper_index + k]
                    aligned_segments.append({
                        "start": original_segment["start"],
                        "end": original_segment["end"],
                        "word": gt_words_original[gt_index + k]  # Use original text
                    })
            whisper_index += (i2 - i1)
            gt_index += (j2 - j1)
            
        elif tag == 'replace':
            # Words don't match - use original ground truth with distributed timing
            gt_chunk = gt_words_original[j1:j2]  # Use original text
            whisper_chunk_len = i2 - i1
            
            if whisper_chunk_len > 0 and whisper_index < len(whisper_words):
                start_time = whisper_words[whisper_index]["start"]
                if whisper_index + whisper_chunk_len - 1 < len(whisper_words):
                    end_time = whisper_words[whisper_index + whisper_chunk_len - 1]["end"]
                else:
                    end_time = whisper_words[-1]["end"]
                
                # Distribute timing across original ground truth words
                time_per_word = (end_time - start_time) / len(gt_chunk) if len(gt_chunk) > 0 else 0
                
                for k, word in enumerate(gt_chunk):
                    word_start = start_time + k * time_per_word
                    word_end = start_time + (k + 1) * time_per_word
                    aligned_segments.append({
                        "start": word_start,
                        "end": word_end,
                        "word": word  # Original ground truth word
                    })
            
            whisper_index += whisper_chunk_len
            gt_index += len(gt_chunk)
            
        elif tag == 'delete':
            # WhisperX has extra words, skip them
            whisper_index += (i2 - i1)
            
        elif tag == 'insert':
            # Ground truth has extra words, estimate timing
            gt_chunk = gt_words_original[j1:j2]  # Use original text
            
            if whisper_index > 0 and whisper_index < len(whisper_words):
                prev_end = whisper_words[whisper_index - 1]["end"]
                next_start = whisper_words[whisper_index]["start"]
                time_per_word = (next_start - prev_end) / len(gt_chunk) if len(gt_chunk) > 0 else 0.1
                
                for k, word in enumerate(gt_chunk):
                    word_start = prev_end + k * time_per_word
                    word_end = prev_end + (k + 1) * time_per_word
                    aligned_segments.append({
                        "start": word_start,
                        "end": word_end,
                        "word": word  # Original ground truth word
                    })
            elif whisper_index == 0 and len(whisper_words) > 0:
                # Insert at beginning
                first_start = whisper_words[0]["start"]
                time_per_word = first_start / len(gt_chunk) if len(gt_chunk) > 0 else 0.1
                
                for k, word in enumerate(gt_chunk):
                    word_start = k * time_per_word
                    word_end = (k + 1) * time_per_word
                    aligned_segments.append({
                        "start": word_start,
                        "end": word_end,
                        "word": word
                    })
            
            gt_index += len(gt_chunk)
    
    return aligned_segments

def eliminate_word_gaps(word_segments, max_gap_threshold=0.5):
    """
    Eliminates small gaps between words by extending each word's end time
    to the start of the next word, unless the gap exceeds max_gap_threshold.
    
    Args:
        word_segments: List of word segments with 'start', 'end', and 'word' keys
        max_gap_threshold: Maximum gap in seconds to bridge (default: 0.5 second)
    
    Returns:
        List of word segments with adjusted end times
    """
    if not word_segments:
        return word_segments
    
    adjusted_segments = []
    
    for i in range(len(word_segments)):
        current_word = word_segments[i].copy()
        
        # Check if there's a next word
        if i < len(word_segments) - 1:
            next_word = word_segments[i + 1]
            gap = next_word["start"] - current_word["end"]
            
            # If gap is small enough, extend current word to start of next word
            if 0 < gap <= max_gap_threshold:
                current_word["end"] = next_word["start"]
        
        adjusted_segments.append(current_word)
    
    return adjusted_segments

def transcribe_word_level_srt(project_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    # 🧭 Resolve path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "0-project-files", project_name))
    
    # Try multiple possible audio file names in order of preference
    audio_candidates = [
        f"{project_name}_original.wav",
        f"{project_name}.wav"
    ]

    audio_path = None
    for candidate in audio_candidates:
        candidate_path = os.path.join(base_dir, candidate)
        if os.path.isfile(candidate_path):
            audio_path = candidate_path
            print(f"[INFO] Using audio file: {candidate}")
            break

    if audio_path is None:
        print(f"[ERROR] No audio file found. Searched for:")
        for candidate in audio_candidates:
            print(f"  - {os.path.join(base_dir, candidate)}")
        sys.exit(1)

    # 🔹 Load ground truth script if available
    ground_truth_path = os.path.join(base_dir, f"{project_name}_raw.txt")
    ground_truth_text = None
    
    if os.path.isfile(ground_truth_path):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read()
        print(f"[INFO] Ground truth script found: {ground_truth_path}")
    else:
        print(f"[INFO] No ground truth script found at: {ground_truth_path}")

    model = whisperx.load_model("small.en", device=device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    # 🔹 Save segment-level SRT
    segment_srt_path = os.path.join(base_dir, f"{project_name}_segments.srt")
    with open(segment_srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"✅ Segment-level SRT saved to: {segment_srt_path}")

    # 🔹 Word-level alignment
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # 🔹 Get word segments
    word_segments = result_aligned["word_segments"]
    
    # 🔹 If ground truth is available, align it with WhisperX timing
    if ground_truth_text:
        print("[INFO] Aligning ground truth text with WhisperX timing...")
        word_segments = align_ground_truth_with_whisper(ground_truth_text, word_segments)
        print(f"[INFO] Aligned {len(word_segments)} words")

    # 🔹 Eliminate gaps between words
    adjusted_word_segments = eliminate_word_gaps(word_segments, max_gap_threshold=0.5)

    # 🔹 Save word-level SRT
    word_srt_path = os.path.join(base_dir, f"{project_name}_wordlevel.srt")
    with open(word_srt_path, "w", encoding="utf-8") as srt_file:
        for i, word_info in enumerate(adjusted_word_segments, start=1):
            start = format_timestamp(word_info["start"])
            end = format_timestamp(word_info["end"])
            text = word_info["word"].strip()
            srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"✅ Word-level SRT saved to: {word_srt_path}")
    
    if ground_truth_text:
        print("✅ Used ground truth text for accurate word-level captions!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_wordlevel_srt.py <project_name>")
        sys.exit(1)

    transcribe_word_level_srt(sys.argv[1])
