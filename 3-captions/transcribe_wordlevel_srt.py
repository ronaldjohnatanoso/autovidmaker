import whisperx
import torch
import sys
import os

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

def transcribe_word_level_srt(project_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    # 🧭 Resolve path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "0-project-files", project_name))
    audio_path = os.path.join(base_dir, f"{project_name}.wav")

    if not os.path.isfile(audio_path):
        print(f"[ERROR] Audio file not found at: {audio_path}")
        sys.exit(1)

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

    # 🔹 Save word-level SRT
    word_srt_path = os.path.join(base_dir, f"{project_name}_wordlevel.srt")
    with open(word_srt_path, "w", encoding="utf-8") as srt_file:
        for i, word_info in enumerate(result_aligned["word_segments"], start=1):
            start = format_timestamp(word_info["start"])
            end = format_timestamp(word_info["end"])
            text = word_info["word"].strip()
            srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"✅ Word-level SRT saved to: {word_srt_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_wordlevel_srt.py <project_name>")
        sys.exit(1)

    transcribe_word_level_srt(sys.argv[1])
