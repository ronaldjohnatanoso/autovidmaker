import whisper
import sys
import os

def transcribe_to_srt(audio_path):
    # Load the model
    model = whisper.load_model("small.en")

    # Transcribe and get segments
    result = model.transcribe(audio_path, verbose=True)
    
    # Output SRT file
    srt_path = os.path.splitext(audio_path)[0] + ".srt"
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()

            srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    print(f"SRT saved to {srt_path}")

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_to_srt.py your_audio_file.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.isfile(audio_file):
        print("File does not exist.")
        sys.exit(1)

    transcribe_to_srt(audio_file)
