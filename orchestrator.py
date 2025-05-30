import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# ------------------- Pipeline Configuration -------------------

STAGE_CONFIG = {
    "voice_gen": {
        "inputs": ["{name}_raw.txt"],
        "command": ["node", "2-voice_gen/pipeline-tts-runner.js", "{name}"],
    },
    "captions": {
        "inputs": ["{name}.wav"],
        "command": ["python", "3-captions/transcribe_wordlevel_srt.py", "{name}"],
    },
    "img_prompts": {
        "inputs": ["{name}_wordlevel.srt", "{name}.txt"],
        "command": ["python", "4-img_prompts/img_prompt_allign.py", "{name}"],
    },
    "image_gen": {
        "inputs": ["{name}_img_prompts.json"],
        "command": ["node", "5-image_gen/pipeline-img-gen.js", "{name}"],
    },
    "upscale_img" : {
        "inputs" : ["images"],
        "command": ["node", "6-upscale_img/batch_upscale.js", "{name}"],
    }
}

STAGE_ORDER = list(STAGE_CONFIG.keys())

# ------------------- Utilities -------------------

def get_project_path(project_name):
    return os.path.join("0-project-files", project_name)

def get_status_path(project_path):
    return os.path.join(project_path, ".status.json")

def load_status(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stages": {}}

def write_status(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def now_utc():
    return datetime.utcnow().isoformat() + "Z"

def duration(start, end):
    start_dt = datetime.fromisoformat(start.replace("Z", ""))
    end_dt = datetime.fromisoformat(end.replace("Z", ""))
    return (end_dt - start_dt).total_seconds()

def find_next_stage(status):
    for stage in STAGE_ORDER:
        stage_info = status["stages"].get(stage, {})
        if stage_info.get("status") not in ("done", "in_progress"):
            return stage
    return None

# ------------------- Stage Handling -------------------

def check_inputs(project_path, project_name, stage):
    config = STAGE_CONFIG[stage]
    missing = []
    for pattern in config["inputs"]:
        expected = os.path.join(project_path, pattern.format(name=project_name))
        if not os.path.exists(expected):
            missing.append(expected)
    return missing

def run_stage(project_name, stage):
    config = STAGE_CONFIG[stage]
    command = [arg.format(name=project_name) for arg in config["command"]]
    print(f"⚙️  Running: {' '.join(command)}")
    subprocess.run(command, check=True)

# ------------------- Main Entry -------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <project_name> [--autoprocess | --autoprocess-all | --staging | <stage>]")
        sys.exit(1)

    project_name = sys.argv[1]
    autoprocess = "--autoprocess" in sys.argv
    autoprocess_all = "--autoprocess-all" in sys.argv
    staging = "--staging" in sys.argv

    if staging:
        print("Staging mode for project:", project_name)

        # Find the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        staging_dir = os.path.join(script_dir, "1-script-staging")
        script_path = os.path.join(staging_dir, f"{project_name}.txt")

        # Check if the file exists
        if not os.path.exists(script_path):
            print(f"[ERROR] Script file not found: {script_path}")
            sys.exit(1)

        print(f"✅ Found script file: {script_path}")
        print(f"📦 Validating and moving script using: validate_and_move.py")

        # Run the validation script
        try:
            subprocess.run(
                ["python", "validate_and_move.py", f"{project_name}.txt"],
                cwd=staging_dir,
                check=True
            )
            print("✅ Validation and move completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] validate_and_move.py failed: {e}")
        sys.exit(1)

    project_path = get_project_path(project_name)
    if not os.path.isdir(project_path):
        print(f"[1ERROR] Project folder not found: {project_path}")
        sys.exit(1)

    status_path = get_status_path(project_path)
    status = load_status(status_path)

    def process_stage(stage):
        missing = check_inputs(project_path, project_name, stage)
        stage_data = status["stages"].get(stage, {})
        start_time = now_utc()

        if missing:
            stage_data.update({
                "status": "error",
                "started_at": start_time,
                "error": f"Missing input files: {missing}"
            })
            status["stages"][stage] = stage_data
            write_status(status, status_path)
            print(f"[❌] Missing input files for stage '{stage}':")
            for m in missing:
                print(f" - {m}")
            return False

        stage_data.update({
            "status": "in_progress",
            "started_at": start_time,
            "error": None
        })
        status["stages"][stage] = stage_data
        write_status(status, status_path)

        try:
            run_stage(project_name, stage)
        except subprocess.CalledProcessError as e:
            stage_data.update({
                "status": "failed",
                "error": f"Process error: {str(e)}"
            })
            status["stages"][stage] = stage_data
            write_status(status, status_path)
            print(f"[✗] Stage '{stage}' failed.")
            return False

        end_time = now_utc()
        stage_data.update({
            "status": "done",
            "ended_at": end_time,
            "duration_seconds": duration(start_time, end_time)
        })
        status["stages"][stage] = stage_data
        write_status(status, status_path)
        print(f"[✅] Stage '{stage}' completed.")
        return True


        
    if autoprocess_all:
        while True:
            stage = find_next_stage(status)
            if not stage:
                print("✅ All stages are completed.")
                break
            print(f"▶️  Processing stage: {stage}")
            success = process_stage(stage)
            if not success:
                break
        return

    if autoprocess:
        stage = find_next_stage(status)
        if not stage:
            print("✅ All stages are already completed.")
            sys.exit(0)
        print(f"🧠 Next stage to process: {stage}")
        if not process_stage(stage):
            sys.exit(1)
        return

    # Manual stage input
    if len(sys.argv) < 3:
        print("Specify a stage name or use --autoprocess / --autoprocess-all")
        sys.exit(1)

    stage = sys.argv[2]
    if stage not in STAGE_CONFIG:
        print(f"[ERROR] Unknown stage: {stage}")
        sys.exit(1)

    if not process_stage(stage):
        sys.exit(1)

if __name__ == "__main__":
    main()
