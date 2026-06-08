import os
from pathlib import Path
import subprocess
import sys


MODEL_PATH = Path("models/best_model.pkl")


def main():
    if os.getenv("FORCE_TRAIN", "").lower() in ("1", "true", "yes"):
        print("FORCE_TRAIN is set. Training model...")
        subprocess.run([sys.executable, "main.py"], check=True)
    elif not MODEL_PATH.exists():
        print(f"{MODEL_PATH} not found. Training model first...")
        subprocess.run([sys.executable, "main.py"], check=True)
    else:
        print(f"Using existing model at {MODEL_PATH}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
