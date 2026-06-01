import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize Italian speech using VoxCPM2 4-bit on Apple Silicon"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Italian text to synthesize")
    group.add_argument("--file", help="Path to a text file with Italian content")
    parser.add_argument(
        "--output",
        help="Output WAV path (default: output/<timestamp>.wav)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text:
        text = args.text
    else:
        filepath = Path(args.file)
        if not filepath.is_file():
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        text = filepath.read_text().strip()

    if not text:
        print("Error: empty text input", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/{timestamp}.wav"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("Loading model...")
    model = load_model("mlx-community/VoxCPM2-4bit")

    print(f"Generating speech for: {text[:80]}{'...' if len(text) > 80 else ''}")
    audio = None
    sample_rate = None
    for result in model.generate(
        text=text,
        inference_timesteps=7,
        cfg_value=2.0,
    ):
        audio = np.array(result.audio, dtype=np.float32)
        sample_rate = result.sample_rate

    if audio is None:
        print("Error: model produced no audio", file=sys.stderr)
        sys.exit(1)

    sf.write(output_path, audio, sample_rate)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
