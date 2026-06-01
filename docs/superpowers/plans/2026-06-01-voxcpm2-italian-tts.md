# VoxCPM2 Italian TTS CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal CLI that synthesizes Italian speech from text using VoxCPM2 4-bit on Apple Silicon via mlx-audio.

**Architecture:** Single-file Python CLI (`synthesize.py`) backed by a uv-managed project with Python 3.12. Uses `mlx_audio.tts.load_model` to load the 4-bit quantized model, generates audio, converts to numpy, and writes WAV output via `soundfile`. Sample rate is read from the generation result (not hardcoded).

**Tech Stack:** Python 3.12, uv, mlx-audio, soundfile, numpy, mlx-community/VoxCPM2-4bit

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "voice-tts"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "mlx-audio",
    "soundfile",
]

[tool.uv]
python = "3.12"
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
output/
*.egg-info/
.venv/
```

- [ ] **Step 3: Initialize the venv and install dependencies**

Run: `uv sync`
Expected: uv downloads Python 3.12, creates `.venv/`, installs mlx-audio and soundfile.

- [ ] **Step 4: Verify Python version in venv**

Run: `uv run python --version`
Expected: `Python 3.12.x`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "scaffold: uv project with mlx-audio and soundfile deps"
```

---

### Task 2: Write the CLI script

**Files:**
- Create: `synthesize.py`

- [ ] **Step 1: Write `synthesize.py`**

```python
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
        text = Path(args.file).read_text().strip()

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
    for result in model.generate(
        text=text,
        inference_timesteps=7,
        cfg_value=2.0,
    ):
        audio = np.array(result.audio, dtype=np.float32)
        sample_rate = result.sample_rate

    sf.write(output_path, audio, sample_rate)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add synthesize.py
git commit -m "feat: add Italian TTS CLI using VoxCPM2 4-bit via mlx-audio"
```

---

### Task 3: End-to-end smoke test

**Files:** none new

- [ ] **Step 1: Run the CLI with a short Italian phrase**

Run: `uv run synthesize.py --text "Buongiorno, come stai oggi?"`
Expected:
- Model downloads on first run (~2.3 GB from Hugging Face, cached afterward)
- Prints "Loading model..." then "Generating speech..."
- Prints "Saved to output/<timestamp>.wav"

- [ ] **Step 2: Verify the output file exists and is non-empty**

Run: `ls -la output/`
Expected: A `.wav` file exists with size > 0

- [ ] **Step 3: Verify the WAV file is valid**

Run: `uv run python -c "import soundfile as sf; data, sr = sf.read('output/*.wav'); print(f'Sample rate: {sr}, Duration: {len(data)/sr:.2f}s')"`
Expected: `Sample rate: 44100, Duration: X.XXs`

- [ ] **Step 4: Listen to the output**

Play the file manually to confirm Italian speech quality.

- [ ] **Step 5: Test with --file flag**

Create a test text file and run:
```bash
echo "Questo è un test di sintesi vocale in italiano." > test_input.txt
uv run synthesize.py --file test_input.txt --output output/test_file.wav
```
Expected: `Saved to output/test_file.wav`

- [ ] **Step 6: Test error case (no input)**

Run: `uv run synthesize.py`
Expected: Error message about requiring `--text` or `--file`, non-zero exit code.

- [ ] **Step 7: Cleanup**

```bash
rm test_input.txt
```

---

### Task 4: Final cleanup

- [ ] **Step 1: Add .python-version file for uv**

```bash
echo "3.12" > .python-version
```

- [ ] **Step 2: Commit**

```bash
git add .python-version
git commit -m "chore: pin Python 3.12 for uv"
```
