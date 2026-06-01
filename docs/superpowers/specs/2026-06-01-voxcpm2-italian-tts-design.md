# PoC: Italian TTS CLI with VoxCPM2 4-bit on MLX

## Goal

Prove that VoxCPM2 can synthesize Italian speech on Apple Silicon using the 4-bit MLX quantized model, via a minimal CLI tool.

## Context

- **Hardware:** MacBook Air M4, 24 GB RAM
- **Model:** `mlx-community/VoxCPM2-4bit` (2B params, 4-bit quantized LM, full-precision VAE/DiT, ~2.3 GB memory, ~0.90x RTF)
- **Framework:** `mlx-audio` handles all MLX inference plumbing
- **Runtime:** Python 3.12 (via uv, since system Python is 3.14 and mlx-audio requires <3.13)
- **Caveat:** VoxCPM2 support in mlx-audio is relatively new. If the `mlx-community/VoxCPM2-4bit` model doesn't load, we may need to update mlx-audio to the latest version or use the CLI directly (`python -m mlx_audio.tts.generate`).

## Architecture

Single-file CLI that wraps mlx-audio's generation API.

```
User → synthesize.py (CLI) → mlx-audio → VoxCPM2-4bit (MLX) → WAV file
```

## Project Structure

```
voice-tts/
├── pyproject.toml      # uv project, Python 3.12, mlx-audio dependency
├── synthesize.py       # CLI entry point
└── output/             # Generated WAV files (gitignored)
```

## CLI Interface

```
uv run synthesize.py --text "Ciao mondo"
uv run synthesize.py --file input.txt --output speech.wav
```

### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--text` | Yes (or `--file`) | Italian text to synthesize |
| `--file` | Yes (or `--text`) | Path to a text file with Italian content |
| `--output` | No | Output WAV path (default: `output/<timestamp>.wav`) |

### Behavior

1. Validate input — exactly one of `--text` or `--file` must be provided; exit with non-zero code and error message on failure
2. Auto-create `output/` directory if it doesn't exist
3. Load model via `from mlx_audio.tts.utils import load_model` → `load_model("mlx-community/VoxCPM2-4bit")` (cached by Hugging Face after first download)
4. Generate audio via `model.generate(text=..., inference_timesteps=7, cfg_value=2.0)` — these are the recommended defaults from the model's HF page, passed explicitly
5. Convert the returned `mx.array` waveform to WAV using `soundfile.write()` at 48kHz
6. Print the output file path

## Dependencies

- `mlx-audio` — MLX-based TTS inference for Apple Silicon
- `soundfile` — WAV serialization from numpy/mx.array (lightweight, no ffmpeg needed)
- Python >=3.10, <3.13 (managed by uv)

## Out of Scope

- Voice cloning, voice design, streaming
- API server or web UI
- Custom voice selection
- Non-Italian language testing (though the model supports 30 languages)
- Model fine-tuning or training
