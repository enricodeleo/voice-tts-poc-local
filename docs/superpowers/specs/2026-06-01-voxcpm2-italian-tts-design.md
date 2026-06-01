# PoC: Italian TTS CLI with VoxCPM2 4-bit on MLX

## Goal

Prove that VoxCPM2 can synthesize Italian speech on Apple Silicon using the 4-bit MLX quantized model, via a minimal CLI tool.

## Context

- **Hardware:** MacBook Air M4, 24 GB RAM
- **Model:** `mlx-community/VoxCPM2-4bit` (2B params, 4-bit quantized LM, full-precision VAE/DiT, ~2.3 GB memory, 0.90x RTF)
- **Framework:** `mlx-audio` handles all MLX inference plumbing
- **Language:** Python 3.12 (via uv, since system Python is 3.14 and VoxCPM2 requires <3.13)

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

1. Validate input (exactly one of `--text` or `--file`)
2. Load model via `mlx_audio.tts.load_model("mlx-community/VoxCPM2-4bit")` (cached after first download)
3. Generate audio with sensible defaults (7 inference timesteps, cfg_value 2.0)
4. Save as 48kHz WAV to the specified output path
5. Print the output file path

## Dependencies

- `mlx-audio` — MLX-based TTS inference for Apple Silicon
- Python >=3.10, <3.13 (managed by uv)

## Out of Scope

- Voice cloning, voice design, streaming
- API server or web UI
- Custom voice selection
- Non-Italian language testing (though the model supports 30 languages)
- Model fine-tuning or training
