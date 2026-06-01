# Voice TTS — Italian Speech Synthesis with VoxCPM2 on Apple Silicon

Proof of concept for synthesizing Italian speech on Apple Silicon using [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) 4-bit quantized via [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## How It Works

```
Text input → synthesize.py → mlx-audio → VoxCPM2-4bit (MLX) → 48kHz WAV
```

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is a 2B-parameter tokenizer-free diffusion autoregressive TTS model by OpenBMB, trained on 2M+ hours of multilingual speech. It supports 30 languages including Italian, and outputs studio-quality 48kHz audio.

The model runs on Apple Silicon through [MLX](https://github.com/ml-explore/mlx) — Apple's machine learning framework. We use the 4-bit quantized variant (`mlx-community/VoxCPM2-4bit`) which quantizes only the LM layers while keeping the VAE and DiT at full precision. This brings memory usage down to ~2.3 GB with minimal quality loss.

Three generation modes are supported:

| Mode | Description | How to use |
|------|-------------|------------|
| **Zero-shot** | Default voice, no reference needed | Just `--text` |
| **Voice Design** | Create a voice from a text description | Add `--instruct` |
| **Voice Cloning** | Clone a voice from a reference audio sample | Add `--ref_audio` + `--ref_text` |

There's also a helper script (`extract_voice.py`) for isolating a specific speaker from a multi-speaker recording (e.g., a podcast), producing a reference audio file suitable for voice cloning.

## Requirements

- **Mac with Apple Silicon** (M1 Pro/Max or newer recommended)
- **Python 3.12** (managed automatically by uv — system Python is not used)
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **ffmpeg** — for MP3/WAV conversion (`brew install ffmpeg`)
- ~3 GB disk space for the model (downloaded once, cached by Hugging Face)

## Setup

```bash
# Clone and enter the project
cd voice-tts

# Install dependencies (uv handles Python 3.12 automatically)
uv sync

# First run will download the VoxCPM2-4bit model (~2.3 GB)
uv run synthesize.py --text "Buongiorno, come stai oggi?"
```

## Usage

### Basic TTS (zero-shot)

```bash
uv run synthesize.py --text "Buongiorno, come stai oggi?"
# → Saved to output/20260601_190700.wav
```

Read text from a file:

```bash
uv run synthesize.py --file input.txt --output output/speech.wav
```

### Voice Design

Describe the voice you want — the model creates it from the text description alone, no reference audio needed.

```bash
uv run synthesize.py \
  --text "Benvenuti alla nostra podcast" \
  --instruct "A young Italian woman, warm and gentle voice"
```

### Voice Cloning

Provide a short reference audio clip (10-30 seconds of clean speech) along with its transcript. The model clones that voice.

```bash
uv run synthesize.py \
  --text "Ciao, piacere di conoscerti" \
  --ref_audio output/enrico_voice.wav \
  --ref_text "Questo è un esempio della mia voce per il cloning"
```

You can also combine voice cloning with instruct for controllable cloning:

```bash
uv run synthesize.py \
  --text "Parla più lentamente, per favore" \
  --instruct "speak slowly and clearly" \
  --ref_audio output/enrico_voice.wav \
  --ref_text "reference transcript here"
```

### Cross-Lingual Cloning

VoxCPM2 can clone a voice from a reference in one language and synthesize in another. For example, cloning an Italian voice and speaking English:

```bash
uv run synthesize.py \
  --text "Would my princess like I start the process for dinner?" \
  --ref_audio output/enrico_voice.wav \
  --ref_text "La prima cosa che vuoi sapere è se questa roba può trovare il proprio mercato"
```

The `--ref_text` should match the language spoken in the reference audio (Italian in this case), while `--text` can be in any of the 30 supported languages.

### Full Walkthrough: Clone Your Voice from a Podcast

This is the end-to-end flow: take a podcast, isolate your voice, then clone it.

**Step 1 — Diarize the podcast**

Run the podcast through a speaker diarization tool (e.g., [Xenova Whisper Speaker Diarization](https://huggingface.co/spaces/Xenova/whisper-speaker-diarization), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [WhisperX](https://github.com/m-bain/whisperX)). Save the output as a JSON file with timestamped segments and speaker labels.

**Step 2 — Extract sample clips to identify yourself**

```bash
uv run extract_voice.py \
  --audio "podcast.mp3" \
  --diarization diarization.json
```

This creates short sample WAVs per speaker (`output/speaker_0_sample.wav`, etc.). Listen to each to find which speaker ID is you.

**Step 3 — Extract all your segments**

```bash
uv run extract_voice.py \
  --audio "podcast.mp3" \
  --diarization diarization.json \
  --speaker 0 \
  --output output/my_voice.wav
```

This concatenates all your speaking turns into a single reference WAV. More reference audio = better clone quality.

**Step 4 — Clone your voice**

```bash
uv run synthesize.py \
  --text "Ciao, questa è la mia voce clonata" \
  --ref_audio output/my_voice.wav \
  --ref_text "La prima cosa che vuoi sapere se questa roba può trovare il proprio mercato"
```

**Step 5 — Convert to MP3 (optional)**

```bash
ffmpeg -i output/20260601_194715.wav -codec:a libmp3lame -qscale:a 2 output/clone.mp3
```

### Extracting a Voice from a Podcast

If you have a multi-speaker recording and want to isolate one speaker for voice cloning, use `extract_voice.py`. This requires a diarization JSON file produced by any speaker diarization tool.

**Diarization JSON format:**

```json
[
  {"start": 0.0, "end": 5.2, "speaker": 0, "text": "Hello and welcome"},
  {"start": 5.5, "end": 9.8, "speaker": 1, "text": "Thanks for having me"}
]
```

The script also accepts the Xenova format (`id`/`label` fields instead of `speaker`).

**Step 1 — Extract samples to identify speakers:**

```bash
uv run extract_voice.py \
  --audio "podcast.mp3" \
  --diarization diarization.json
```

This creates `output/speaker_0_sample.wav`, `output/speaker_1_sample.wav`, etc. Listen to each to determine which speaker is you.

**Step 2 — Extract all segments for your speaker:**

```bash
uv run extract_voice.py \
  --audio "podcast.mp3" \
  --diarization diarization.json \
  --speaker 1 \
  --output output/enrico_voice.wav
```

This concatenates all of speaker 1's segments into a single WAV file, ready to use as `--ref_audio` for voice cloning.

**Alternative diarization tools:**
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) (Python, open source)
- [WhisperX](https://github.com/m-bain/whisperX) (Python, open source)
- [Xenova Whisper Speaker Diarization](https://huggingface.co/spaces/Xenova/whisper-speaker-diarization) (browser-based)

## Project Structure

```
voice-tts/
├── pyproject.toml       # uv project config, Python 3.12, dependencies
├── synthesize.py        # TTS CLI — text/voice-design/cloning
├── extract_voice.py     # Speaker isolation from multi-speaker audio
├── .python-version      # Pins Python 3.12 for uv
├── .gitignore
└── output/              # Generated audio files (gitignored)
```

## Performance

On Apple Silicon with the 4-bit model (from [mlx-community benchmarks](https://huggingface.co/mlx-community/VoxCPM2-4bit)):

| Variant | Memory | RTF (7 timesteps) |
|---------|--------|--------------------|
| bf16    | 4.96 GB | 0.48x |
| 8-bit   | 3.23 GB | 0.85x |
| **4-bit** | **2.30 GB** | **0.90x** |

RTF = Real-Time Factor. A value below 1.0 means generation is slower than real-time. At 0.90x, generating 10 seconds of audio takes ~11 seconds.

## Technical Notes

- **mlx-audio** is pinned to the GitHub `main` branch because VoxCPM2 support hasn't been released on PyPI yet (as of June 2026). Once it lands on PyPI, the `[tool.uv.sources]` override in `pyproject.toml` can be removed.
- The 4-bit quantization applies only to the LM (language model) component. The VAE (audio encoder/decoder) and DiT (diffusion transformer) remain at full precision, which is why quality holds up well.
- Inference uses 7 timesteps and cfg_value of 2.0 by default, matching the model card recommendations. These trade a small amount of quality for faster generation.

## License

This project uses [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) under the Apache 2.0 license. The [mlx-audio](https://github.com/Blaizzy/mlx-audio) framework is under the MIT license.
