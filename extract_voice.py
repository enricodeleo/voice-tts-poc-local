import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a speaker's voice from a multi-speaker audio file using diarization"
    )
    parser.add_argument("--audio", required=True, help="Input audio file (mp3/wav)")
    parser.add_argument(
        "--diarization",
        required=True,
        help="Path to diarization JSON file with segments",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=None,
        help="Speaker ID to extract (omit to see samples first)",
    )
    parser.add_argument(
        "--sample_duration",
        type=float,
        default=15.0,
        help="Duration of sample clip per speaker in seconds (default: 15)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV path (default: output/speaker_<N>.wav)",
    )
    return parser.parse_args()


def extract_segments(audio_path, segments, speaker_id):
    data, sr = sf.read(audio_path)
    clips = []
    for seg in segments:
        if seg["speaker"] != speaker_id:
            continue
        start_sample = max(0, int(seg["start"] * sr))
        end_sample = min(len(data), int(seg["end"] * sr))
        clips.append(data[start_sample:end_sample])

    if not clips:
        return None, sr

    combined = np.concatenate(clips)
    return combined, sr


def extract_sample(audio_path, segments, speaker_id, duration_sec):
    speaker_segs = sorted(
        [s for s in segments if s["speaker"] == speaker_id],
        key=lambda s: s["start"],
    )
    if not speaker_segs:
        return None, 0

    collected = []
    total = 0.0
    for seg in speaker_segs:
        if total >= duration_sec:
            break
        collected.append(seg)
        total += seg["end"] - seg["start"]

    if not collected:
        return None, 0

    data, sr = sf.read(audio_path)
    clips = []
    for seg in collected:
        start_sample = max(0, int(seg["start"] * sr))
        end_sample = min(len(data), int(seg["end"] * sr))
        clips.append(data[start_sample:end_sample])

    return np.concatenate(clips), sr


def main():
    args = parse_args()

    wav_path = args.audio
    if not wav_path.endswith(".wav"):
        wav_path = "/tmp/diarize_input.wav"
        print(f"Converting to WAV: {wav_path}")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-i", args.audio, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
        )

    segments = json.loads(Path(args.diarization).read_text())

    speakers = sorted(set(s["speaker"] for s in segments))
    print(f"Found {len(speakers)} speakers: {speakers}")

    for sid in speakers:
        segs = [s for s in segments if s["speaker"] == sid]
        total = sum(s["end"] - s["start"] for s in segs)
        print(f"  Speaker {sid}: {len(segs)} segments, {total:.1f}s total")

    if args.speaker is None:
        print(f"\nExtracting {args.sample_duration}s sample from each speaker...")
        os.makedirs("output", exist_ok=True)
        for sid in speakers:
            sample, sr = extract_sample(wav_path, segments, sid, args.sample_duration)
            if sample is not None:
                out_path = f"output/speaker_{sid}_sample.wav"
                sf.write(out_path, sample, sr)
                print(f"  Speaker {sid} sample: {out_path}")

        print("\nListen to the samples and re-run with --speaker <ID> to extract full voice.")
        print("Example: uv run extract_voice.py --audio 'podcast.mp3' --diarization diarization.json --speaker 0")
        return

    print(f"\nExtracting all segments for speaker {args.speaker}...")
    combined, sr = extract_segments(wav_path, segments, args.speaker)
    if combined is None:
        print(f"Error: no segments found for speaker {args.speaker}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"output/speaker_{args.speaker}_full.wav"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, combined, sr)
    duration = len(combined) / sr
    print(f"Extracted {duration:.1f}s of speech from speaker {args.speaker}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
