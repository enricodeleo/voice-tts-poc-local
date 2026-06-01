#!/usr/bin/env bash
set -euo pipefail

REF_AUDIO="output/enrico_voice.wav"
REF_TEXT="La prima cosa che vuoi sapere è se questa roba può trovare il proprio mercato, no? E questo è fondamentale per un imprenditore, perché in realtà la cosa più importante è capire se fallirai nel più breve tempo possibile. Un sito che si chiamava agrigento.it, ma long story short, su questo sito mi faceva. Così li potevo fare invece il creator di vari, per me era proprio così."

INSTRUCT=""
TEXT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --text)
      TEXT="$2"; shift 2 ;;
    --instruct)
      INSTRUCT="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TEXT" ]]; then
  echo "Usage: enrico_speaks.sh --text \"Something to say\" [--instruct \"Voice style description\"]" >&2
  exit 1
fi

if [[ ! -f "$REF_AUDIO" ]]; then
  echo "Error: reference audio not found at $REF_AUDIO" >&2
  echo "Run extract_voice.py first to generate it." >&2
  exit 1
fi

CMD="uv run synthesize.py --text \"$TEXT\" --ref_audio $REF_AUDIO --ref_text \"$REF_TEXT\""
if [[ -n "$INSTRUCT" ]]; then
  CMD="$CMD --instruct \"$INSTRUCT\""
fi

eval $CMD
