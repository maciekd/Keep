"""Transcribes one audio file with mlx-whisper and writes its words with their timestamps as JSON.

Run by fetch_and_transcribe.py as a subprocess, never imported for the work, so the model's
MLX/Metal memory is released the moment the transcription ends. It imports nothing of Keep.

    python transcribe_audio.py <audio> <out_words.json> [<model repo>]
"""
import json
import sys
import time


def transcribe(audio_path: str, model: str) -> list[dict]:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        language="en",
        word_timestamps=True,
        # Off: on 2 h+ audio, conditioning on the previous output is the classic trigger of
        # Whisper's repetition loops.
        condition_on_previous_text=False,
        verbose=None,
    )
    return [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
            for segment in result["segments"] for w in segment.get("words", [])]


def main():
    audio, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "mlx-community/whisper-large-v3-turbo"
    started = time.time()
    words = transcribe(audio, model)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    print(f"Transcribed {len(words)} words in {time.time() - started:.0f}s -> {out}")


if __name__ == "__main__":
    main()
