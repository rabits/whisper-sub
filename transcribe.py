#!/usr/bin/env python3
"""Transcribe local audio or video files with faster-whisper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

FORMATS = ("json", "text", "srt", "vtt")
EXT = {"json": ".json", "text": ".txt", "srt": ".srt", "vtt": ".vtt"}
EXT_TO_FORMAT = {ext: fmt for fmt, ext in EXT.items()}
ZH_CONVERT_OFF = {"", "off", "none", "no", "false"}

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "default")
DEFAULT_ZH_CONVERT = os.getenv("WHISPER_ZH_CONVERT", "s2twp")
DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT") or None

_models: dict[tuple[str, str, str], object] = {}
_converters: dict[str, object | None] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe.py",
        description="Transcribe or translate local audio/video files with faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s interview.mp3
  %(prog)s interview.mp3 -f srt
  %(prog)s interview.mp3 -o interview.srt
  %(prog)s episode.mp4 -l ru -m small --device cpu --compute-type int8
  %(prog)s episode.mp4 -l ru -m large-v3 --device cuda --compute-type float16 -f srt
  %(prog)s a.mp3 b.wav -f srt -o subs/
  %(prog)s interview.mp3 --task translate --stdout
""",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="audio or video files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file, or a directory when several inputs are given (default: next to each input)",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=FORMATS,
        help="json, text, srt, or vtt (default: text, or the extension of --output)",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="transcribe in the source language, or translate speech into English (default: transcribe)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="",
        help="language code such as ru, en, zh, ja; empty means auto-detect",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help=f"model name: tiny, base, small, medium, large-v3, turbo (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="",
        help=f"auto, cpu, or cuda. ROCm uses cuda as well (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--compute-type",
        default="",
        help=f"default, int8, float16, int8_float16, ... (default: {DEFAULT_COMPUTE_TYPE})",
    )
    parser.add_argument("--beam-size", type=int, default=5, metavar="N", help="1-20 (default: 5)")
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop silent stretches (default: on)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="include per-word timestamps (shown in json)",
    )
    parser.add_argument(
        "--initial-prompt",
        default="",
        help="names and domain terms to bias recognition",
    )
    parser.add_argument(
        "--zh-convert",
        default="",
        help=f"Chinese script conversion: s2twp, s2tw, s2t, or off (default: {DEFAULT_ZH_CONVERT})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the transcript instead of writing a file",
    )
    return parser


def looks_like_dir(output: Path) -> bool:
    return str(output).endswith(("/", "\\")) or output.expanduser().is_dir()


def infer_format(explicit: str | None, output: Path | None) -> str:
    if explicit:
        return explicit
    if output is not None and not looks_like_dir(output):
        mapped = EXT_TO_FORMAT.get(output.suffix.lower())
        if mapped:
            return mapped
    return "text"


def output_path_for(source: Path, fmt: str, output: Path | None, multiple: bool) -> Path:
    ext = EXT[fmt]
    source = source.expanduser().resolve()
    if output is None:
        return source.with_suffix(ext)
    target = output.expanduser()
    # A suffix-less path is a directory when several files are transcribed,
    # so `-o subs` works before that directory exists.
    if looks_like_dir(target) or (multiple and not target.suffix):
        return (target / f"{source.stem}{ext}").resolve()
    if multiple:
        raise ValueError("--output must be a directory when transcribing more than one file")
    return target.resolve()


def plan_outputs(sources: list[Path], fmt: str, output: Path | None) -> list[tuple[Path, Path]]:
    multiple = len(sources) > 1
    planned: list[tuple[Path, Path]] = []
    seen: dict[Path, Path] = {}
    for source in sources:
        resolved = source.expanduser().resolve()
        dest = output_path_for(resolved, fmt, output, multiple)
        if dest == resolved:
            raise ValueError(f"output path would overwrite the input: {dest}")
        previous = seen.get(dest)
        if previous is not None:
            raise ValueError(f"output collision: {previous} and {resolved} both map to {dest}")
        seen[dest] = resolved
        planned.append((resolved, dest))
    return planned


def stamp(seconds: float, srt: bool = False) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds_part, ms = divmod(ms, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}{separator}{ms:03d}"


def render(result: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    if fmt == "text":
        return result["text"].strip() + "\n"
    if fmt == "srt":
        blocks = []
        for index, segment in enumerate(result["segments"], 1):
            blocks.append(
                f"{index}\n{stamp(segment['start'], True)} --> {stamp(segment['end'], True)}\n{segment['text'].strip()}"
            )
        return "\n\n".join(blocks) + "\n"
    if fmt == "vtt":
        blocks = ["WEBVTT"]
        for segment in result["segments"]:
            blocks.append(f"{stamp(segment['start'])} --> {stamp(segment['end'])}\n{segment['text'].strip()}")
        return "\n\n".join(blocks) + "\n"
    raise ValueError("output_format must be json, text, srt, or vtt")


def get_converter(config: str):
    """Return an OpenCC converter, or None when OpenCC is missing or the config is invalid."""
    if config not in _converters:
        try:
            import opencc

            _converters[config] = opencc.OpenCC(config)
        except Exception:
            _converters[config] = None
    return _converters[config]


def to_traditional(result: dict, config: str) -> dict:
    """Convert Chinese text to Traditional Chinese. A disabled config leaves the result unchanged.

    Whisper's Chinese training data is mostly Simplified, so recognition comes
    back in Simplified characters. Conversion runs after recognition.
    """
    if config.strip().lower() in ZH_CONVERT_OFF:
        return result
    converter = get_converter(config.strip())
    if converter is None:
        return result

    convert = converter.convert
    result["text"] = convert(result["text"])
    for segment in result["segments"]:
        segment["text"] = convert(segment["text"])
        for word in segment.get("words", []):
            word["word"] = convert(word["word"])
    result["zh_convert"] = config.strip()
    return result


def get_model(model: str, device: str, compute_type: str):
    key = (model, device, compute_type)
    cached = _models.get(key)
    if cached is None:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            missing = exc.name or "a dependency"
            raise ModuleNotFoundError(
                f"missing dependency {missing}. Install the project requirements first: pip install -r requirements.txt"
            ) from exc
        if DOWNLOAD_ROOT:
            Path(DOWNLOAD_ROOT).mkdir(parents=True, exist_ok=True)
        cached = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=DOWNLOAD_ROOT,
        )
        _models[key] = cached
    return cached


@dataclass
class Transcript:
    rendered: str
    output_path: Path | None
    language: str
    duration: float
    segments: int


def transcribe(
    audio_path: Path,
    *,
    language: str,
    task: str,
    model: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    initial_prompt: str,
    output_format: str,
    output_path: Path | None,
    zh_convert: str,
) -> Transcript:
    source = audio_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio/video file not found: {source}")
    if task not in {"transcribe", "translate"}:
        raise ValueError("task must be transcribe or translate")
    fmt = output_format.lower().strip()
    if fmt not in FORMATS:
        raise ValueError("output_format must be json, text, srt, or vtt")
    if not 1 <= beam_size <= 20:
        raise ValueError("beam_size must be between 1 and 20")

    selected_model = model.strip() or DEFAULT_MODEL
    selected_device = device.strip() or DEFAULT_DEVICE
    selected_compute = compute_type.strip() or DEFAULT_COMPUTE_TYPE
    whisper = get_model(selected_model, selected_device, selected_compute)
    iterator, info = whisper.transcribe(
        str(source),
        language=language.strip() or None,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt.strip() or None,
    )
    segments = []
    parts = []
    for segment in iterator:
        item = {
            "id": segment.id,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text,
        }
        if word_timestamps and segment.words:
            item["words"] = [
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "word": word.word,
                    "probability": round(word.probability, 5),
                }
                for word in segment.words
            ]
        segments.append(item)
        parts.append(segment.text)

    result = {
        "source": str(source),
        "model": selected_model,
        "task": task,
        "language": info.language,
        "language_probability": round(info.language_probability, 5),
        "duration": round(info.duration, 3),
        "text": "".join(parts).strip(),
        "segments": segments,
    }
    # English translation output is left as English. Conversion only applies
    # to Chinese transcription.
    if task == "transcribe" and info.language == "zh":
        result = to_traditional(result, zh_convert.strip() or DEFAULT_ZH_CONVERT)

    rendered = render(result, fmt)
    written = None
    if output_path is not None:
        written = output_path.expanduser().resolve()
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text(rendered, encoding="utf-8")
    return Transcript(
        rendered=rendered,
        output_path=written,
        language=info.language,
        duration=round(info.duration, 3),
        segments=len(segments),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stdout and args.output is not None:
        parser.error("--stdout and --output cannot be used together")
    if not 1 <= args.beam_size <= 20:
        parser.error("beam_size must be between 1 and 20")

    multiple = len(args.audio) > 1
    fmt = infer_format(args.output_format, args.output)
    if args.stdout and multiple and fmt == "json":
        parser.error("JSON on stdout supports a single file; use --output with a directory for several files")

    missing = [path for path in args.audio if not path.expanduser().is_file()]
    if missing:
        for path in missing:
            print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    planned: list[tuple[Path, Path | None]]
    if args.stdout:
        planned = [(path.expanduser().resolve(), None) for path in args.audio]
    else:
        try:
            planned = plan_outputs(args.audio, fmt, args.output)
        except ValueError as exc:
            parser.error(str(exc))

    failures = 0
    for index, (source, dest) in enumerate(planned):
        print(f"{args.task} {source.name} ({fmt})", file=sys.stderr, flush=True)
        try:
            transcript = transcribe(
                source,
                language=args.language,
                task=args.task,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                vad_filter=args.vad_filter,
                word_timestamps=args.word_timestamps,
                initial_prompt=args.initial_prompt,
                output_format=fmt,
                output_path=dest,
                zh_convert=args.zh_convert,
            )
        except ModuleNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: {source}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if transcript.output_path is None:
            if index:
                sys.stdout.write("\n")
            text = transcript.rendered
            sys.stdout.write(text if text.endswith("\n") else text + "\n")
            continue

        print(
            f"wrote {transcript.output_path} | {transcript.language} | {transcript.duration}s | {transcript.segments} segments",
            file=sys.stderr,
        )

    if failures:
        print(f"done: {len(planned) - failures} ok, {failures} failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
