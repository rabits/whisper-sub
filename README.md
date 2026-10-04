# transcribe.py

Transcribe local audio and video files with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

I found there are not much simple apps to create subtitles, so made this one. It uses CPU and can
process large files, but that takes ~1/3 of duration (20min per 1hr of audio).

The script writes a transcript next to the audio file. Plain text is the default. Use `-f srt`, `-f vtt`, or `-f json` for subtitles or structured output. The first run downloads the selected model from Hugging Face.

```bash
python3 transcribe.py interview.mp3 -l en -o interview.srt
```

## CPU

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 transcribe.py interview.mp3 -l en -m small --device cpu --compute-type int8 -o interview.srt
```

On CPU, `small` with `--compute-type int8` is the practical default. `tiny` and `base` are faster and less accurate. `medium` and `large-v3` are slower.

`--device auto` uses a CUDA device when the installed CTranslate2 build can see one, and otherwise uses the CPU. The wheel from PyPI is the CUDA build, so an AMD GPU stays unused in this virtualenv. Use the ROCm image below for AMD.

## Options

`python transcribe.py --help` lists every flag.

| Flag | Meaning |
|---|---|
| `-o, --output` | Output file, or a directory for several inputs. Default: next to each input. |
| `-f, --format` | `text` (default), `srt`, `vtt`, or `json`. A known extension on `--output` selects the format. |
| `-t, --task` | `transcribe` or `translate`. Translation produces English. |
| `-l, --language` | Language code such as `ru`, `en`, `zh`, or `ja`. Empty means auto-detect. |
| `-m, --model` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`, or `turbo`. |
| `--device` | `auto`, `cpu`, or `cuda`. ROCm uses `cuda`. |
| `--compute-type` | `int8` on CPU. `float16` or `int8_float16` on GPU. |
| `--beam-size` | Search width from 1 to 20. Default: 5. |
| `--vad-filter` / `--no-vad-filter` | Drop silent stretches. On by default. |
| `--word-timestamps` | Include per-word times in `json` output. |
| `--initial-prompt` | Names and domain terms that should bias recognition. |
| `--zh-convert` | Chinese script conversion. See below. |
| `--stdout` | Print the transcript instead of writing a file. |

## Environment

| Variable | Default | Effect |
|---|---|---|
| `WHISPER_MODEL` | `small` | Model used when `--model` is omitted. |
| `WHISPER_DEVICE` | `auto` | Device used when `--device` is omitted. |
| `WHISPER_COMPUTE_TYPE` | `default` | Compute type used when `--compute-type` is omitted. |
| `WHISPER_DOWNLOAD_ROOT` | Hugging Face cache | Directory for downloaded models. The Docker images set this to `/work/.cache/whisper`. |
| `WHISPER_ZH_CONVERT` | `s2twp` | Default Chinese conversion. |

## Traditional Chinese

Whisper's Chinese training data is overwhelmingly Simplified, so Chinese transcripts come back in Simplified characters. When the detected language is `zh` and the task is transcription, the script converts the text with OpenCC.

| Value | Result |
|---|---|
| `s2twp` (default) | Traditional Chinese with Taiwan vocabulary |
| `s2tw` | Traditional characters, original wording |
| `s2t` | Generic Traditional Chinese |
| `off` | Leave the transcript unchanged |

Translation to English is left unchanged. If OpenCC is missing, the transcript is returned as the model produced it.
