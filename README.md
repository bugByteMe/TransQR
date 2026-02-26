# QR Code Folder Transfer

This project provides three scripts:

- `encoder.py`: encodes a folder into multiple QR code images.
- `decoder.py`: decodes frames from an image folder, webcam, or live screen capture, then restores the folder.
- `viewer.py`: shows QR frame images in the terminal one by one.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Encode a Folder

```bash
python encoder.py --input-folder ./my_folder --out-dir ./qr_frames
```

Useful options:

- `--chunk-size` raw bytes per frame (default: `700`)
- `--error-correction` one of `L`, `M`, `Q`, `H` (default: `M`)

If encoding fails with a QR capacity error, lower `--chunk-size`.

The encoder also writes a `*_payloads.jsonl` index (one payload per frame), which helps
the CLI viewer avoid image re-decoding misses.

## Decode From an Image Folder

```bash
python decoder.py --images ./qr_frames --output ./restored
```

## Decode From Webcam

```bash
python decoder.py --camera 0 --output ./restored
```

## Decode From Live Screen Capture

```bash
python decoder.py --screen --output ./restored
```

Useful screen options:

- `--screen-monitor` monitor index (default: `1`, `0` means virtual all-monitors view)
- `--screen-region X Y WIDTH HEIGHT` monitor-relative crop for faster scanning
- `--screen-fps` capture rate (default: `8`)

Optional webcam timeout:

```bash
python decoder.py --camera 0 --output ./restored --max-seconds 180
```

Optional screen timeout:

```bash
python decoder.py --screen --output ./restored --max-seconds 180
```

## View QR Frames In CLI

Manual stepping mode:

```bash
python viewer.py --folder ./qr_frames
```

Autoplay mode:

```bash
python viewer.py --folder ./qr_frames --delay 0.5
```

The viewer uses `pyqrcode`: it reads payloads from `*_payloads.jsonl` when present,
otherwise decodes image frames (`zxing-cpp` first, OpenCV fallback), then re-renders QR in terminal.

Useful options:

- `--start` start from a 1-based frame number
- `--loop` keep looping frames in autoplay mode
- `--quiet-zone` set quiet zone modules around the rendered QR
- `--max-frames` stop after rendering N frames
- `--no-clear` keep previous frame output in terminal

## Notes

- The encoder packages the input folder into a ZIP archive before chunking.
- Every frame includes metadata and CRC32 to validate each chunk.
- The decoder verifies full archive SHA256 before extraction.
- Extraction is path-safe and rejects unsafe archive paths.
- Screen mode uses `mss`; on macOS, grant Screen Recording permission to the terminal/app running Python.
