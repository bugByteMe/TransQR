from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

try:
    import pyqrcode
except ImportError as exc:
    raise SystemExit(
        "viewer.py requires 'pyqrcode'. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

zxingcpp: Any | None = None
Image: Any | None = None
try:
    import zxingcpp as _zxingcpp
    from PIL import Image as _PILImage

    zxingcpp = _zxingcpp
    Image = _PILImage
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_frame_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Frame folder does not exist: {folder}")

    frame_paths = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )

    if not frame_paths:
        raise ValueError(f"No image frames found in: {folder}")

    return frame_paths


def clear_terminal() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def load_payload_index(folder: Path) -> dict[str, str]:
    payload_index: dict[str, str] = {}
    payload_index_paths = sorted(folder.glob("*_payloads.jsonl"))

    for payload_index_path in payload_index_paths:
        for line_number, line in enumerate(payload_index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid payload index JSON at {payload_index_path.name}:{line_number}"
                ) from exc

            frame_name = record.get("frame")
            payload = record.get("payload")
            if isinstance(frame_name, str) and isinstance(payload, str) and frame_name and payload:
                payload_index[frame_name] = payload

    return payload_index


def _decode_with_zxing(image_path: Path) -> str | None:
    if not HAS_ZXING or zxingcpp is None or Image is None:
        return None

    with Image.open(image_path) as image:
        results = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=False,
            try_downscale=True,
            try_invert=True,
        )
        for result in results:
            if result.text:
                return result.text

        pure_results = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=False,
            try_downscale=True,
            try_invert=True,
            is_pure=True,
        )
        for result in pure_results:
            if result.text:
                return result.text

    return None


def _decode_with_cv(detector: cv2.QRCodeDetector, image_path: Path) -> str | None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        2,
    )
    inverted = cv2.bitwise_not(gray)

    base_candidates = [image, gray, otsu, adaptive, inverted]
    scales = [1.0, 0.9, 0.8, 0.75, 0.7, 0.66, 0.6, 0.55, 0.5, 0.45, 1.2, 1.5]

    for candidate in base_candidates:
        for scale in scales:
            if scale == 1.0:
                resized = candidate
            else:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                resized = cv2.resize(candidate, None, fx=scale, fy=scale, interpolation=interpolation)

            try:
                single_result = detector.detectAndDecode(resized)
                single_text = single_result[0] if isinstance(single_result, tuple) else single_result
                if single_text:
                    return single_text
            except cv2.error:
                pass

            try:
                multi_result = detector.detectAndDecodeMulti(resized)
            except cv2.error:
                continue

            if not isinstance(multi_result, tuple):
                continue

            maybe_texts: list[str] = []
            texts_candidate = multi_result[1] if len(multi_result) > 1 else None
            if isinstance(texts_candidate, (list, tuple)):
                maybe_texts = [text for text in texts_candidate if isinstance(text, str)]

            for text in maybe_texts:
                if text:
                    return text

    return None


def decode_qr_payload(detector: cv2.QRCodeDetector, image_path: Path) -> str:
    zxing_text = _decode_with_zxing(image_path)
    if zxing_text:
        return zxing_text

    cv_text = _decode_with_cv(detector, image_path)
    if cv_text:
        return cv_text

    raise ValueError(f"No QR code detected in frame: {image_path.name}")


def render_qr_terminal(payload: str, quiet_zone: int) -> str:
    qr_object = pyqrcode.create(payload, error="M")
    return qr_object.terminal(
        module_color="black",
        background="white",
        quiet_zone=quiet_zone,
    )


def view_frames(
    frame_paths: list[Path],
    start_index: int,
    delay: float,
    loop: bool,
    no_clear: bool,
    quiet_zone: int,
    max_frames: int | None,
    payload_index: dict[str, str],
    verbose: bool,
) -> None:
    detector = cv2.QRCodeDetector()
    index = start_index
    auto_mode = delay > 0
    shown_frames = 0

    if not auto_mode and not sys.stdin.isatty():
        raise RuntimeError("Manual mode needs an interactive terminal; use --delay for autoplay")

    while True:
        if max_frames is not None and shown_frames >= max_frames:
            break

        if index >= len(frame_paths):
            if loop:
                index = 0
            else:
                break

        frame_path = frame_paths[index]
        if not no_clear:
            clear_terminal()

        payload = payload_index.get(frame_path.name)
        if payload is None:
            try:
                payload = decode_qr_payload(detector, frame_path)
            except ValueError as exc:
                print(f"[WARN] {exc}")
                index += 1
                if auto_mode:
                    continue
                user_text = input("Press Enter for next frame, or 'q' to quit: ").strip().lower()
                if user_text in {"q", "quit", "exit"}:
                    break
                continue

        if payload is None:
            index += 1
            continue

        qr_render = render_qr_terminal(payload, quiet_zone=quiet_zone)

        print(f"Frame {index + 1}/{len(frame_paths)}: {frame_path.name}")
        if verbose and frame_path.name in payload_index:
            print("[INFO] payload source: index")
        elif verbose:
            print("[INFO] payload source: decoded")
        print(qr_render)
        shown_frames += 1

        if auto_mode:
            time.sleep(delay)
            index += 1
            continue

        user_text = input("\nPress Enter for next frame, 'p' previous, 'q' quit: ").strip().lower()
        if user_text in {"q", "quit", "exit"}:
            break
        if user_text in {"p", "prev", "previous"}:
            index = max(0, index - 1)
            continue

        index += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display QR image frames from a folder in CLI one by one.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing QR frame images",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based frame number to start from (default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Autoplay delay in seconds; 0 enables manual step mode (default: 0)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop frames continuously",
    )
    parser.add_argument(
        "--quiet-zone",
        type=int,
        default=4,
        help="Quiet zone modules around QR code (default: 4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to show",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear terminal between frames",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.start <= 0:
        parser.error("--start must be a positive integer")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.quiet_zone < 0:
        parser.error("--quiet-zone cannot be negative")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")

    frame_paths = collect_frame_paths(args.folder)
    payload_index = load_payload_index(args.folder)

    if args.verbose:
        source = "present" if payload_index else "not found"
        print(f"[INFO] payload index: {source}")
        if args.quiet_zone < 4:
            print("[WARN] quiet-zone below 4 modules can reduce scanner reliability")
        if HAS_ZXING:
            print("[INFO] zxing-cpp decoder: available")
        else:
            print("[INFO] zxing-cpp decoder: unavailable (using OpenCV fallback)")

    start_index = args.start - 1
    if start_index >= len(frame_paths):
        parser.error(
            f"--start is {args.start}, but only {len(frame_paths)} frame(s) were found"
        )

    try:
        view_frames(
            frame_paths=frame_paths,
            start_index=start_index,
            delay=args.delay,
            loop=args.loop,
            no_clear=args.no_clear,
            quiet_zone=args.quiet_zone,
            max_frames=args.max_frames,
            payload_index=payload_index,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
