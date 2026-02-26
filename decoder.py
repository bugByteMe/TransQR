from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qrtransfer_protocol import FramePayload, unpack_frame

zxingcpp: Any | None = None
try:
    import zxingcpp as _zxingcpp

    zxingcpp = _zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class TransferAssembly:
    transfer_id: str | None = None
    total_chunks: int | None = None
    archive_sha256: str | None = None
    archive_size: int | None = None
    root_folder: str | None = None
    chunks: dict[int, bytes] = field(default_factory=dict)
    duplicates: int = 0
    ignored_foreign: int = 0

    def ingest(self, frame: FramePayload) -> str:
        if self.transfer_id is None:
            self.transfer_id = frame.transfer_id
            self.total_chunks = frame.total_chunks
            self.archive_sha256 = frame.archive_sha256
            self.archive_size = frame.archive_size
            self.root_folder = frame.root_folder
        else:
            if frame.transfer_id != self.transfer_id:
                self.ignored_foreign += 1
                return "foreign"

            metadata_matches = (
                frame.total_chunks == self.total_chunks
                and frame.archive_sha256 == self.archive_sha256
                and frame.archive_size == self.archive_size
                and frame.root_folder == self.root_folder
            )
            if not metadata_matches:
                raise ValueError("Frame metadata does not match transfer metadata")

        existing = self.chunks.get(frame.index)
        if existing is not None:
            if existing != frame.chunk_bytes:
                raise ValueError(f"Chunk {frame.index} payload conflict")
            self.duplicates += 1
            return "duplicate"

        self.chunks[frame.index] = frame.chunk_bytes
        return "new"

    def is_started(self) -> bool:
        return self.transfer_id is not None

    def is_complete(self) -> bool:
        return self.total_chunks is not None and len(self.chunks) == self.total_chunks

    def missing_indices(self) -> list[int]:
        if self.total_chunks is None:
            return []
        return [index for index in range(self.total_chunks) if index not in self.chunks]

    def assemble_archive(self) -> bytes:
        if not self.is_complete():
            raise ValueError("Transfer is incomplete")

        assert self.total_chunks is not None
        assert self.archive_sha256 is not None
        assert self.archive_size is not None

        archive_bytes = b"".join(self.chunks[index] for index in range(self.total_chunks))
        if len(archive_bytes) != self.archive_size:
            raise ValueError(
                f"Archive size mismatch: expected {self.archive_size}, got {len(archive_bytes)}"
            )

        digest = hashlib.sha256(archive_bytes).hexdigest()
        if digest != self.archive_sha256:
            raise ValueError("Archive SHA256 mismatch")

        return archive_bytes


def _deduplicate_texts(texts: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _decode_with_zxing(image) -> list[str]:
    if not HAS_ZXING or zxingcpp is None:
        return []

    try:
        results = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=False,
            try_downscale=True,
            try_invert=True,
        )
    except Exception:
        return []

    texts = [result.text for result in results if result.text]
    if texts:
        return _deduplicate_texts(texts)

    try:
        pure_results = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=False,
            try_downscale=True,
            try_invert=True,
            is_pure=True,
        )
    except Exception:
        return []

    return _deduplicate_texts([result.text for result in pure_results if result.text])


def _decode_with_cv(detector: cv2.QRCodeDetector, candidate_image) -> list[str]:
    texts: list[str] = []

    try:
        single_result = detector.detectAndDecode(candidate_image)
        single_text = single_result[0] if isinstance(single_result, tuple) else single_result
        if isinstance(single_text, str) and single_text:
            texts.append(single_text)
    except cv2.error:
        pass

    try:
        multi_result = detector.detectAndDecodeMulti(candidate_image)
    except cv2.error:
        return _deduplicate_texts(texts)

    if not isinstance(multi_result, tuple):
        return _deduplicate_texts(texts)

    decoded_candidate = multi_result[1] if len(multi_result) > 1 else None
    if isinstance(decoded_candidate, (list, tuple)):
        for text in decoded_candidate:
            if isinstance(text, str) and text:
                texts.append(text)

    return _deduplicate_texts(texts)


def decode_qr_texts(detector: cv2.QRCodeDetector, image) -> list[str]:
    if image is None or getattr(image, "size", 0) == 0:
        return []

    zxing_texts = _decode_with_zxing(image)
    if zxing_texts:
        return zxing_texts

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    quick_candidates = [image, gray]
    height, width = gray.shape[:2]
    for scale in (0.9, 0.75, 0.6, 0.5, 1.2, 1.5):
        scaled_width = int(round(width * scale))
        scaled_height = int(round(height * scale))
        if scaled_width < 80 or scaled_height < 80:
            continue

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        quick_candidates.append(
            cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=interpolation,
            )
        )

    for candidate in quick_candidates:
        quick_texts = _decode_with_cv(detector, candidate)
        if quick_texts:
            return quick_texts

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        2,
    )
    inverted_gray = cv2.bitwise_not(gray)

    fallback_candidates = [gray, otsu, adaptive, inverted_gray]
    scales = [1.0, 0.9, 0.8, 0.75, 0.7, 0.66, 0.6, 0.55, 0.5, 0.45, 1.2, 1.5]

    for candidate in fallback_candidates:
        for scale in scales:
            if scale == 1.0:
                resized = candidate
            else:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                resized = cv2.resize(candidate, None, fx=scale, fy=scale, interpolation=interpolation)

            fallback_texts = _decode_with_cv(detector, resized)
            if fallback_texts:
                return fallback_texts

    return []


def consume_decoded_text(
    text: str,
    state: TransferAssembly,
    source: str,
    verbose: bool,
) -> None:
    try:
        frame = unpack_frame(text)
    except ValueError as exc:
        if verbose:
            print(f"[WARN] {source}: {exc}")
        return

    if frame is None:
        return

    status = state.ingest(frame)
    if status == "new":
        assert state.total_chunks is not None
        print(
            f"[OK] Received chunk {frame.index + 1}/{state.total_chunks} "
            f"({len(state.chunks)}/{state.total_chunks}) from {source}"
        )
    elif status == "duplicate" and verbose:
        print(f"[INFO] Duplicate chunk {frame.index + 1} from {source}")
    elif status == "foreign" and verbose:
        print(f"[INFO] Ignored frame from another transfer at {source}")


def decode_from_image_folder(images_dir: Path, state: TransferAssembly, verbose: bool) -> None:
    if not images_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {images_dir}")

    image_paths = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )
    if not image_paths:
        raise ValueError(f"No image files found in: {images_dir}")

    detector = cv2.QRCodeDetector()
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            if verbose:
                print(f"[WARN] Could not read image: {image_path}")
            continue

        texts = decode_qr_texts(detector, image)
        if not texts and verbose:
            print(f"[INFO] No QR code found in: {image_path.name}")

        for text in texts:
            consume_decoded_text(
                text=text,
                state=state,
                source=image_path.name,
                verbose=verbose,
            )

        if state.is_complete():
            break


def decode_from_camera(
    camera_index: int,
    state: TransferAssembly,
    max_seconds: int | None,
    verbose: bool,
) -> None:
    detector = cv2.QRCodeDetector()
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    deadline = time.monotonic() + max_seconds if max_seconds is not None else None
    print("Scanning from camera. Present QR codes to the webcam.")
    print("Press Ctrl+C to stop scanning.")

    try:
        while not state.is_complete():
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.05)
                continue

            texts = decode_qr_texts(detector, frame)
            for text in texts:
                consume_decoded_text(
                    text=text,
                    state=state,
                    source=f"camera:{camera_index}",
                    verbose=verbose,
                )

            if deadline is not None and time.monotonic() >= deadline:
                print("[WARN] Camera scan timed out.")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Camera scan interrupted.")
    finally:
        camera.release()


def _build_screen_capture_area(
    monitor_bounds: dict[str, int],
    region: tuple[int, int, int, int] | None,
) -> dict[str, int]:
    if region is None:
        return {
            "left": monitor_bounds["left"],
            "top": monitor_bounds["top"],
            "width": monitor_bounds["width"],
            "height": monitor_bounds["height"],
        }

    x, y, width, height = region
    if x < 0 or y < 0:
        raise ValueError("--screen-region X and Y must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError("--screen-region width and height must be positive")
    if x + width > monitor_bounds["width"] or y + height > monitor_bounds["height"]:
        raise ValueError("--screen-region exceeds selected monitor bounds")

    return {
        "left": monitor_bounds["left"] + x,
        "top": monitor_bounds["top"] + y,
        "width": width,
        "height": height,
    }


def decode_from_screen(
    monitor_index: int,
    screen_region: tuple[int, int, int, int] | None,
    screen_fps: float,
    state: TransferAssembly,
    max_seconds: int | None,
    verbose: bool,
) -> None:
    try:
        from mss import mss
    except ImportError as exc:
        raise RuntimeError(
            "Screen capture mode requires the 'mss' package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    detector = cv2.QRCodeDetector()
    capture_interval = 1.0 / screen_fps

    with mss() as screen_capture:
        monitors = screen_capture.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise ValueError(
                f"Invalid --screen-monitor {monitor_index}. "
                f"Available range: 0..{len(monitors) - 1}"
            )

        if verbose:
            for index, monitor in enumerate(monitors):
                print(
                    f"[INFO] Monitor {index}: "
                    f"left={monitor['left']} top={monitor['top']} "
                    f"width={monitor['width']} height={monitor['height']}"
                )

        monitor_bounds = monitors[monitor_index]
        capture_area = _build_screen_capture_area(monitor_bounds, screen_region)
        deadline = time.monotonic() + max_seconds if max_seconds is not None else None

        print(
            "Scanning from screen in realtime. "
            f"Monitor {monitor_index}, region "
            f"({capture_area['left']},{capture_area['top']},"
            f"{capture_area['width']}x{capture_area['height']})."
        )
        print("Press Ctrl+C to stop scanning.")

        try:
            while not state.is_complete():
                loop_started = time.monotonic()

                frame_bgra = np.asarray(screen_capture.grab(capture_area))
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                texts = decode_qr_texts(detector, frame_bgr)

                for text in texts:
                    consume_decoded_text(
                        text=text,
                        state=state,
                        source=f"screen:{monitor_index}",
                        verbose=verbose,
                    )

                if deadline is not None and time.monotonic() >= deadline:
                    print("[WARN] Screen scan timed out.")
                    break

                remaining = capture_interval - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("\n[INFO] Screen scan interrupted.")


def _safe_member_path(output_dir: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    member_path = Path(normalized)

    if not normalized:
        raise ValueError("Archive contains an empty member path")
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe path in archive: {member_name}")

    output_root = output_dir.resolve()
    target = (output_root / member_path).resolve()
    try:
        target.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes output directory: {member_name}") from exc
    return target


def extract_archive(archive_bytes: bytes, output_dir: Path) -> list[str]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_items: set[str] = set()

    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        for info in archive.infolist():
            member_path = Path(info.filename.replace("\\", "/"))
            if member_path.parts:
                root_items.add(member_path.parts[0])

            target_path = _safe_member_path(output_dir, info.filename)
            is_directory = info.is_dir() or info.filename.endswith("/")

            if is_directory:
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, mode="r") as source_file, target_path.open("wb") as output_file:
                shutil.copyfileobj(source_file, output_file)

    return sorted(root_items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode QR transfer frames from images, webcam, or live screen capture.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--images",
        type=Path,
        help="Directory containing QR frame images",
    )
    source_group.add_argument(
        "--camera",
        type=int,
        help="Camera index for webcam scanning",
    )
    source_group.add_argument(
        "--screen",
        action="store_true",
        help="Capture screen in realtime and scan QR codes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where decoded files are extracted",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Optional timeout for webcam or screen scanning",
    )
    parser.add_argument(
        "--screen-monitor",
        type=int,
        default=1,
        help="Monitor index used with --screen (default: 1)",
    )
    parser.add_argument(
        "--screen-region",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help=(
            "Optional region within selected monitor for --screen "
            "(coordinates are monitor-relative)"
        ),
    )
    parser.add_argument(
        "--screen-fps",
        type=float,
        default=8.0,
        help="Capture rate for --screen mode in frames per second (default: 8)",
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

    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be a positive integer")
    if args.screen_fps <= 0:
        parser.error("--screen-fps must be a positive number")
    if args.screen_region is not None and (args.screen_region[2] <= 0 or args.screen_region[3] <= 0):
        parser.error("--screen-region width and height must be positive")

    state = TransferAssembly()

    if args.images is not None:
        decode_from_image_folder(
            images_dir=args.images,
            state=state,
            verbose=args.verbose,
        )
    elif args.screen:
        decode_from_screen(
            monitor_index=args.screen_monitor,
            screen_region=tuple(args.screen_region) if args.screen_region is not None else None,
            screen_fps=args.screen_fps,
            state=state,
            max_seconds=args.max_seconds,
            verbose=args.verbose,
        )
    else:
        assert args.camera is not None
        decode_from_camera(
            camera_index=args.camera,
            state=state,
            max_seconds=args.max_seconds,
            verbose=args.verbose,
        )

    if not state.is_started():
        raise SystemExit("No valid transfer frames were detected.")

    if not state.is_complete():
        missing = state.missing_indices()
        missing_display = ", ".join(str(index + 1) for index in missing[:20])
        if len(missing) > 20:
            missing_display += ", ..."
        raise SystemExit(
            "Transfer incomplete. "
            f"Received {len(state.chunks)}/{state.total_chunks} chunks. "
            f"Missing chunk numbers: {missing_display}"
        )

    archive_bytes = state.assemble_archive()
    root_items = extract_archive(archive_bytes=archive_bytes, output_dir=args.output)

    print("Transfer complete.")
    print(f"Transfer ID: {state.transfer_id}")
    print(f"Duplicates ignored: {state.duplicates}")
    print(f"Frames from other transfers ignored: {state.ignored_foreign}")
    print(f"Output directory: {args.output.resolve()}")
    if root_items:
        print(f"Top-level extracted item(s): {', '.join(root_items)}")


if __name__ == "__main__":
    main()
