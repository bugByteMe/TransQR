from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
import zipfile
import zlib
from pathlib import Path

import qrcode
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)
from qrcode.exceptions import DataOverflowError

from qrtransfer_protocol import FramePayload, pack_frame

ERROR_CORRECTION_LEVELS = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}


def create_zip_archive(input_folder: Path) -> bytes:
    input_folder = input_folder.resolve()
    if not input_folder.is_dir():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    root_folder = input_folder.name
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(f"{root_folder}/", b"")

        for path in sorted(input_folder.rglob("*")):
            relative = path.relative_to(input_folder)
            archive_path = (Path(root_folder) / relative).as_posix()

            if path.is_dir():
                archive.writestr(f"{archive_path}/", b"")
            else:
                archive.write(path, archive_path)

    return buffer.getvalue()


def chunk_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def make_qr_image(payload: str, error_correction: int, box_size: int, border: int):
    qr = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode a folder into QR code frames.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Folder to encode",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where QR frame PNG files are written",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Raw bytes per QR frame before base64 wrapping (default: 700)",
    )
    parser.add_argument(
        "--error-correction",
        choices=tuple(ERROR_CORRECTION_LEVELS.keys()),
        default="M",
        help="QR error correction level (default: M)",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=10,
        help="QR box size in pixels (default: 10)",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=4,
        help="QR border width in boxes (default: 4)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    if args.box_size <= 0:
        parser.error("--box-size must be a positive integer")
    if args.border < 0:
        parser.error("--border cannot be negative")
    if not args.input_folder.is_dir():
        parser.error(f"--input-folder is not a directory: {args.input_folder}")

    output_dir: Path = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_bytes = create_zip_archive(args.input_folder)
    chunks = chunk_bytes(archive_bytes, args.chunk_size)
    if not chunks:
        chunks = [b""]

    transfer_id = uuid.uuid4().hex
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    root_folder = args.input_folder.resolve().name
    error_correction = ERROR_CORRECTION_LEVELS[args.error_correction]
    frame_payload_records: list[dict[str, str]] = []

    for index, chunk in enumerate(chunks):
        frame_payload = FramePayload(
            transfer_id=transfer_id,
            index=index,
            total_chunks=len(chunks),
            chunk_crc32=f"{zlib.crc32(chunk) & 0xFFFFFFFF:08x}",
            archive_sha256=archive_sha256,
            archive_size=len(archive_bytes),
            root_folder=root_folder,
            chunk_bytes=chunk,
        )
        qr_payload = pack_frame(frame_payload)

        try:
            qr_image = make_qr_image(
                payload=qr_payload,
                error_correction=error_correction,
                box_size=args.box_size,
                border=args.border,
            )
        except DataOverflowError as exc:
            chunk_number = index + 1
            raise SystemExit(
                f"Frame {chunk_number} exceeds QR capacity. "
                "Use a smaller --chunk-size or lower --error-correction."
            ) from exc

        frame_name = f"{transfer_id}_{index + 1:05d}.png"
        with (output_dir / frame_name).open("wb") as frame_file:
            qr_image.save(frame_file)
        frame_payload_records.append({"frame": frame_name, "payload": qr_payload})

    payload_index_path = output_dir / f"{transfer_id}_payloads.jsonl"
    with payload_index_path.open("w", encoding="utf-8") as payload_index_file:
        for record in frame_payload_records:
            payload_index_file.write(json.dumps(record, separators=(",", ":")))
            payload_index_file.write("\n")

    manifest = {
        "transfer_id": transfer_id,
        "source_folder": str(args.input_folder.resolve()),
        "root_folder": root_folder,
        "total_frames": len(chunks),
        "chunk_size": args.chunk_size,
        "archive_size": len(archive_bytes),
        "archive_sha256": archive_sha256,
        "error_correction": args.error_correction,
        "payload_index": payload_index_path.name,
    }
    manifest_path = output_dir / f"{transfer_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Transfer ID: {transfer_id}")
    print(f"Output directory: {output_dir}")
    print(f"Frames created: {len(chunks)}")
    print(f"Archive bytes: {len(archive_bytes)}")
    print(f"Archive SHA256: {archive_sha256}")
    print(f"Payload index: {payload_index_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
