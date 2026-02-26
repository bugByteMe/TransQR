from __future__ import annotations

import base64
import binascii
import json
import re
import zlib
from dataclasses import dataclass

FRAME_PREFIX = "QRT1:"
PROTOCOL_VERSION = 1

_CRC_RE = re.compile(r"^[0-9a-f]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FramePayload:
    transfer_id: str
    index: int
    total_chunks: int
    chunk_crc32: str
    archive_sha256: str
    archive_size: int
    root_folder: str
    chunk_bytes: bytes


def pack_frame(payload: FramePayload) -> str:
    frame_obj = {
        "v": PROTOCOL_VERSION,
        "id": payload.transfer_id,
        "i": payload.index,
        "n": payload.total_chunks,
        "c": payload.chunk_crc32,
        "h": payload.archive_sha256,
        "b": payload.archive_size,
        "f": payload.root_folder,
        "d": base64.b64encode(payload.chunk_bytes).decode("ascii"),
    }
    return FRAME_PREFIX + json.dumps(frame_obj, separators=(",", ":"), sort_keys=True)


def unpack_frame(frame_text: str) -> FramePayload | None:
    if not frame_text.startswith(FRAME_PREFIX):
        return None

    try:
        frame_obj = json.loads(frame_text[len(FRAME_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid frame JSON") from exc

    required_keys = {"v", "id", "i", "n", "c", "h", "b", "f", "d"}
    missing_keys = required_keys - set(frame_obj.keys())
    if missing_keys:
        missing_display = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing frame keys: {missing_display}")

    if frame_obj["v"] != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {frame_obj['v']}")

    transfer_id = frame_obj["id"]
    index = frame_obj["i"]
    total_chunks = frame_obj["n"]
    chunk_crc32 = frame_obj["c"]
    archive_sha256 = frame_obj["h"]
    archive_size = frame_obj["b"]
    root_folder = frame_obj["f"]
    payload_base64 = frame_obj["d"]

    if not isinstance(transfer_id, str) or not transfer_id:
        raise ValueError("Invalid transfer id")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("Invalid chunk index")
    if not isinstance(total_chunks, int) or isinstance(total_chunks, bool) or total_chunks <= 0:
        raise ValueError("Invalid total chunk count")
    if index >= total_chunks:
        raise ValueError("Chunk index out of bounds")
    if not isinstance(chunk_crc32, str) or not _CRC_RE.match(chunk_crc32):
        raise ValueError("Invalid chunk CRC32")
    if not isinstance(archive_sha256, str) or not _SHA256_RE.match(archive_sha256):
        raise ValueError("Invalid archive SHA256")
    if not isinstance(archive_size, int) or isinstance(archive_size, bool) or archive_size < 0:
        raise ValueError("Invalid archive size")
    if not isinstance(root_folder, str) or not root_folder:
        raise ValueError("Invalid root folder name")
    if "/" in root_folder or "\\" in root_folder or root_folder in {".", ".."}:
        raise ValueError("Invalid root folder name")
    if not isinstance(payload_base64, str):
        raise ValueError("Invalid chunk payload")

    try:
        chunk_bytes = base64.b64decode(payload_base64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("Invalid base64 payload") from exc

    computed_crc32 = f"{zlib.crc32(chunk_bytes) & 0xFFFFFFFF:08x}"
    if computed_crc32 != chunk_crc32:
        raise ValueError("Chunk CRC32 mismatch")

    return FramePayload(
        transfer_id=transfer_id,
        index=index,
        total_chunks=total_chunks,
        chunk_crc32=chunk_crc32,
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        root_folder=root_folder,
        chunk_bytes=chunk_bytes,
    )
