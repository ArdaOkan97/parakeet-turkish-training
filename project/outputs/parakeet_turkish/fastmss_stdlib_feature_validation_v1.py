"""Read-only bounded validation of owned NumPy-save feature files.

The controller deliberately has no NumPy runtime. This module validates the
producer's float32/bool NPY bytes without allocating arrays or changing them.
Native take_row still uses the frozen NumPy array reader, independently.
"""
import ast
import hashlib
import math
from pathlib import Path
import re
import stat
import struct
import sys

CHUNK_BYTES = 64 * 1024
MAXIMUM_HEADER_BYTES = 1024
MAXIMUM_FILE_BYTES = 512 * 1024**2
MAGIC = b"\x93NUMPY"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=1, maximum=MAXIMUM_FILE_BYTES):
    require(type(value) is int and minimum <= value <= maximum,
            "Exact bounded nonboolean NPY count required")
    return value


def _signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _header(stream):
    """Parse only the bounded, non-executable producer NPY1/2 header."""
    prefix = stream.read(8)
    require(len(prefix) == 8 and prefix[:6] == MAGIC
            and prefix[6:] in (b"\x01\x00", b"\x02\x00"),
            "Owned np.save NPY1/2 magic/version required")
    size_bytes = 2 if prefix[6] == 1 else 4
    size_raw = stream.read(size_bytes)
    require(len(size_raw) == size_bytes, "Truncated NPY header length")
    length = int.from_bytes(size_raw, "little")
    _integer(length, maximum=MAXIMUM_HEADER_BYTES)
    raw = stream.read(length)
    require(len(raw) == length and raw.endswith(b"\n"),
            "Complete bounded newline-terminated NPY header required")
    try:
        tree = ast.parse(raw.decode("latin1").strip(), mode="eval")
        require(isinstance(tree.body, ast.Dict)
                and len(tree.body.keys) == 3
                and sum(1 for _ in ast.walk(tree)) <= 64,
                "Exact bounded NPY dictionary syntax required")
        keys = [ast.literal_eval(key) for key in tree.body.keys]
        require(all(type(key) is str for key in keys)
                and len(set(keys)) == 3
                and set(keys) == {"descr", "fortran_order", "shape"},
                "Unique exact NPY header fields required")
        value = ast.literal_eval(tree)
    except (SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError("Invalid non-executable NPY header") from exc
    require(type(value) is dict and value["fortran_order"] is False
            and type(value["descr"]) is str and type(value["shape"]) is tuple,
            "C-order primitive dtype and tuple shape required")
    return value, prefix + size_raw + raw


def _scan(item, *, dtype, shape, folder, owned_names, guard, scan):
    require(type(item) is dict and type(item["file"]) is dict
            and set(item["file"]) == {"path", "bytes", "sha256"}
            and item["dtype"] == dtype and type(item["shape"]) is list
            and item["shape"] == list(shape), "Exact feature descriptor required")
    path = Path(item["file"]["path"]).resolve(strict=True)
    require(path.parent == folder and path.name in owned_names
            and re.fullmatch(r"u\d{2}_r\d{2}_(features|mask)\.npy", path.name),
            "Exact owned feature path required")
    require(type(item["file"]["path"]) is str
            and str(path) == item["file"]["path"], "Canonical resolved file binding required")
    _integer(item["file"]["bytes"])
    for value in (item["file"]["sha256"], item["array_sha256"]):
        require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
                "Exact file and raw-array SHA256 required")
    width, descriptor = (4, "<f4") if dtype == "float32" else (1, "|b1")
    require(sys.byteorder == "little", "Pinned little-endian producer runtime required")
    data_bytes = math.prod(shape) * width
    _integer(data_bytes)
    guard()
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == item["file"]["bytes"],
            "Actual owned file size/type differs")
    full, raw = hashlib.sha256(), hashlib.sha256()
    maximum_read = 0
    with path.open("rb") as stream:
        require(_signature(before) == _signature(path.stat()), "File changed before NPY read")
        header, header_raw = _header(stream)
        require(header["descr"] == descriptor
                and header["shape"] == tuple(shape)
                and all(type(value) is int and value > 0 for value in header["shape"])
                and stream.tell() + data_bytes == before.st_size,
                "NPY dtype/shape/complete byte length differs; no allocation")
        full.update(header_raw)
        offset = 0
        while offset < data_bytes:
            guard()
            block = stream.read(min(CHUNK_BYTES, data_bytes - offset))
            require(0 < len(block) <= CHUNK_BYTES and len(block) % width == 0,
                    "Truncated or misaligned NPY payload")
            raw.update(block); full.update(block)
            scan(block, offset)
            maximum_read = max(maximum_read, len(block)); offset += len(block)
        require(offset == data_bytes and not stream.read(1), "Exact NPY payload EOF required")
    require(_signature(before) == _signature(path.stat())
            and full.hexdigest() == item["file"]["sha256"]
            and raw.hexdigest() == item["array_sha256"],
            "Owned NPY file/raw-byte SHA or identity changed")
    return maximum_read


def validate_feature_arrays(entry, *, folder, owned_names, guard):
    """Validate the same feature/mask invariants as _Store.arrays, no arrays.

    The producer saves little-endian float32 and canonical bool bytes. Padded
    float values may be positive OR negative zero, matching NumPy.any behavior.
    This does not prove a child/process exit, commit a draw or publish a window.
    """
    folder = Path(folder).resolve(strict=True)
    require(folder.is_relative_to(Path(r"E:\gpu_training_scratchpad").resolve()),
            "E-only owned fixture/feature namespace required")
    require(type(owned_names) is list and len(owned_names) == len(set(owned_names)) <= 704
            and all(type(name) is str and Path(name).name == name for name in owned_names),
            "Exact bounded safe owned names required")
    require(callable(guard), "Actual caller guard required")
    features, mask = entry["input_features"], entry["attention_mask"]
    shape = features["shape"]
    require(type(shape) is list and len(shape) == 3
            and all(type(value) is int and value > 0 for value in shape)
            and shape[0] == 1 and shape[2] == 128
            and mask["shape"] == shape[:2]
            and all(type(value) is int and value > 0 for value in mask["shape"]),
            "Original [1,physical_frames,128] features and [1,physical_frames] mask required")
    frames = _integer(shape[1])
    encoder_frames = _integer(entry["encoder_frames"])
    valid = 0
    padding_seen = False

    def mask_scan(block, _offset):
        nonlocal valid, padding_seen
        for value in block:
            require(value in (0, 1), "Canonical bool mask bytes0/1 required")
            if value:
                require(not padding_seen, "Actual mask must be one right-padded prefix")
                valid += 1
            else:
                padding_seen = True

    mask_read = _scan(mask, dtype="bool", shape=(1, frames), folder=folder,
                      owned_names=owned_names, guard=guard, scan=mask_scan)
    require(0 < valid <= frames and (valid + 7) // 8 == encoder_frames,
            "Nonempty original valid-mask/subsampled encoder geometry required")
    padding_offset = valid * 128 * 4

    def feature_scan(block, offset):
        for index, (bits,) in enumerate(struct.iter_unpack("<I", block)):
            require(bits & 0x7f800000 != 0x7f800000, "All original float32 features must be finite")
            if offset + index * 4 >= padding_offset:
                require(bits & 0x7fffffff == 0, "Original masked padding must be numerically zero")

    feature_read = _scan(features, dtype="float32", shape=tuple(shape), folder=folder,
                         owned_names=owned_names, guard=guard, scan=feature_scan)
    return {"kind": "FastMSS_stdlib_owned_NPY_validation_only_v1", "passed": True,
            "physical_feature_frames": frames, "valid_mask_frames": valid,
            "encoder_frames": encoder_frames, "maximum_payload_read_bytes": max(mask_read, feature_read),
            "NumPy_imported_here": False, "waveform_or_feature_bytes_changed": False,
            "process_exit_or_window_completion_attested": False,
            "sampler_or_optimizer_committed": False}
