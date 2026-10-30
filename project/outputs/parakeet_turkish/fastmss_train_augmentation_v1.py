"""Opt-in, length-preserving single-target augmentation at the audio_reader hook.

This is our adapter, not the upstream FastMSS meeting simulator. It accepts
reviewed FastMSS-generated RIRs and separately licensed noise. No Torch/Lhotse,
GPU, disk waveform cache, global RNG, new speakers, speed changes or labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

STORAGE = Path(r"E:\gpu_training_scratchpad")
UPSTREAM = "e481ceac821ef0cb31033bf43b45922e377f528d"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def storage_path(value):
    path = Path(value).resolve(strict=True)
    require(path.is_relative_to(STORAGE.resolve(strict=True)), "Asset/registry must be on designated E storage")
    return path


@dataclass(frozen=True)
class AugmentationPolicy:
    enabled: bool = False
    seed: int = 20261017
    sample_rate: int = 16000
    phases: tuple[str, ...] = ("weak_label_continued_training",)
    apply_probability: float = 0.5
    noise_probability: float = 0.5
    rir_probability: float = 0.5
    noise_mode: str = "assets"  # assets or synthetic_gaussian; no automatic fallback
    snr_db: tuple[float, float] = (10.0, 25.0)
    max_waveform_seconds: float = 120.0
    max_rir_seconds: float = 1.5
    peak_ceiling: float = 0.99
    registry_path: str | None = None
    max_asset_bytes: int = 32 * 1024 * 1024
    max_total_asset_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        require(type(self.enabled) is bool and type(self.seed) is int, "Explicit enabled boolean/integer seed required")
        require(self.sample_rate == 16000, "Current training audio hook is fixed 16 kHz")
        require(self.phases and len(set(self.phases)) == len(self.phases) and
                all(v in ("weak_label_continued_training", "cleaner_data_fine_tuning") for v in self.phases), "Explicit training phases required")
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in
                    (self.apply_probability, self.noise_probability, self.rir_probability)), "Invalid augmentation probability")
        require(self.noise_mode in ("assets", "synthetic_gaussian"), "Unknown noise mode")
        require(len(self.snr_db) == 2 and all(math.isfinite(v) for v in self.snr_db) and
                0 <= self.snr_db[0] <= self.snr_db[1] <= 60, "SNR must be within 0..60 dB")
        require(math.isfinite(self.max_waveform_seconds) and 0 < self.max_waveform_seconds <= 120, "Bounded input duration required")
        require(math.isfinite(self.max_rir_seconds) and 0 < self.max_rir_seconds <= 1.5, "Bounded RIR duration required")
        require(math.isfinite(self.peak_ceiling) and 0 < self.peak_ceiling <= 1, "Peak ceiling must be within (0,1]")
        require(0 < self.max_asset_bytes <= 32 * 1024 * 1024 and
                self.max_asset_bytes <= self.max_total_asset_bytes <= 64 * 1024 * 1024, "Asset byte limits exceed declared bounds")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for key in ("phases", "snr_db"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)  # reject misspelled/unknown settings


class TrainWaveformAugmenter:
    def __init__(self, policy: AugmentationPolicy, *, guard: Callable[[], None], fixture_mode=False):
        self.policy = policy
        self.guard = guard  # caller retains its admitted system/private/I/O guards
        self.assets = {"noise": [], "rir": []}
        self.registry_sha256 = None
        if policy.enabled and policy.registry_path is not None:
            path = storage_path(policy.registry_path)
            require(path.stat().st_size <= 256 * 1024, "Oversized asset registry")
            raw = path.read_bytes()
            self.registry_sha256 = hashlib.sha256(raw).hexdigest()
            registry = json.loads(raw)
            require(registry.get("schema") == 1 and len(registry["assets"]) <= 128, "Invalid/big asset registry")
            total = 0
            for asset in registry["assets"]:
                require(asset["kind"] in self.assets, "Asset must be noise or RIR")
                require(asset.get("training_use_confirmed") is True and
                        asset.get("license") and asset.get("provenance"), "Explicit asset rights/provenance required")
                require(not asset.get("fixture_only", False) or fixture_mode, "Fixture assets are forbidden in production")
                require(asset["sample_rate"] == policy.sample_rate, "Assets must already be 16 kHz; no hidden resampling")
                asset_path = storage_path(asset["path"])
                size = asset_path.stat().st_size
                total += size
                require(0 < size <= policy.max_asset_bytes and total <= policy.max_total_asset_bytes, "Asset disk/verification bound exceeded")
                digest = hashlib.sha256()
                with asset_path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self.guard()
                        digest.update(chunk)
                require(digest.hexdigest() == asset["sha256"], "Asset SHA256 mismatch")
                stat = asset_path.stat()
                self.assets[asset["kind"]].append({**asset, "path": str(asset_path),
                    "verified_size": stat.st_size, "verified_mtime_ns": stat.st_mtime_ns})
        if policy.enabled:
            require(not policy.rir_probability or self.assets["rir"], "RIR enabled without reviewed assets")
            require(not policy.noise_probability or policy.noise_mode == "synthetic_gaussian" or self.assets["noise"],
                    "Real background noise enabled without reviewed assets")
        self.policy_sha256 = hashlib.sha256(canonical({"adapter": 1, "policy": asdict(policy),
            "registry_sha256": self.registry_sha256, "upstream_revision": UPSTREAM})).hexdigest()

    def _asset_audio(self, kind, samples, rng):
        import numpy as np
        import soundfile as sf
        self.guard()
        asset = self.assets[kind][int(rng.integers(len(self.assets[kind])))]
        path = storage_path(asset["path"])
        stat = path.stat()
        require(stat.st_size == asset["verified_size"] and stat.st_mtime_ns == asset["verified_mtime_ns"],
                "Asset changed after hash admission")
        with sf.SoundFile(path) as source:
            require(source.samplerate == self.policy.sample_rate and source.channels == 1,
                    "Asset header must be mono 16 kHz")
            require(source.frames > 0, "Empty audio asset")
            asset_frames = source.frames
            if kind == "rir":
                require(source.frames <= math.ceil(self.policy.max_rir_seconds * source.samplerate), "RIR exceeds duration bound")
                data = source.read(dtype="float32")
                offset = 0
            else:
                if source.frames >= samples:
                    offset = int(rng.integers(source.frames - samples + 1))
                    source.seek(offset)
                    data = source.read(samples, dtype="float32")
                else:
                    offset = int(rng.integers(source.frames))
                    data = source.read(dtype="float32")
        require(np.isfinite(data).all() and data.size and float(np.max(np.abs(data))) > 0, "Invalid/silent asset")
        if kind == "noise" and data.size < samples:
            # Circular wrapping is bounded, deterministic, and logged; no padding silence.
            data = np.resize(data, samples + offset)[offset:]
        self.guard()
        return data, {"id": asset["id"], "sha256": asset["sha256"], "start_sample": offset,
                      "wrapped": bool(kind == "noise" and asset_frames < samples)}

    def apply(self, waveform, row: Mapping, *, split: str, phase: str,
              role: str, epoch: int, draw_index: int):
        """Return (waveform, bounded receipt); labels/rows are never modified.

        epoch/draw_index are the caller's durable sampler position, not a new RNG
        or training clock. Repeated calls at the same position are byte-identical.
        """
        base = {"policy_sha256": self.policy_sha256, "applied": False,
                "source_id": row.get("source_id"), "label_policy": "unchanged_single_target"}
        if not self.policy.enabled:
            return waveform, {**base, "reason": "disabled"}
        if split != "train" or phase not in self.policy.phases or role != "student_tr" or row.get("language") != "tr":
            return waveform, {**base, "reason": "protected_split_phase_role_or_language"}
        require(row.get("source_id") and type(epoch) is int and type(draw_index) is int and epoch >= 0 and draw_index >= 0,
                "Durable source ID and nonnegative epoch/draw position required")
        import numpy as np
        require(isinstance(waveform, np.ndarray) and waveform.dtype == np.float32 and waveform.ndim == 1,
                "Decoded mono float32 waveform required")
        require(waveform.size and np.isfinite(waveform).all(), "Invalid source waveform")
        if waveform.size > math.ceil(self.policy.max_waveform_seconds * self.policy.sample_rate):
            return waveform, {**base, "reason": "augmentation_duration_bound_clean_source_retained"}
        if not np.any(waveform):
            return waveform, {**base, "reason": "silent_clean_source_retained"}
        identity = {"seed": self.policy.seed, "policy": self.policy_sha256,
                    "source_id": row["source_id"], "epoch": epoch, "draw_index": draw_index}
        seed = int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:16], "little")
        rng = np.random.Generator(np.random.PCG64(seed))
        base = {**base, "epoch": epoch, "draw_index": draw_index, "augmentation_seed": str(seed)}
        if rng.random() >= self.policy.apply_probability:
            return waveform, {**base, "reason": "probability_clean_control"}
        self.guard()
        output = waveform.copy()
        operations = []
        if rng.random() < self.policy.rir_probability:
            from scipy.signal import fftconvolve
            rir, receipt = self._asset_audio("rir", waveform.size, rng)
            direct = int(np.argmax(np.abs(rir)))
            rir = rir[direct:]  # upstream-compatible direct-arrival alignment
            rir = rir / np.sqrt(np.mean(rir.astype(np.float64) ** 2) * rir.size)
            original_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2)))
            output = fftconvolve(output, rir, mode="full")[:waveform.size].astype(np.float32)
            new_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2)))
            require(new_rms > 0, "RIR removed all speech")
            output *= np.float32(original_rms / new_rms)
            operations.append({"kind": "rir", **receipt, "direct_offset_removed_samples": direct,
                               "tail_policy": "crop_after_original_sample_count", "rms_restored": True})
        if rng.random() < self.policy.noise_probability:
            if self.policy.noise_mode == "assets":
                noise, receipt = self._asset_audio("noise", waveform.size, rng)
            else:
                noise = rng.standard_normal(waveform.size, dtype=np.float32)
                receipt = {"id": "synthetic_gaussian", "not_environmental_recording": True}
            # Measured on complete supplied clip; not a VAD/speech-only SNR claim.
            noise = noise - np.mean(noise, dtype=np.float64)
            noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
            speech_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2)))
            require(noise_rms > 0, "Zero-variance noise asset")
            snr = float(rng.uniform(*self.policy.snr_db))
            gain = speech_rms / (noise_rms * 10 ** (snr / 20))
            output += noise * np.float32(gain)
            operations.append({"kind": "noise", **receipt, "requested_clip_rms_snr_db": snr,
                               "noise_gain": gain})
        if not operations:
            return waveform, {**base, "reason": "operation_probability_clean_control"}
        before_peak = float(np.max(np.abs(output)))
        scale = min(1.0, self.policy.peak_ceiling / before_peak) if before_peak else 1.0
        output *= np.float32(scale)
        require(output.shape == waveform.shape and output.dtype == np.float32 and np.isfinite(output).all(),
                "Augmented waveform geometry/finite contract violated")
        self.guard()
        return output, {**base, "applied": True, "operations": operations, "peak_scale": scale,
                        "samples_unchanged": waveform.size, "sample_rate": self.policy.sample_rate,
                        "final_peak": float(np.max(np.abs(output))), "speed_or_speaker_mixing": False}

    def audio_reader(self, original_reader: Callable, *, context_for: Callable,
                     record: Callable[[dict], None]):
        """Factory for existing collate_training_batch(..., audio_reader=...).

        context_for(row) must supply split/phase/role/epoch/draw_index from the
        admitted durable batch draw metadata. record should stream receipts;
        this module never accumulates them or writes in training directories.
        """
        def read(row):
            original = original_reader(row)
            waveform, receipt = self.apply(original, row, **context_for(row))
            record(receipt)
            return waveform
        return read
