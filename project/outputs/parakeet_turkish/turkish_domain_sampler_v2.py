"""Checkpointable TRAIN-domain sampler with replay independent of TR draws.

The uniform comparator and domain30 arm use the same replay RNG/stream order.
Neither is claimed to reproduce the legacy sampler's interleaved shared RNG.
No ML imports, data reads, or changes to the frozen v1 sampler.
"""
from __future__ import annotations

from fractions import Fraction
import math
import random

from turkish_domain_sampler_v1 import LANGUAGES, fingerprint, turkish_domain


class DomainBalancedBucketSampler:
    def __init__(self, tr, replay, *, seed, replay_seed, sampling_mode, batch_size,
                 accumulation, replay_fraction=.5, fleurs_fraction=.3,
                 bucket_size=128, cycler_factory, cycler_source_sha256):
        if sampling_mode not in ("domain30", "uniform"):
            raise ValueError("Explicit domain30 or independent-replay uniform comparator required")
        if type(seed) is not int or type(replay_seed) is not int:
            raise ValueError("Explicit integer Turkish/replay seeds required")
        self.tr_rng = random.Random(seed + 1)
        self.replay_rng = random.Random(replay_seed)
        self.sampling_mode = sampling_mode
        self.batch_size, self.accumulation = batch_size, accumulation
        self.effective_batch = batch_size * accumulation
        self.replay_per_update = round(self.effective_batch * replay_fraction)
        if (batch_size < 1 or accumulation < 1 or not math.isclose(self.replay_per_update,
                self.effective_batch * replay_fraction, abs_tol=1e-9)
                or not 0 <= self.replay_per_update < self.effective_batch):
            raise ValueError("Integer replay quota and positive Turkish batch required")
        self.tr_per_update = self.effective_batch - self.replay_per_update
        self.fleurs_ratio = Fraction(str(fleurs_fraction))
        if self.fleurs_ratio != Fraction(3, 10):
            raise ValueError("This isolated scientific arm fixes FLEURS to 30% of Turkish clips")
        if len(cycler_source_sha256) != 64:
            raise ValueError("Bind exact BucketCycler source SHA")
        self.tr_by_id = {row["source_id"]: row for row in tr}
        if not tr or len(self.tr_by_id) != len(tr):
            raise ValueError("Unique nonempty Turkish source IDs required")
        by_domain = {name: [] for name in ("fleurs", "cv")}
        self.source_domains = {}
        for row in tr:
            domain = turkish_domain(row)
            by_domain[domain].append(row)
            self.source_domains[row["source_id"]] = domain
        if any(not rows for rows in by_domain.values()):
            raise ValueError("Both vetted Turkish TRAIN domains required")
        self.domain_pool_sizes = {name: len(rows) for name, rows in by_domain.items()}
        streams = by_domain if sampling_mode == "domain30" else {"uniform": tr}
        self.turkish_streams = {name: cycler_factory(rows, self.tr_rng, bucket_size)
                                for name, rows in streams.items()}
        if (len({row["source_id"] for row in replay}) != len(replay)
                or set(self.tr_by_id).intersection(row["source_id"] for row in replay)
                or any(row.get("language") not in LANGUAGES for row in replay)):
            raise ValueError("Unique disjoint genuine old-language replay sources required")
        self.replay = {lang: cycler_factory([row for row in replay if row["language"] == lang],
                                           self.replay_rng, bucket_size) for lang in LANGUAGES}
        self.updates = 0
        self.replay_language_counts = {lang: 0 for lang in LANGUAGES}
        self.domain_draws = {name: 0 for name in by_domain}
        self.domain_seen = {name: set() for name in by_domain}
        self.domain_audio_seconds = {name: 0. for name in by_domain}
        self.identity = {"sampling_mode": sampling_mode, "turkish_seed": seed,
            "replay_seed": replay_seed, "replay_rng_independent_of_turkish": True,
            "batch_size": batch_size, "accumulation": accumulation,
            "replay_per_update": self.replay_per_update,
            "fleurs_fraction_numerator": 3, "fleurs_fraction_denominator": 10,
            "bucket_size": bucket_size, "cycler_source_sha256": cycler_source_sha256,
            "turkish_pool_fingerprint": fingerprint(tr), "replay_pool_fingerprint": fingerprint(replay)}

    @property
    def tr_draws(self):
        return sum(self.domain_draws.values())

    @property
    def tr_unique(self):
        return sum(len(seen) for seen in self.domain_seen.values())

    @property
    def turkish_epoch(self):
        """Evaluation/epoch clock: minimum per-domain exposure for domain30."""
        if self.sampling_mode == "uniform":
            return self.tr_draws / len(self.tr_by_id)
        return min(self.domain_draws[name] / count for name, count in self.domain_pool_sizes.items())

    def fleurs_quota_at(self, draws):
        return draws * 3 // 10

    def next_batches(self):
        if self.sampling_mode == "uniform":
            tr = [self.turkish_streams["uniform"].next() for _ in range(self.tr_per_update)]
        else:
            before = self.updates * self.tr_per_update
            nf = self.fleurs_quota_at(before + self.tr_per_update) - self.fleurs_quota_at(before)
            tr = [self.turkish_streams[name].next() for name, count in
                  (("fleurs", nf), ("cv", self.tr_per_update - nf)) for _ in range(count)]
            # Same selected-row sorting as reviewed v1; no source discarded.
            tr.sort(key=lambda row: (int(math.log2(int(row["encoder_frames"]))),
                int(math.log2(int(row["target_tokens"]))), int(row["encoder_frames"]),
                int(row["target_tokens"]), row["source_id"]))
        for row in tr:
            domain = self.source_domains[row["source_id"]]
            self.domain_draws[domain] += 1
            self.domain_seen[domain].add(row["source_id"])
            self.domain_audio_seconds[domain] += float(row["duration"])
        replay = []
        for _ in range(self.replay_per_update):
            lang = self.replay_rng.choice(list(self.replay))
            replay.append(self.replay[lang].next())
            self.replay_language_counts[lang] += 1
        batches, remainder = [], []
        for stream in (tr, replay):
            full = len(stream) // self.batch_size * self.batch_size
            batches.extend(stream[start:start+self.batch_size] for start in range(0, full, self.batch_size))
            remainder.extend(stream[full:])
        if remainder:
            if len(remainder) != self.batch_size:
                raise RuntimeError("One complete mixed remainder batch required")
            batches.append(remainder)
        if len(batches) != self.accumulation or any(len(batch) != self.batch_size for batch in batches):
            raise RuntimeError("Effective optimizer batch changed")
        self.updates += 1
        return batches

    def coverage(self):
        domains = {name: {"draws": self.domain_draws[name],
            "unique_clips_seen": len(self.domain_seen[name]), "pool_clips": count,
            "repeat_visits": self.domain_draws[name] - len(self.domain_seen[name]),
            "fraction_unique_seen": len(self.domain_seen[name]) / count,
            "audio_seen_seconds": self.domain_audio_seconds[name]}
            for name, count in self.domain_pool_sizes.items()}
        return {"domain": domains, "turkish_draws": self.tr_draws,
            "turkish_unique_clips_seen": self.tr_unique,
            "turkish_draw_equivalent_epochs": self.tr_draws / len(self.tr_by_id),
            "minimum_domain_exposure_epochs": min(self.domain_draws[name] / count
                                                   for name, count in self.domain_pool_sizes.items()),
            "epoch_clock": "minimum_domain_exposure" if self.sampling_mode == "domain30" else "uniform_draw_equivalent",
            "all_Turkish_pool_sources_seen": all(len(self.domain_seen[name]) == count
                                                 for name, count in self.domain_pool_sizes.items()),
            "full_pool_pass_claim_requires_all_sources_seen": True,
            "domain_ratio_is_clip_weighted": True}

    def updates_for_epochs(self, epochs):
        if not math.isfinite(epochs) or epochs <= 0:
            raise ValueError("Positive finite domain exposure required")
        if self.sampling_mode == "uniform":
            return math.ceil(epochs * len(self.tr_by_id) / self.tr_per_update)
        nf, nc = (math.ceil(epochs * self.domain_pool_sizes[name]) for name in ("fleurs", "cv"))
        low, high = 0, math.ceil(max(nf / .3, nc / .7) / self.tr_per_update) + 2
        while low < high:
            middle = (low + high) // 2
            draws = middle * self.tr_per_update
            actual_f = self.fleurs_quota_at(draws)
            if actual_f >= nf and draws - actual_f >= nc:
                high = middle
            else:
                low = middle + 1
        return low

    def state_dict(self):
        return {"schema": 2, "kind": "Turkish_domain_independent_replay_sampler_v2",
            "identity": self.identity.copy(), "turkish_rng": self.tr_rng.getstate(),
            "replay_rng": self.replay_rng.getstate(), "updates": self.updates,
            "turkish_streams": {name: stream.state_dict() for name, stream in self.turkish_streams.items()},
            "replay": {name: stream.state_dict() for name, stream in self.replay.items()},
            "replay_language_counts": self.replay_language_counts.copy(),
            "domain_draws": self.domain_draws.copy(),
            "domain_seen": {name: sorted(seen) for name, seen in self.domain_seen.items()},
            "domain_audio_seconds": self.domain_audio_seconds.copy()}

    def load_state_dict(self, state):
        if (state.get("schema") != 2 or state.get("kind") != "Turkish_domain_independent_replay_sampler_v2"
                or state.get("identity") != self.identity):
            raise ValueError("Exact domain/replay seeds, source pools and layout required")
        updates = state["updates"]
        if type(updates) is not int or updates < 0:
            raise ValueError("Nonnegative completed updates required")
        if list(state["turkish_streams"]) != list(self.turkish_streams) or list(state["replay"]) != list(self.replay):
            raise ValueError("Stream order changed")
        for name, stream in self.turkish_streams.items():
            stream.load_state_dict(state["turkish_streams"][name])
        for name, stream in self.replay.items():
            stream.load_state_dict(state["replay"][name])
        if sum(stream.draws for stream in self.turkish_streams.values()) != updates * self.tr_per_update:
            raise ValueError("TR draws disagree with update count")
        if (sum(stream.draws for stream in self.replay.values()) != updates * self.replay_per_update
                or state["replay_language_counts"] != {name: stream.draws for name, stream in self.replay.items()}):
            raise ValueError("Replay quotas/counters disagree")
        expected_seen = {name: set() for name in self.domain_pool_sizes}
        for stream in self.turkish_streams.values():
            for source in stream.seen:
                expected_seen[self.source_domains[source]].add(source)
        if state["domain_seen"] != {name: sorted(seen) for name, seen in expected_seen.items()}:
            raise ValueError("Domain unique counters disagree with actual stream seen IDs")
        draws = state["domain_draws"]
        if set(draws) != set(self.domain_pool_sizes) or any(type(n) is not int or n < 0 for n in draws.values()):
            raise ValueError("Invalid domain draw counters")
        if sum(draws.values()) != updates * self.tr_per_update:
            raise ValueError("Domain draws disagree with updates")
        if self.sampling_mode == "domain30":
            expected_f = self.fleurs_quota_at(updates * self.tr_per_update)
            if draws != {"fleurs": expected_f, "cv": updates * self.tr_per_update - expected_f}:
                raise ValueError("30/70 deterministic domain quotas changed")
        else:
            stream = self.turkish_streams["uniform"]
            complete_cycles = max(0, stream.cycles_started - 1)
            prefix = stream.rows[:stream.position] if stream.cycles_started else []
            expected_draws = {name: complete_cycles * count for name, count in self.domain_pool_sizes.items()}
            for row in prefix:
                expected_draws[self.source_domains[row["source_id"]]] += 1
            if draws != expected_draws:
                raise ValueError("Uniform domain counts disagree with actual shuffled cycles")
        audio = state["domain_audio_seconds"]
        if set(audio) != set(self.domain_pool_sizes) or any(not math.isfinite(n) or n < 0 for n in audio.values()):
            raise ValueError("Invalid domain audio counters")
        self.tr_rng.setstate(state["turkish_rng"])
        self.replay_rng.setstate(state["replay_rng"])
        self.updates = updates
        self.domain_draws = draws.copy()
        self.domain_seen = expected_seen
        self.domain_audio_seconds = audio.copy()
        self.replay_language_counts = state["replay_language_counts"].copy()
