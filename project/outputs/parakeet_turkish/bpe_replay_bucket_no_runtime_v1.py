"""Exact retained BucketCycler/ids_sha bodies, without numerical imports."""
import hashlib,math

def ids_sha(rows):
    return hashlib.sha256("\n".join(sorted(row["source_id"] for row in rows)).encode()).hexdigest()

class BucketCycler:
    """Shuffled cycles, sorted only inside bounded shuffled reservoirs.

    Each cycle contains every source exactly once. Two-dimensional log-length
    bins avoid pairing long-T/short-U with short-T/long-U by scalar cost alone.
    """
    def __init__(self, rows, rng, bucket_size=128):
        self.by_id = {row["source_id"]: row for row in rows}
        if len(self.by_id) != len(rows) or not rows or bucket_size < 1:
            raise ValueError("Each positive sampler pool needs unique IDs and positive bucket size")
        if any(int(row["encoder_frames"]) < 1 or int(row["target_tokens"]) < 1 for row in rows):
            raise ValueError("Length buckets require positive encoder frames and target tokens")
        self.rows = list(rows)
        self.rng = rng
        self.bucket_size = bucket_size
        self.position = len(rows)
        self.draws = 0
        self.cycles_started = 0
        self.seen = set()
        self.pool_sha256 = ids_sha(rows)

    @staticmethod
    def length_key(row):
        t = int(row["encoder_frames"])
        u = int(row["target_tokens"])
        return (int(math.log2(t)), int(math.log2(u)), t, u, row["source_id"])

    def next(self):
        if self.position == len(self.rows):
            self.rng.shuffle(self.rows)
            for start in range(0, len(self.rows), self.bucket_size):
                self.rows[start:start+self.bucket_size] = sorted(self.rows[start:start+self.bucket_size], key=self.length_key)
            self.position = 0
            self.cycles_started += 1
        row = self.rows[self.position]
        self.position += 1
        self.draws += 1
        self.seen.add(row["source_id"])
        return row

    def state_dict(self):
        return {"pool_sha256": self.pool_sha256, "order": [row["source_id"] for row in self.rows],
                "position": self.position, "draws": self.draws, "cycles_started": self.cycles_started,
                "seen": sorted(self.seen), "bucket_size": self.bucket_size}

    def load_state_dict(self, state):
        if (state["pool_sha256"] != self.pool_sha256 or state["bucket_size"] != self.bucket_size
                or len(state["order"]) != len(self.rows) or set(state["order"]) != set(self.by_id)):
            raise ValueError("Exact-resume sampler pool/bucket identity changed")
        if not 0 <= state["position"] <= len(self.rows) or not set(state["seen"]).issubset(self.by_id):
            raise ValueError("Invalid sampler cursor or seen IDs")
        n = len(self.rows)
        cycles = state["cycles_started"]
        expected_draws = 0 if cycles == 0 else (cycles-1)*n + state["position"]
        expected_seen = (set() if cycles == 0 else set(state["order"][:state["position"]])
                         if cycles == 1 else set(self.by_id))
        if (cycles < 0 or state["draws"] != expected_draws or set(state["seen"]) != expected_seen
                or (cycles == 0 and state["position"] != n)):
            raise ValueError("Exact-resume sampler cycle/exposure counters disagree")
        self.rows = [self.by_id[source] for source in state["order"]]
        self.position, self.draws, self.cycles_started = state["position"], state["draws"], state["cycles_started"]
        self.seen = set(state["seen"])
