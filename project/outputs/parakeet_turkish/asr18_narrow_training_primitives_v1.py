"""Exact retained initialization/sampler bodies; no unused CLI imports."""
import hashlib,json,math
import torch
from torch.utils.checkpoint import checkpoint
from extend_vocab import _make_tokenizer
from bpe_full_decoder_optimizer_v1 import TABLES,require

def checkpoint_encoder_blocks(model):
    """Recompute activations without enabling dropout or BatchNorm updates."""
    for layer in model.encoder.layers:
        original = layer.forward
        def wrapped(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)
        layer.forward = wrapped

def expand_candidate(model, processor, source):
    """Preserve all old rows and RNG; explicitly move duration-value rows."""
    from tsc_bpe_acquisition_transition_v1 import candidate_assets
    from tokenizers import Tokenizer
    candidate, donors = candidate_assets()
    embedding, head = model.decoder.embedding, model.joint.head
    require(model.config.vocab_size == 8199 and embedding.num_embeddings == 8199
            and head.out_features == 8204 and model.config.blank_token_id == 8192,
            "Original lexical/blank/duration geometry required")
    rng = (torch.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()])
    try:
        new_embedding = torch.nn.Embedding(8711, embedding.embedding_dim, padding_idx=embedding.padding_idx,
                                           device=embedding.weight.device, dtype=embedding.weight.dtype)
        new_head = torch.nn.Linear(head.in_features, 8716, device=head.weight.device, dtype=head.weight.dtype)
    finally:
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
    with torch.no_grad():
        new_embedding.weight[:8199].copy_(embedding.weight)
        new_head.weight[:8199].copy_(head.weight[:8199]); new_head.bias[:8199].copy_(head.bias[:8199])
        new_head.weight[8711:].copy_(head.weight[8199:]); new_head.bias[8711:].copy_(head.bias[8199:])
        for donor in donors:
            index, leaves = donor["HF_id"], donor["original_leaf_donor_ids"]
            new_embedding.weight[index].copy_(source["weights"][TABLES[0]][leaves].mean(0))
            new_head.weight[index].copy_(source["weights"][TABLES[1]][leaves].mean(0))
            new_head.bias[index].copy_(source["weights"][TABLES[2]][leaves].mean() - 2.)
    model.decoder.embedding, model.joint.head = new_embedding, new_head
    model.config.vocab_size = model.joint.vocab_size = 8711
    if hasattr(model.decoder, "vocab_size"): model.decoder.vocab_size = 8711
    if hasattr(model.generation_config, "vocab_size"): model.generation_config.vocab_size = 8711
    suppression = {}
    for field in ("suppress_tokens", "begin_suppress_tokens"):
        old = getattr(model.generation_config, field, None)
        require(old is None or all(type(v) is int and 0 <= v < 8204 for v in old), "Unknown original suppression")
        new = None if old is None else [v + 512 if 8199 <= v < 8204 else v for v in old]
        require(not any(8199 <= v < 8711 for v in new or []), "New BPE rows must be unsuppressed")
        setattr(model.generation_config, field, new)
        suppression[field] = {"before": old, "after": new}
    processor.tokenizer = _make_tokenizer(processor.tokenizer,
        Tokenizer.from_str(json.dumps(candidate, ensure_ascii=False)))
    return {"vocabulary": 8711, "duration_rows": [8711, 8716], "blank": 8192,
            "new512_unsuppressed": True, "suppression": suppression,
            "initialization": "Original leaf donor means; bias mean minus 2",
            "torch_RNG_preserved": torch.equal(torch.get_rng_state(), rng[0]) and
                all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng[1]))}

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
