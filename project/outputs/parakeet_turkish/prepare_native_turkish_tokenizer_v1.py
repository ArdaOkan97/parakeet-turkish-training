"""Offline SentencePiece extension for the first direct, native NeMo pilot.

This script imports SentencePiece/protobuf only. It never imports Torch or NeMo.
The native base initialization is distinct from the learned HF adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DONORS = {"ğ": "g", "Ğ": "G", "ı": "i", "İ": "I", "ş": "s", "Ş": "S"}
CONTROLS = [
    "Hello world. I've called 42 people; their names are Alice and Michael.",
    "¿Cómo estás? El niño pidió una canción y café para mañana.",
    "L'Italia è bella: perché abbiamo già un'altra città?",
    "Bonjour à tous. Noël, cœur, français, été et garçon: 123.",
    "Die Straße führt über Köln. Öl, Größe, grüße und zwölf Wörter.",
    "çÇöÖüÜ abc ABC xyz XYZ 0123456789",
    "",
]
TURKISH = ["Işık İstanbul şehri dağların aşağısında", "Türkçe çığ yağış çağrı ışık", "Ğğ Iı İi Şş Çç Öö Üü"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source_dir, destination_dir, manifest_path):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    source_dir, destination_dir, manifest_path = map(Path, (source_dir, destination_dir, manifest_path))
    if destination_dir.exists() or manifest_path.exists():
        raise FileExistsError("Use a fresh destination; original assets are immutable.")
    source_model, = source_dir.glob("*_tokenizer.model")
    source_vocab, = source_dir.glob("*_tokenizer.vocab")
    source_text_vocab, = source_dir.glob("*_vocab.txt")
    original = pb.ModelProto.FromString(source_model.read_bytes())
    if len(original.pieces) != 8192 or original.trainer_spec.model_type != 2:
        raise ValueError("Expected the pinned native Parakeet-v3 BPE tokenizer with 8192 pieces.")
    existing = {piece.piece: i for i, piece in enumerate(original.pieces)}
    if any(char in existing for char in DONORS):
        raise ValueError("A proposed Turkish piece already exists.")
    modified = pb.ModelProto()
    modified.CopyFrom(original)
    minimum_score = min(p.score for p in original.pieces)
    additions = []
    for index, (char, donor) in enumerate(DONORS.items()):
        if donor not in existing:
            raise ValueError(f"Donor {donor!r} is missing.")
        piece = modified.pieces.add()
        piece.piece, piece.score, piece.type = char, minimum_score - index - 1, pb.ModelProto.SentencePiece.NORMAL
        additions.append({"piece": char, "native_id": 8192 + index, "donor_piece": donor,
                          "donor_id": existing[donor], "score": piece.score})
    modified.trainer_spec.vocab_size = 8198
    # Exactly two permitted protobuf changes: appended pieces and declared size.
    roundtrip = pb.ModelProto()
    roundtrip.CopyFrom(modified)
    del roundtrip.pieces[8192:]
    roundtrip.trainer_spec.vocab_size = original.trainer_spec.vocab_size
    if roundtrip.SerializeToString() != original.SerializeToString():
        raise AssertionError("Tokenizer metadata or original pieces changed.")
    old_sp = spm.SentencePieceProcessor(model_proto=original.SerializeToString())
    new_sp = spm.SentencePieceProcessor(model_proto=modified.SerializeToString())
    for i in range(8192):
        if old_sp.id_to_piece(i) != new_sp.id_to_piece(i):
            raise AssertionError("Original token ID changed.")
    controls = []
    for text in CONTROLS:
        old_ids, new_ids = old_sp.encode(text), new_sp.encode(text)
        if old_ids != new_ids:
            raise AssertionError(f"Control tokenization changed: {text}")
        controls.append({"text": text, "ids": new_ids, "decoded": new_sp.decode(new_ids)})
    turkish = []
    for text in TURKISH:
        ids = new_sp.encode(text)
        if new_sp.unk_id() in ids or new_sp.decode(ids) != text:
            raise AssertionError(f"Turkish lexical round trip failed: {text}")
        turkish.append({"text": text, "ids": ids, "decoded": new_sp.decode(ids)})
    destination_dir.mkdir(parents=True)
    (destination_dir / "tokenizer.model").write_bytes(modified.SerializeToString())
    # Preserve the two original human-readable artifacts as exact byte prefixes.
    for old, target, lines in (
        (source_vocab, "tokenizer.vocab", [f"{x['piece']}\t{x['score']:g}" for x in additions]),
        (source_text_vocab, "vocab.txt", [x["piece"] for x in additions]),
    ):
        previous = old.read_bytes()
        separator = b"" if previous.endswith(b"\n") else b"\n"
        (destination_dir / target).write_bytes(previous + separator + ("\n".join(lines) + "\n").encode("utf-8"))
    report = {
        "schema_version": 1, "kind": "native_parakeet_turkish_sentencepiece_extension_v1", "passed": True,
        "initialization": "pinned native Parakeet-v3 base; learned HF heads/LoRA are not transferred",
        "native_model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "original_lexical_size": 8192, "new_lexical_size": 8198,
        "original_blank_id": 8192, "new_blank_id": 8198,
        "old_duration_rows": [8193, 8194, 8195, 8196, 8197],
        "new_duration_rows": [8199, 8200, 8201, 8202, 8203],
        "additions": additions, "original_piece_ids_preserved": 8192,
        "all_other_model_proto_fields_identical": True,
        "byte_fallback": original.trainer_spec.byte_fallback,
        "normalizer_name": original.normalizer_spec.name,
        "normalizer_spec_sha256": hashlib.sha256(original.normalizer_spec.SerializeToString()).hexdigest(),
        "tokenizer_dir": str(destination_dir.resolve()),
        "source_files": {str(p.resolve()): sha(p) for p in (source_model, source_vocab, source_text_vocab)},
        "files_sha256": {str(p.resolve()): sha(p) for p in sorted(destination_dir.iterdir())},
        "control_encodings_unchanged": controls, "turkish_exact_roundtrips": turkish,
        "head_row_initialization": "six ASCII donor rows; subtract 2 from each new output bias",
        "native_semantics": "blank remains final lexical class and padding_idx; native zero SOS is preserved",
        "limitations": "Static tokenizer proof only; native model resize and Lightning training have not run.",
        "script_sha256": sha(__file__),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "manifest": str(manifest_path.resolve()), "manifest_sha256": sha(manifest_path),
                      "tokenizer_dir": str(destination_dir.resolve()), "pieces": new_sp.get_piece_size()}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--destination-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.destination_dir, args.manifest)
