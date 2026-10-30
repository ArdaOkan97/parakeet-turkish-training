"""Convert an official Parakeet v2 .nemo archive without installing NeMo.

Reuses the weight/config/tokenizer conversion helpers shipped with the pinned
Hugging Face Transformers version. Extracts only selected regular archive files
and requires a complete strict state-dict match before saving anything usable.

Usage:
  python convert_v2.py --nemo-path work/cache/parakeet-tdt-0.6b-v2.nemo \
      --output-dir work/models/parakeet-v2 --extract-dir work/v2-extracted

Source: https://github.com/huggingface/transformers/blob/
536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/parakeet/convert_nemo_to_hf.py
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tarfile
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, ParakeetForTDT, ParakeetTDTConfig
from transformers.models.parakeet.convert_nemo_to_hf import (
    convert_encoder_config,
    convert_tdt_config,
    load_and_convert_tdt_state_dict,
    write_processor,
)


def extract_selected(nemo_path: Path, extract_dir: Path) -> dict[str, str]:
    extract_dir = extract_dir.resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path, mode="r:*") as archive:
        regular = [member for member in archive.getmembers() if member.isfile()]

        def choose(kind: str) -> tarfile.TarInfo:
            if kind == "model_config":
                candidates = [member for member in regular if Path(member.name).name == "model_config.yaml"]
            elif kind == "model_weights":
                candidates = [member for member in regular if Path(member.name).name == "model_weights.ckpt"]
            else:
                candidates = [member for member in regular if Path(member.name).suffix == ".model"]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one {kind} archive member, found {[member.name for member in candidates]}")
            return candidates[0]

        files = {}
        for kind in ("model_config", "model_weights", "tokenizer_model_file"):
            member = choose(kind)
            # The output is explicitly flat; archive paths are never executed or
            # recreated, and symlinks/devices are excluded by member.isfile().
            target = (extract_dir / Path(member.name).name).resolve()
            if target.parent != extract_dir:
                raise ValueError("Archive member resolves outside the extraction directory.")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=2 * 1024 * 1024)
            files[kind] = str(target)
            print(f"Extracted {kind}: {target.name} ({member.size} bytes)", flush=True)
    return files


def convert(nemo_path: Path, output_dir: Path, extract_dir: Path, reuse_extracted: bool = False) -> dict:
    if reuse_extracted:
        tokenizer_paths = list(extract_dir.glob("*.model"))
        if len(tokenizer_paths) != 1:
            raise ValueError("Reuse requires exactly one extracted SentencePiece model.")
        files = {"model_config": str(extract_dir / "model_config.yaml"),
                 "model_weights": str(extract_dir / "model_weights.ckpt"),
                 "tokenizer_model_file": str(tokenizer_paths[0])}
        if not all(Path(path).is_file() for path in files.values()):
            raise FileNotFoundError("Required extracted files do not exist.")
    else:
        files = extract_selected(nemo_path, extract_dir)
    with Path(files["model_config"]).open(encoding="utf-8") as handle:
        nemo_config = yaml.safe_load(handle)
    encoder_config = convert_encoder_config(nemo_config)
    labels = nemo_config["labels"]
    blank_as_pad = "<pad>" not in labels
    if blank_as_pad:
        # v2 has 1024 real tokenizer entries and blank at id 1024. Adding a
        # separate pad would shift blank beyond the pretrained joint head.
        # Reuse blank as padding, as the native archive's decoder specifies.
        if not nemo_config["decoder"].get("blank_as_pad", False):
            raise ValueError("Missing pad is only supported when NeMo specifies blank_as_pad.")
        prednet = nemo_config["decoder"].get("prednet", {})
        config = ParakeetTDTConfig(
            vocab_size=len(labels) + 1,
            blank_token_id=len(labels),
            pad_token_id=len(labels),
            decoder_hidden_size=prednet.get("pred_hidden", 640),
            num_decoder_layers=prednet.get("pred_rnn_layers", 2),
            durations=nemo_config["decoding"].get("durations", [0, 1, 2, 3, 4]),
            hidden_act=nemo_config["joint"]["jointnet"].get("activation", "relu"),
            max_symbols_per_step=10,
            encoder_config=encoder_config.to_dict(),
        )
    else:
        config = convert_tdt_config(nemo_config, encoder_config)
    if config.durations != list(range(len(config.durations))):
        raise ValueError("This pinned Transformers generation code requires contiguous duration values.")
    converted = load_and_convert_tdt_state_dict(files, config.vocab_size)
    with torch.device("meta"):
        model = ParakeetForTDT(config)
    model.load_state_dict(converted, strict=True, assign=True)
    model.eval()
    model.generation_config.decoder_start_token_id = config.blank_token_id
    model.generation_config.suppress_tokens = list(range(config.vocab_size, config.vocab_size + len(config.durations)))
    output_dir.mkdir(parents=True, exist_ok=True)
    # The official RNNT processor branch preserves vocab when blank doubles as
    # padding. Change only the explicit decoder type back to TDT afterwards.
    write_processor(nemo_config, files, str(output_dir), model_type="rnnt" if blank_as_pad else "tdt")
    processor = AutoProcessor.from_pretrained(output_dir, local_files_only=True)
    processor.decoder_type = "tdt"
    processor.save_pretrained(output_dir)
    if processor.tokenizer.convert_tokens_to_ids("<blank>") != config.blank_token_id:
        raise ValueError("Converted tokenizer blank id does not match the joint head.")
    if processor.tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError("Converted tokenizer pad id does not match model configuration.")
    if len(processor.tokenizer) != config.vocab_size:
        raise ValueError("Converted tokenizer size does not match model vocabulary.")
    if any(processor.tokenizer.convert_ids_to_tokens(index) != label for index, label in enumerate(labels)):
        raise ValueError("A converted tokenizer label moved to a different pretrained weight row.")
    if model.joint.head.out_features != config.vocab_size + len(config.durations):
        raise ValueError("Joint output width does not match token and duration classes.")
    if model.decoder.embedding.num_embeddings != config.vocab_size:
        raise ValueError("Decoder embedding rows do not match the original token vocabulary.")
    model.save_pretrained(output_dir, safe_serialization=True)
    summary = {
        "source_nemo": str(nemo_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "token_vocabulary_including_blank": config.vocab_size,
        "blank_id": config.blank_token_id,
        "pad_id": config.pad_token_id,
        "native_blank_as_pad": blank_as_pad,
        "duration_values": config.durations,
        "strict_state_dict_match": True,
        "missing_weight_keys": [],
        "unexpected_weight_keys": [],
        "state_dict_tensor_count": len(converted),
        "original_token_ids_preserved": True,
        "joint_output_width": model.joint.head.out_features,
        "decoder_embedding_rows": model.decoder.embedding.num_embeddings,
        "source_converter": "transformers.models.parakeet.convert_nemo_to_hf",
        "transformers_commit": "536ecc007387a50e77603bb5d92100e9b07514cc",
        "source_revision": "ae9ad07059c7c739ffaf932226a8fe64ae2620b0",
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(__file__).parent / "v2_conversion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, converted
    gc.collect()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--reuse-extracted", action="store_true", help="Use previously extracted local files.")
    arguments = parser.parse_args()
    convert(arguments.nemo_path, arguments.output_dir, arguments.extract_dir, arguments.reuse_extracted)
