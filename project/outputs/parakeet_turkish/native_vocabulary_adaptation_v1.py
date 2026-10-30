"""Preserve native Parakeet-v3 weights while adding six SentencePiece symbols.

Importing this module is lightweight. Torch/NeMo are imported only when the
adaptation function is called. Run on a native base model restored to CPU,
before constructing an optimizer or installing LoRA. This is native-base
initialization, not a conversion of the learned HF model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

OLD_SIZE = 8192
NEW_SIZE = 8198
DURATIONS = [0, 1, 2, 3, 4]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_output_row_plan(additions):
    """Pure-Python exact row map, also usable without loading Torch/NeMo."""
    if [x["native_id"] for x in additions] != list(range(OLD_SIZE, NEW_SIZE)):
        raise ValueError("Unexpected native Turkish piece IDs.")
    donors = [int(x["donor_id"]) for x in additions]
    if any(x < 0 or x >= OLD_SIZE for x in donors):
        raise ValueError("Invalid old-vocabulary donor ID.")
    return list(range(OLD_SIZE)) + donors + [OLD_SIZE] + list(range(OLD_SIZE + 1, OLD_SIZE + 6))


def apply_native_turkish_vocabulary(model, manifest_path, new_row_bias_penalty=2.0):
    """Resize a native EncDecRNNTBPEModel using its official API, retaining rows.

    Return a JSON-serializable audit. All existing encoder/frontend objects and
    old decoder/joint parameters remain unchanged. Original lexical rows keep
    IDs; the native blank and five duration rows move six places. Newly added
    lexical rows copy declared ASCII donors. Native zero-SOS/blank-as-pad
    semantics and the released sigma=.02, omega=.1, mean_volume TDT recipe are
    retained. The official BPE change_vocabulary default-RNNT reset is repaired
    using the same loss constructor as EncDecRNNTModel.__init__.
    """
    import torch
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "native_parakeet_turkish_sentencepiece_extension_v1" or not manifest.get("passed"):
        raise ValueError("A verified native tokenizer asset manifest is required.")
    for path, expected in manifest["files_sha256"].items():
        if _sha(path) != expected:
            raise ValueError(f"Tokenizer artifact changed: {path}")
    tokenizer_dir = Path(manifest["tokenizer_dir"])
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Resize the model on CPU before GPU/optimizer setup.")
    if int(model.decoder.blank_idx) != OLD_SIZE or int(model.cfg.decoder.vocab_size) != OLD_SIZE:
        raise ValueError("Expected the unextended native base, with blank at lexical end.")
    if int(model.joint.num_extra_outputs) != len(DURATIONS) or int(model.joint.num_classes_with_blank) != OLD_SIZE + 6:
        raise ValueError("Expected 8192 lexical + blank + five TDT duration outputs.")
    loss_cfg = copy.deepcopy(model.cfg.loss)
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(loss_cfg)
    kwargs = OmegaConf.to_container(loss_kwargs, resolve=True) if OmegaConf.is_config(loss_kwargs) else dict(loss_kwargs)
    reduction = model.cfg.get("rnnt_reduction", "mean_batch")
    if loss_name != "tdt" or list(kwargs.get("durations", [])) != DURATIONS:
        raise ValueError("The native model must retain its released TDT loss.")
    if float(kwargs.get("sigma", -1)) != 0.02 or float(kwargs.get("omega", -1)) != 0.1 or reduction != "mean_volume":
        raise ValueError("Unexpected native model loss recipe; do not silently change it.")
    original_pieces = [model.tokenizer.ids_to_tokens([i])[0] for i in range(OLD_SIZE)]
    for item in manifest["additions"]:
        if original_pieces[item["donor_id"]] != item["donor_piece"]:
            raise ValueError("The supplied model/tokenizer donor IDs disagree.")
    encoder, preprocessor = model.encoder, model.preprocessor
    decoder_training, joint_training = model.decoder.training, model.joint.training
    old_decoder = {name: tensor.detach().clone() for name, tensor in model.decoder.state_dict().items()}
    old_joint = {name: tensor.detach().clone() for name, tensor in model.joint.state_dict().items()}
    embedding_name, head_weight, head_bias = "prediction.embed.weight", "joint_net.2.weight", "joint_net.2.bias"
    if old_decoder[embedding_name].shape[0] != OLD_SIZE + 1:
        raise ValueError("Unexpected native decoder embedding shape.")
    if old_joint[head_weight].shape[0] != OLD_SIZE + 6 or old_joint[head_bias].shape != (OLD_SIZE + 6,):
        raise ValueError("Unexpected native joint head layout.")
    decoder_rows = native_output_row_plan(manifest["additions"])[:NEW_SIZE + 1]
    joint_rows = native_output_row_plan(manifest["additions"])
    state_before = torch.get_rng_state()
    try:
        model.change_vocabulary(str(tokenizer_dir), "bpe")
        with open_dict(model.cfg):
            model.cfg.loss = loss_cfg
        # BPE change_vocabulary creates RNNTLoss(default) and incorrectly counts
        # duration outputs as tokens. Mirror the native constructor explicitly.
        model.loss = RNNTLoss(num_classes=NEW_SIZE, loss_name=loss_name, loss_kwargs=loss_kwargs, reduction=reduction)
        model.change_decoding_strategy(copy.deepcopy(model.cfg.decoding), verbose=False)
        new_decoder = model.decoder.state_dict()
        new_joint = model.joint.state_dict()
        if set(new_decoder) != set(old_decoder) or set(new_joint) != set(old_joint):
            raise ValueError("Native module state keys changed beyond vocabulary resizing.")
        expected_decoder = {}
        expected_joint = {}
        for name, tensor in old_decoder.items():
            expected_decoder[name] = tensor[decoder_rows].clone() if name == embedding_name else tensor
            if expected_decoder[name].shape != new_decoder[name].shape:
                raise ValueError(f"Unapproved decoder shape change: {name}")
        for name, tensor in old_joint.items():
            expected_joint[name] = tensor[joint_rows].clone() if name in (head_weight, head_bias) else tensor
            if name == head_bias:
                expected_joint[name][OLD_SIZE:NEW_SIZE] -= float(new_row_bias_penalty)
            if expected_joint[name].shape != new_joint[name].shape:
                raise ValueError(f"Unapproved joint shape change: {name}")
        model.decoder.load_state_dict(expected_decoder, strict=True)
        model.joint.load_state_dict(expected_joint, strict=True)
        model.decoder.train(decoder_training)
        model.joint.train(joint_training)
        if model.encoder is not encoder or model.preprocessor is not preprocessor:
            raise AssertionError("Encoder/frontend were unexpectedly replaced.")
        for module, expected in ((model.decoder, expected_decoder), (model.joint, expected_joint)):
            for name, tensor in module.state_dict().items():
                if not torch.equal(tensor, expected[name]):
                    raise AssertionError(f"Native weight preservation failed: {name}")
        if model.decoder.prediction["embed"].padding_idx != NEW_SIZE or model.decoder.blank_idx != NEW_SIZE:
            raise AssertionError("Native blank-as-pad was not preserved.")
        if model.loss._blank != NEW_SIZE or model.joint.num_classes_with_blank != NEW_SIZE + 6:
            raise AssertionError("Loss blank/head duration layout is wrong.")
        if model.joint.fuse_loss_wer and model.joint.loss is not model.loss:
            raise AssertionError("Fused native joint is bound to a different loss.")
        if [model.tokenizer.ids_to_tokens([i])[0] for i in range(OLD_SIZE)] != original_pieces:
            raise AssertionError("Original lexical token IDs changed.")
    finally:
        # Unused random resize initialization should not change training RNG.
        torch.set_rng_state(state_before)
    if not torch.equal(torch.get_rng_state(), state_before):
        raise AssertionError("CPU Torch RNG was not restored.")
    return {
        "kind": "native_parakeet_turkish_vocabulary_adaptation_v1", "passed": True,
        "initialization": "native base; learned HF weights/optimizer not transferred",
        "tokenizer_manifest_path": str(manifest_path), "tokenizer_manifest_sha256": _sha(manifest_path),
        "helper_sha256": _sha(__file__), "old_lexical_size": OLD_SIZE, "new_lexical_size": NEW_SIZE,
        "old_blank_id": OLD_SIZE, "new_blank_id": NEW_SIZE, "duration_rows": list(range(NEW_SIZE + 1, NEW_SIZE + 6)),
        "new_row_bias_penalty": float(new_row_bias_penalty), "additions": manifest["additions"],
        "old_decoder_nonvocabulary_parameters_identical": True, "old_joint_nonvocabulary_parameters_identical": True,
        "original_lexical_rows_identical": True, "blank_duration_rows_identical_after_move": True,
        "encoder_preprocessor_objects_unchanged": True, "cpu_torch_rng_restored": True,
        "loss_name": loss_name, "loss_kwargs": kwargs, "reduction": reduction,
        "loss_implementation": type(model.loss._loss).__module__ + "." + type(model.loss._loss).__name__,
        "decoder_semantics": "stock native zero SOS and final lexical blank padding_idx",
        "functionality_limit": "This validates row/config preservation; WER and Lightning throughput require an actual pilot.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--output", required=True, help="Fresh output .nemo path")
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.audit).exists():
        raise FileExistsError("Use fresh native output/audit paths.")
    # Standalone conversion uses the same successful CUDA12.9 linkage as the
    # trial. The callable API expects its owning native launcher to do this.
    from preload_nemo_linker_v1 import preload
    preload()
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    model = EncDecRNNTBPEModel.restore_from(args.native_model, map_location="cpu")
    audit = apply_native_turkish_vocabulary(model, args.tokenizer_manifest)
    model.save_to(args.output)
    audit.update({"native_source_path": str(Path(args.native_model).resolve()), "saved_native_model_path": str(Path(args.output).resolve())})
    Path(args.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "audit": args.audit, "output": args.output}))


if __name__ == "__main__":
    main()
