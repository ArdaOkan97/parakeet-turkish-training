import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import ParakeetEncoderConfig, ParakeetForTDT, ParakeetTDTConfig
from transformers.models.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from transformers.models.parakeet.processing_parakeet import ParakeetProcessor
from transformers.models.parakeet.tokenization_parakeet import ParakeetTokenizer

task_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(task_root / "outputs/parakeet_turkish"))
from confidence import generate_with_confidence
torch.set_num_threads(4)
utility = task_root / "outputs/parakeet_turkish/extend_vocab.py"
spec = importlib.util.spec_from_file_location("extension", utility)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

native = Tokenizer.from_file(str(task_root / "work/research/tokenizer.json"))
tokenizer = ParakeetTokenizer(
    tokenizer_object=native,
    unk_token="<unk>",
    pad_token="<pad>",
    eos_token="<|endoftext|>",
)
processor = ParakeetProcessor(ParakeetFeatureExtractor(), tokenizer, decoder_type="tdt")
encoder_cfg = ParakeetEncoderConfig(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    num_mel_bins=16,
    subsampling_conv_channels=4,
)
config = ParakeetTDTConfig(
    encoder_config=encoder_cfg,
    vocab_size=8193,
    blank_token_id=8192,
    pad_token_id=2,
    decoder_hidden_size=8,
    num_decoder_layers=1,
    durations=[0, 1, 2, 3, 4],
)
torch.manual_seed(123)
model = ParakeetForTDT(config)
model.eval()
model.generation_config.decoder_start_token_id = 8192
model.generation_config.eos_token_id = 3
model.generation_config.pad_token_id = 2
model.generation_config.suppress_tokens = [0, 8193, 8194, 8195, 8196, 8197]
model.generation_config.begin_suppress_tokens = [2, 8193]
old_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
hidden = torch.randn(100, 8)
old_logits = model.joint.head(hidden).detach()
audit = extension.extend_processor_and_model(processor, model)
assert model.generation_config.suppress_tokens == [0, 8199, 8200, 8201, 8202, 8203]
assert model.generation_config.begin_suppress_tokens == [2, 8199]
new_logits = model.joint.head(hidden).detach()
assert torch.equal(old_logits[:, :8193], new_logits[:, :8193])
assert torch.equal(old_logits[:, 8193:], new_logits[:, 8199:])
assert torch.equal(old_logits[:, :8193].argmax(-1), new_logits[:, :8199].argmax(-1))
for key, value in old_state.items():
    if key not in ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias"):
        assert torch.equal(value, model.state_dict()[key]), key
random.seed(32)
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ çÇöÖüÜéèñßœ0123456789.,?!'"
for _ in range(1000):
    text = "".join(random.choices(alphabet, k=random.randint(0, 100)))
    assert native.encode(text).ids == processor.tokenizer.backend_tokenizer.encode(text).ids

save_dir = task_root / "work/research/extension_roundtrip"
processor.save_pretrained(save_dir)
model.save_pretrained(save_dir)
restored_processor = ParakeetProcessor.from_pretrained(save_dir, local_files_only=True)
restored_model = ParakeetForTDT.from_pretrained(save_dir, local_files_only=True)
for text in extension.TURKISH_TEXTS:
    before = processor.tokenizer.encode(text, add_special_tokens=False)
    after = restored_processor.tokenizer.encode(text, add_special_tokens=False)
    assert before == after
    assert restored_processor.decode(after, skip_special_tokens=True) == text
assert restored_model.config.vocab_size == 8199
assert restored_model.config.blank_token_id == 8192
assert restored_model.generation_config.suppress_tokens == model.generation_config.suppress_tokens
assert restored_model.generation_config.begin_suppress_tokens == model.generation_config.begin_suppress_tokens
for key, value in model.state_dict().items():
    assert torch.equal(value, restored_model.state_dict()[key]), key
assert extension.extend_processor_and_model(restored_processor, restored_model)["changed"] is False
model.zero_grad(set_to_none=True)
from transformers.loss.loss_tdt import tdt_loss
def integer_safe_loss(token_logits, duration_logits, labels, logit_lengths, label_lengths,
                      blank_token_id, durations, **kwargs):
    return tdt_loss(token_logits, duration_logits, labels.long(), logit_lengths.long(),
                    label_lengths.long(), blank_token_id, durations)
model.loss_function = integer_safe_loss
target_ids = [8193, 8195, 8197, 7881]
inputs = {
    "input_features": torch.randn(1, 40, 16),
    "attention_mask": torch.ones(1, 40, dtype=torch.long),
    "labels": torch.tensor([target_ids], dtype=torch.long),
    "decoder_input_ids": torch.tensor([[8192] + target_ids], dtype=torch.long),
}
loss = model(**inputs).loss
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert model.decoder.embedding.weight.grad[8193:8199].abs().sum() > 0
assert model.joint.head.weight.grad[8193:8199].abs().sum() > 0
assert model.joint.head.bias.grad[8193:8199].abs().sum() > 0

# Force the first new Turkish character, while a duration row has an even
# larger raw logit. Generation must suppress relocated duration rows rather
# than the new character, and the confidence hook must observe identical IDs.
model.eval()
with torch.no_grad():
    model.joint.head.weight.zero_()
    model.joint.head.bias.fill_(-100)
    model.joint.head.bias[8193] = 50
    model.joint.head.bias[8200] = 60  # duration=1, higher than every token
generation_inputs = {
    "input_features": torch.randn(1, 40, 16),
    "attention_mask": torch.ones(1, 40, dtype=torch.long),
}
generated, confidence = generate_with_confidence(
    model, processor.tokenizer, generation_inputs, language="tr", max_new_tokens=8,
)
assert generated.sequences[0, 0].item() == 8192
assert generated.sequences[0, 1:].numel() > 0
assert generated.sequences[0, 1:].eq(8193).all()
assert generated.durations[0, 1:].eq(1).all()
assert [token["token_id"] for token in confidence["tokens"]] == generated.sequences[0, 1:].tolist()
assert confidence["text"] == "ğ" * (generated.sequences.shape[1] - 1)
audit["additional_verification"] = {
    "unchanged_random_old_alphabet_tokenizations": 1000,
    "unchanged_random_old_token_logits": 100,
    "unchanged_random_duration_logits": 100,
    "unchanged_greedy_token_choices": 100,
    "unchanged_all_other_model_state": True,
    "save_reload_exact_model_and_tokenizer": True,
    "idempotent_second_invocation": True,
    "tdt_loss_and_all_gradients_finite": True,
    "new_character_embedding_and_head_rows_receive_gradient": True,
    "gradient_test_loss_wrapper": "Direct HF tdt_loss with int64 labels to repair upstream int32 gather bug",
    "test_model": "tiny random ParakeetForTDT with official v3 tokenizer",
    "generation_duration_suppression_relocated": True,
    "generation_original_valid_suppressions_preserved": True,
    "forced_new_turkish_character_generated": True,
    "higher_raw_duration_logit_excluded_from_generated_tokens": True,
    "generated_new_token_matches_confidence_hook": True,
    "forced_generation_sequence": generated.sequences[0].tolist(),
    "forced_generation_durations": generated.durations[0].tolist(),
}
(task_root / "outputs/parakeet_turkish/vocabulary_extension_verification.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(audit, indent=2, ensure_ascii=False))
