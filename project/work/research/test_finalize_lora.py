"""CPU-only reconstruction, gate and publication guards for finalization."""
import copy
import json
import sys
import uuid
from pathlib import Path

import torch
from transformers import ParakeetEncoderConfig, ParakeetForTDT, ParakeetTDTConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'outputs/parakeet_turkish'))
import finalize_lora as finalizer
from encoder_lora import install_encoder_lora

torch.set_num_threads(4)
torch.manual_seed(20261010)
encoder = ParakeetEncoderConfig(hidden_size=16, intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    num_mel_bins=16, subsampling_conv_channels=4)
config = ParakeetTDTConfig(encoder_config=encoder, vocab_size=16,
    blank_token_id=15, pad_token_id=2, decoder_hidden_size=8,
    num_decoder_layers=1)
model = ParakeetForTDT(config).eval()
layout = install_encoder_lora(model, rank=4, alpha=8)
expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
weights = {name: torch.randn_like(parameter) for name, parameter in expected.items()}
state = {'run_id':'cpu_guard_test','step':200,'lora_metadata':layout,'weights':weights}
loaded = finalizer.load_selected_trainables(model, state,
    run_id='cpu_guard_test',selected_step=200,layout=layout)
assert all(torch.equal(parameter, weights[name]) for name, parameter in expected.items())
assert loaded['all_trainable_keys_loaded']

def rejected(bad):
    try:
        finalizer.load_selected_trainables(model,bad,run_id='cpu_guard_test',selected_step=200,layout=layout)
    except RuntimeError:
        return True
    raise AssertionError('An invalid state was accepted')

key = next(iter(weights))
cases = []
bad = copy.deepcopy(state); bad['run_id']='wrong'; cases.append(rejected(bad))
bad = copy.deepcopy(state); bad['step']=400; cases.append(rejected(bad))
bad = copy.deepcopy(state); bad['lora_metadata']['targets'][0]['alpha']=99; cases.append(rejected(bad))
bad = copy.deepcopy(state); del bad['weights'][key]; cases.append(rejected(bad))
bad = copy.deepcopy(state); bad['weights']['unknown.weight']=torch.zeros(1); cases.append(rejected(bad))
bad = copy.deepcopy(state); bad['weights'][key]=torch.zeros(1); cases.append(rejected(bad))
bad = copy.deepcopy(state); bad['weights'][key]=bad['weights'][key].half(); cases.append(rejected(bad))
bad = copy.deepcopy(state); bad['weights'][key].view(-1)[0]=float('nan'); cases.append(rejected(bad))

baseline = {f'validation_{language}':{'asr':{'wer':1.0 if language=='tr' else .05,'utterances':50}}
    for language in finalizer.LANGUAGES}
development = copy.deepcopy(baseline)
development['validation_tr']['asr']['wer']=.7
development['validation_it']['asr']['wer']=.054
assert finalizer.development_gate(development,baseline)['passed']
development['validation_it']['asr']['wer']=.056
assert not finalizer.development_gate(development,baseline)['passed']
development['validation_it']['asr']['wer']=.05
development['validation_tr']['asr']['wer']=1.0
assert not finalizer.development_gate(development,baseline)['passed']

identifier = uuid.uuid4().hex[:12]
scratch = ROOT/'work/research'/f'finalizer_guards_{identifier}'
scratch.mkdir(parents=True)
candidate = ROOT/'work/models'/f'finalizer_cpu_fake_{identifier}'
candidate.mkdir(parents=True)
(candidate/'config.json').write_text('{"cpu_test":true}',encoding='utf-8')
(candidate/'model.safetensors').write_bytes(b'CPU helper test; not a model')
(candidate/'finalization_provenance.json').write_text('{"test":true}',encoding='utf-8')
manifest = finalizer.hf_file_manifest(candidate)
finalizer.OUTPUT = scratch
destination = scratch/'checkpoint'
destination.mkdir()
(destination/'prior.txt').write_text('previous recommendation',encoding='utf-8')
publication = finalizer.publish_checkpoint(candidate,destination,manifest)
assert finalizer.hf_file_manifest(destination)==manifest
backup = Path(publication['previous_checkpoint_backup'])
assert (backup/'prior.txt').read_text(encoding='utf-8')=='previous recommendation'
reuse = finalizer.publish_checkpoint(candidate,destination,manifest)
assert reuse['reused_identical_export']
assert not reuse['previous_checkpoint_backup']

source = scratch/'dev_results'; source.mkdir()
metadata = {'schema_version':1,'model_fingerprint':{'sha256':'cpu_fixture'},'manifests':{}}
(source/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
for language in finalizer.LANGUAGES:
    (source/f'validation_{language}.jsonl').write_text(json.dumps({'language':language})+'\n',encoding='utf-8')
final_results = scratch/'adapted_results'
finalizer.seed_validation_predictions(source,final_results)
finalizer.seed_validation_predictions(source,final_results)
assert all((final_results/f'validation_{language}.jsonl').read_bytes()==
    (source/f'validation_{language}.jsonl').read_bytes() for language in finalizer.LANGUAGES)
(final_results/'metadata.json').write_text(json.dumps({**metadata,'model_fingerprint':{'sha256':'wrong'}}),encoding='utf-8')
try:
    finalizer.seed_validation_predictions(source,final_results)
except RuntimeError:
    pass
else:
    raise AssertionError('Stale model predictions were accepted')

report = {'test_device':'CPU only','gpu_execution':False,
    'exact_trainable_tensor_restore':True,'invalid_state_guard_cases':len(cases),
    'development_gate_pass_fail_boundary_checks':True,
    'full_export_manifest_copy_equality':True,'prior_checkpoint_preserved':True,
    'identical_checkpoint_reuse':True,'development_prediction_seed_exact':True,
    'stale_prediction_fingerprint_rejected':True,
    'scope':'Helper guard tests; actual selected model merge/reload and held-out ASR are performed only by finalization'}
(ROOT/'outputs/parakeet_turkish/finalizer_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
