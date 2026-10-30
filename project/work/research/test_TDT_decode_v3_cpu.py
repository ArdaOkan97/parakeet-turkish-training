"""Installed official tokenizer regression using TRAIN-only literal fixtures."""
from pathlib import Path
import sys,json,hashlib,ast
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(D))
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from transformers import AutoTokenizer
from metrics import normalize_text
from resource_watchdog_v2 import sample_memory
from public_turkish_data_policy_v3 import sha256
import os
target=D/'TDT_decode_v3_cpu_verification.json'
if target.exists():raise FileExistsError(target)
tokenizer=AutoTokenizer.from_pretrained(D/'checkpoint',local_files_only=True)
checks=[];examples=[]
for text in ('sözlerinin','yaşayamam','Bununla','getiririz','hepsinin'):
    ids=tokenizer(text)['input_ids'];tdt=tokenizer.decode(ids,skip_special_tokens=True,group_tokens=False);ctc=tokenizer.decode(ids,skip_special_tokens=True)
    if normalize_text(tdt,'tr')!=normalize_text(text,'tr') or normalize_text(ctc,'tr')==normalize_text(text,'tr'):raise AssertionError(text)
    checks.append('TDT_preserves_repeated_subwords_'+text)
    examples.append(dict(TRAIN_fixture=text,label_ids=ids,TDT_ungrouped_decode=tdt,CTC_grouped_decode=ctc))
tree=ast.parse((D/'train_lora_batched_data_v3.py').read_text(encoding='utf-8'))
call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='processor.tokenizer.decode')
if not any(k.arg=='group_tokens' and isinstance(k.value,ast.Constant) and k.value.value is False for k in call.keywords):raise AssertionError('ExplicitFalse required')
checks.append('futuretrainer_explicitFalse')
if torch.cuda.is_initialized():raise AssertionError('No CUDA')
checks.append('CUDA_not_initialized')
semantic=hashlib.sha256(json.dumps(json.loads(tokenizer._tokenizer.to_str()),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
if semantic!='b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75':raise AssertionError('Tokenizer unchanged')
checks.append('exact_tokenizer_semantic_identity')
value=dict(schema=1,passed=True,CPU_only=True,check_count=len(checks),checks=checks,TRAIN_only_literal_examples=examples,
    tokenizer_semantic_sha256=semantic,trainer_sha256=sha256(D/'train_lora_batched_data_v3.py'),
    installed_tokenizer_source_sha256=sha256(Path(sys.modules[tokenizer.__class__.__module__].__file__)),
    torch_CUDA_initialized=False,vocabulary_changed=False,owned_memory=sample_memory(os.getpid(),'TDT_regression_complete'))
target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks),CUDA_initialized=False)))
