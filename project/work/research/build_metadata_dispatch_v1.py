from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(name,text):
    p=D/name
    if p.exists():raise FileExistsError(p)
    ast.parse(text);p.write_text(text,encoding='utf-8',newline='\n');return sha(p)
trainer=sha(D/'train_lora_batched_data_v5.py')
text=(D/'launch_with_io_retry_v2.py').read_text().replace('launch_with_io_retry_v2.py','launch_with_io_retry_v3.py').replace('train_lora_batched_data_v4.py','train_lora_batched_data_v5.py').replace("'kind': 'atomic_io_retry_v2'","'kind': 'atomic_io_retry_v3'")
io=create('launch_with_io_retry_v3.py',text)
text=(D/'launch_with_cache_reclaim_v5.py').read_text().replace('launch_with_io_retry_v2','launch_with_io_retry_v3').replace('c01e5530000ae653e2acd53c2459713a2bd08243e557b3173bdc5a57d193b1a4',io).replace('train_lora_batched_data_v4.py','train_lora_batched_data_v5.py').replace('cebb01f64826ad88a05e3577552d049fe0b374d744a64ea16bb7bd96ecf08695',trainer).replace('cache_reclamation_training_launcher_v5','cache_reclamation_training_launcher_v6')
cache=create('launch_with_cache_reclaim_v6.py',text)
text=(D/'launch_with_memory_guard_v8.py').read_text().replace('launch_with_cache_reclaim_v5','launch_with_cache_reclaim_v6').replace('019eb20ed982f91a645558c4fb2c2bf0c94a6eef366cec4edd9ddda3a52288dc',cache).replace('train_lora_batched_data_v4.py','train_lora_batched_data_v5.py').replace('cebb01f64826ad88a05e3577552d049fe0b374d744a64ea16bb7bd96ecf08695',trainer).replace('memory_guard_training_launcher_v8','memory_guard_training_launcher_v9')
memory=create('launch_with_memory_guard_v9.py',text)
checks=[]
for name in ['launch_with_io_retry_v3.py','launch_with_cache_reclaim_v6.py','launch_with_memory_guard_v9.py','train_lora_batched_data_v5.py','batched_lora_tr_metadata_v1.py']:
    ast.parse((D/name).read_text());checks.append('AST_valid_'+name)
io_text=(D/'launch_with_io_retry_v3.py').read_text()
assert io_text.count("if args.worker == 'train_lora_batched_data_v5.py':")==2
assert "ALLOWED_WORKERS = ('train_lora_batched_data_v5.py', 'finalize_lora.py')" in io_text
assert 'parsed = worker.validate_arguments(worker.parser().parse_args(arguments))' in io_text
checks+=['all_three_IO_trainer_dispatch_branches','real_validated_trainer_parser_route']
for new,old,replacements in [('launch_with_io_retry_v3.py','launch_with_io_retry_v2.py',[('launch_with_io_retry_v3.py','launch_with_io_retry_v2.py'),('train_lora_batched_data_v5.py','train_lora_batched_data_v4.py'),('atomic_io_retry_v3','atomic_io_retry_v2')])]:
    revised=(D/new).read_text()
    for a,b in replacements:revised=revised.replace(a,b)
    assert revised==(D/old).read_text();checks.append('IO3_dispatch_only_version_changes')
report={'schema':1,'passed':True,'CPU_stdlib_only':True,'checks':checks,'trainer_sha256':trainer,'IO3_sha256':io,'cache6_sha256':cache,'memory9_sha256':memory,
    'scope':'Narrow version/name/hash substitutions from CPU actual-dispatch19 tested IO2/cache5/memory8; no GPU adoption claim',
    'prior_actual_dispatch_proof_sha256':sha(D/'data_v4_actual_io_dispatch_cpu_verification.json')}
target=D/'TR_metadata_dispatch_static_verification_v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
