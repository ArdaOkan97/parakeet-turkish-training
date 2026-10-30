from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(name,text):
    path=D/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n');return sha(path)
text=(D/'train_lora_batched_data_v5.py').read_text()
text=text.replace('train_lora_batched_data_v5.py','train_lora_batched_data_v6.py')
text=text.replace('from public_turkish_data_policy_v3 import audit_public_training_inputs, source_isolation_v3, confirm_acoustic_cache_acceptance','from public_turkish_data_policy_v3 import source_isolation_v3, confirm_acoustic_cache_acceptance; from training_input_audit_reuse_v1 import audit_public_training_inputs_reused as audit_public_training_inputs')
text=text.replace('train_limit=args.train_limit, supplement=args.supplement)','train_limit=args.train_limit, supplement=args.supplement, reuse_receipt=args.reuse_input_audit)')
text=text.replace('"tr_frame_metadata_v1.py", "batched_lora.py"','"tr_frame_metadata_v1.py", "training_input_audit_reuse_v1.py", "batched_lora.py"')
text=text.replace('args.tr_frame_metadata_proof, OUTPUT / "tr_public_v3_tokenizer_acceptance.json"','args.tr_frame_metadata_proof, *([args.reuse_input_audit] if args.reuse_input_audit else []), OUTPUT / "tr_public_v3_tokenizer_acceptance.json"')
text=text.replace('p.add_argument("--run-id",required=True)','p.add_argument("--reuse-input-audit",type=Path); p.add_argument("--run-id",required=True)')
tree=ast.parse(text);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize' and n.lineno==539 for n in ast.walk(run))
trainer=create('train_lora_batched_data_v6.py',text)
text=(D/'launch_with_io_retry_v3.py').read_text().replace('launch_with_io_retry_v3.py','launch_with_io_retry_v4.py').replace('train_lora_batched_data_v5.py','train_lora_batched_data_v6.py').replace('atomic_io_retry_v3','atomic_io_retry_v4')
io=create('launch_with_io_retry_v4.py',text)
text=(D/'launch_with_cache_reclaim_v6.py').read_text().replace('launch_with_io_retry_v3','launch_with_io_retry_v4').replace(sha(D/'launch_with_io_retry_v3.py'),io).replace('train_lora_batched_data_v5.py','train_lora_batched_data_v6.py').replace(sha(D/'train_lora_batched_data_v5.py'),trainer).replace('cache_reclamation_training_launcher_v6','cache_reclamation_training_launcher_v7')
cache=create('launch_with_cache_reclaim_v7.py',text)
text=(D/'launch_with_memory_guard_v9.py').read_text().replace('launch_with_cache_reclaim_v6','launch_with_cache_reclaim_v7').replace(sha(D/'launch_with_cache_reclaim_v6.py'),cache).replace('train_lora_batched_data_v5.py','train_lora_batched_data_v6.py').replace(sha(D/'train_lora_batched_data_v5.py'),trainer).replace('memory_guard_training_launcher_v9','memory_guard_training_launcher_v10')
memory=create('launch_with_memory_guard_v10.py',text)
print(json.dumps({'trainer6':trainer,'IO4':io,'cache7':cache,'memory10':memory,'audit_reuse_helper':sha(D/'training_input_audit_reuse_v1.py')}))
