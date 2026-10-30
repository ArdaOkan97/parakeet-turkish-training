"""Make a separate data-policy version, preserving original math and sync line."""
from pathlib import Path
import ast
import hashlib

ROOT = Path(__file__).resolve().parents[2]
directory = ROOT/'outputs/parakeet_turkish'
old = directory/'train_lora_batched.py'
if hashlib.sha256(old.read_bytes()).hexdigest() != 'b32aa1dc09d5ff7dd85be9c309e8d69a095b8cf859b5498136a27d5d5af7cd9d':
    raise ValueError('Frozen trainer differs')
text = old.read_text(encoding='utf-8')
def once(left,right):
    global text
    if text.count(left)!=1:raise ValueError('Expected unique source span: '+left[:100])
    text=text.replace(left,right)

once('from train import make_cache, checkpoint_state',
     'from train import make_cache, checkpoint_state; from public_turkish_data_policy_v2 import audit_public_training_inputs, source_isolation_v2, confirm_acoustic_cache_acceptance')
start=text.index('def source_isolation(tr, replay, heldout):')
end=text.index('\n\ndef annotate_cache_lengths(rows):',start)
span=text[start:end]
replacement='def source_isolation(tr, replay, heldout):\n    return source_isolation_v2(tr, replay, heldout)'
replacement+='\n'*(span.count('\n')-replacement.count('\n'))
text=text[:start]+replacement+text[end:]
once('    process_start = time.perf_counter()',
     '    process_start = time.perf_counter(); public_data_audit = audit_public_training_inputs(args.tr_manifest, args.replay_manifest, args.data_v2_proof, train_limit=args.train_limit, supplement=args.supplement)')
once('    replay = annotate_cache_lengths(make_cache(model, processor, replay, cache, args.max_tokens))',
     '    replay = annotate_cache_lengths(make_cache(model, processor, replay, cache, args.max_tokens)); confirm_acoustic_cache_acceptance(tr, public_data_audit)')
once('    source_names = ("train_lora_batched.py", "batched_lora.py",',
     '    source_names = ("train_lora_batched_data_v2.py", "public_turkish_data_policy_v2.py", "train_lora_batched.py", "batched_lora.py",')
once('    input_paths = [args.tr_manifest, args.replay_manifest, *heldout_paths]',
     '    input_paths = [args.tr_manifest, args.replay_manifest, args.data_v2_proof, OUTPUT / "tr_public_v2_tokenizer_acceptance.json", *heldout_paths]')
once('    config_identity = {"base_revision":BASE_REVISION,',
     '    config_identity = {"public_turkish_data_policy":public_data_audit["identity"], "base_revision":BASE_REVISION,')
once('    config = {**arguments, "steps":configured_steps,"base_revision":BASE_REVISION,',
     '    config = {**arguments, "public_turkish_data_audit":public_data_audit, "steps":configured_steps,"base_revision":BASE_REVISION,')
once('    write_json(run_dir / "training_pool.json", {"turkish":tr,"replay":replay,"input_sha256":input_sha})',
     '    write_json(run_dir / "training_pool.json", {"turkish":tr,"replay":replay,"input_sha256":input_sha}); write_json(run_dir / "public_data_v2_acceptance.json", public_data_audit)')
once('    p.add_argument("--tr-manifest",type=Path,default=DATA/"tr_expanded_train.jsonl")',
     '    p.add_argument("--tr-manifest",type=Path,default=DATA/"tr_public_v2_train.jsonl"); p.add_argument("--data-v2-proof",type=Path,default=OUTPUT/"tr_public_v2_preparation.json")')
tree=ast.parse(text)
sync=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.cuda.synchronize']
if len(sync)!=1 or sync[0].lineno!=539:raise ValueError('Reviewed optimizer-end synchronize location changed')
oldtree=ast.parse(old.read_text(encoding='utf-8'))
def calls(tree,name):
    return [ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node.func)==name]
for name in ('batch_objective','loss.backward','optimizer.step','torch.nn.utils.clip_grad_norm_','BalancedBucketSampler','save_training_state','development_gate'):
    if calls(tree,name)!=calls(oldtree,name):raise ValueError('Unexpected training/selection operation change: '+name)
target=directory/'train_lora_batched_data_v2.py'
if target.exists():raise FileExistsError(target)
target.write_text(text,encoding='utf-8',newline='\n')
print(hashlib.sha256(target.read_bytes()).hexdigest())
