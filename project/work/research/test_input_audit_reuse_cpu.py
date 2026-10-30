from pathlib import Path
import tempfile,sys,json,hashlib,copy,ast
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(D))
import training_input_audit_reuse_v1 as reuse
checks=[]
def check(name,truth):assert truth,name;checks.append(name)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj)+'\n');return digest(path)
with tempfile.TemporaryDirectory(dir=ROOT/'work') as folder:
    temp=Path(folder);audio=temp/'audio.wav';audio.write_bytes(b'CPU fixture no PCM reads')
    row={'audio_filepath':str(audio),'duration':1.,'sample_rate':16000,'source_id':'TRAINfixture'}
    manifest=temp/'train.jsonl';write(manifest,row)
    values=['--run-id','CPU_fixture'];argument_sha=hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    identity=temp/'identity.json';write(identity,{'expected_run_id':'CPU_fixture','pid':123,'frozen_worker_sha256':reuse.WORKER_SHA,'data_policy_sha256':reuse.POLICY_SHA,'worker_arguments_sha256':argument_sha})
    io=temp/'io.json';write(io,{'arguments':values,'worker_sha256':reuse.WORKER_SHA})
    log=temp/'log.txt';log.write_text('public v3 PCM/input verification 70000/74883\nacoustic cache 100/74883\n')
    snapshot={name:{'path':str(path),'sha256':digest(path)} for name,path in [('owned_identity',identity),('IO_provenance',io),('owned_worker_log',log)]}
    stat=audio.stat();fingerprints=temp/'fingerprints.jsonl';write(fingerprints,{'path':str(audio),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'duration':1.,'sample_rate':16000})
    receipt={'passed':True,'kind':'observed_full_input_audit_reuse_receipt_v1','observed_completed_full_audit':True,
        'original_returned_audit_JSON_separately_available':False,'new_PCM_byte_hashes_computed':False,
        'source_run_id':'CPU_fixture','actual_owned_worker_pid':123,'worker_sha256':reuse.WORKER_SHA,'data_policy_sha256':reuse.POLICY_SHA,
        'worker_arguments_sha256':argument_sha,'source_snapshots':snapshot,'input_files_sha256':{str(manifest):digest(manifest)},
        'fingerprints_path':str(fingerprints),'fingerprints_sha256':digest(fingerprints),'unique_TRAIN_replay_heldout_WAV_files':1}
    path=temp/'receipt.json';write(path,receipt)
    check('actual_stat_only_receipt_validation',reuse.validate_reuse_receipt(path,check_audio_headers=False)[1]==1)
    for name,change in [('wrong_owned_PID',lambda r:r.update(actual_owned_worker_pid=124)),('false_separate_returned_JSON',lambda r:r.update(original_returned_audit_JSON_separately_available=True)),('missing_fingerprint_count',lambda r:r.update(unique_TRAIN_replay_heldout_WAV_files=0)),('wrong_original_worker',lambda r:r.update(worker_sha256='0'*64)),('argument_identity_mismatch',lambda r:r.update(worker_arguments_sha256='0'*64))]:
        revised=copy.deepcopy(receipt);change(revised);write(path,revised)
        try:reuse.validate_reuse_receipt(path,check_audio_headers=False)
        except ValueError:checks.append('reject_'+name)
        else:raise AssertionError(name)
    write(path,receipt);audio.write_bytes(b'changed file size')
    try:reuse.validate_reuse_receipt(path,check_audio_headers=False)
    except ValueError:checks.append('reject_changed_actual_WAV_fingerprint')
    else:raise AssertionError('Changed audio')
    # Full PCM audit remains the default, without any receipt/header shortcut.
    calls=[]
    import types
    fake=types.ModuleType('public_turkish_data_policy_v3');fake.OUTPUT=D
    fake.audit_public_training_inputs=lambda *a,**kw:calls.append(kw) or {'CPU_fixture':True}
    sys.modules['public_turkish_data_policy_v3']=fake
    result=reuse.audit_public_training_inputs_reused('manifest','replay','proof')
    check('default_is_full_actual_PCM_audit',result['CPU_fixture'] and calls[0]['verify_actual_audio'] is True)
def find_function(file,name):return next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
old=find_function(D/'train_lora_batched_data_v5.py','run');new=find_function(D/'train_lora_batched_data_v6.py','run')
old_update=next(n for n in old.body if isinstance(n,ast.For) and ast.unparse(n.target)=='step');new_update=next(n for n in new.body if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
check('entire_training_update_AST_unchanged',ast.dump(old_update,include_attributes=False)==ast.dump(new_update,include_attributes=False))
check('exact_sync539_unchanged',any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize' and n.lineno==539 for n in ast.walk(new_update)))
check('reuse_flag_optional_parser',"p.add_argument(\"--reuse-input-audit\",type=Path)" in (D/'train_lora_batched_data_v6.py').read_text())
check('IO4_allthree_dispatch_branches', (D/'launch_with_io_retry_v4.py').read_text().count("if args.worker == 'train_lora_batched_data_v6.py':")==2 and "ALLOWED_WORKERS = ('train_lora_batched_data_v6.py', 'finalize_lora.py')" in (D/'launch_with_io_retry_v4.py').read_text())
target=D/'input_audit_reuse_cpu_verification_v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps({'schema':1,'passed':True,'CPU_stdlib_only':True,'checks':checks,'check_count':len(checks),'helper_sha256':digest(reuse.__file__),'trainer6_sha256':digest(D/'train_lora_batched_data_v6.py')},indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks)}))
