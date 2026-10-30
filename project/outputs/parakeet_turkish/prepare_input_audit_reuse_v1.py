"""Snapshot observed frozen full-audit completion plus immutable WAV fingerprints.

The original auditor returned its result in memory only. Completion is evidenced
by pinned sequential worker source and its owned log progressing into cache.
This receipt does not claim a separately saved returned audit JSON or a new PCM
byte hash. A future worker rejects changed file fingerprints and retains the
full byte audit as its default when this optional receipt is not supplied.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,re,time
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
WORKER_SHA='cebb01f64826ad88a05e3577552d049fe0b374d744a64ea16bb7bd96ecf08695'
POLICY_SHA='08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41'
MANIFEST_SHA='d2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-loop-id',default='adaptive_20261011_public_data_02')
    p.add_argument('--source-run-id',default='adaptive_20261011_public_data_02_public_rank8_control')
    p.add_argument('--output-dir',type=Path,required=True);return p
def main(args):
    started=time.perf_counter();output=args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('NEW workspace receipt directory required')
    source_dir=D/'runs'/args.source_loop_id
    paths={'owned_identity':source_dir/(args.source_run_id+'_identity.json'),
        'IO_provenance':source_dir/(args.source_run_id+'_io.json'),
        'owned_worker_log':source_dir/(args.source_run_id+'_training.log')}
    identity=json.loads(paths['owned_identity'].read_text());io=json.loads(paths['IO_provenance'].read_text())
    if (identity['expected_run_id']!=args.source_run_id or identity['frozen_worker_sha256']!=WORKER_SHA
        or identity['data_policy_sha256']!=POLICY_SHA or io['worker_sha256']!=WORKER_SHA
        or sha(D/'train_lora_batched_data_v4.py')!=WORKER_SHA or sha(D/'public_turkish_data_policy_v3.py')!=POLICY_SHA):raise ValueError('Observed owned worker source identity differs')
    values=list(io['arguments']);values=values[1:] if values[:1]==['--'] else values
    argument_sha=hashlib.sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if argument_sha!=identity['worker_arguments_sha256'] or '--train-limit' in values or '--supplement' in values:raise ValueError('Whole vetted corpus worker arguments required')
    log=paths['owned_worker_log'].read_bytes();decoded=log.decode('utf-8',errors='strict')
    if 'public v3 PCM/input verification 70000/74883' not in decoded or not re.search(r'acoustic cache \d+/74883',decoded):raise ValueError('Pinned full-audit-to-cache sequential completion evidence absent')
    manifest=ROOT/'work/data/tr_public_v3_train.jsonl'
    if sha(manifest)!=MANIFEST_SHA:raise ValueError('Exact public corpus required')
    inputs=[manifest,ROOT/'work/data/replay_train.jsonl',D/'tr_public_v3_preparation.json',D/'tr_public_v3_tokenizer_acceptance.json']
    inputs += [ROOT/'work/data'/(split+'_'+lang+'.jsonl') for lang in ['en','es','it','fr','de','tr'] for split in ['validation','test']]
    rows=[]
    for path in [manifest,ROOT/'work/data/replay_train.jsonl',*inputs[4:]]:
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
    files={str(Path(row['audio_filepath']).resolve()):row for row in rows}
    output.mkdir(parents=True)
    snapshots={}
    for role,path in paths.items():
        target=output/(role+'.snapshot'+path.suffix)
        value=log if role=='owned_worker_log' else path.read_bytes()
        target.write_bytes(value);snapshots[role]={'path':str(target),'sha256':sha(target),'source_path':str(path)}
    fingerprints=output/'WAV_fingerprints.jsonl'
    with fingerprints.open('x',encoding='utf-8',newline='\n') as stream:
        for path,row in sorted(files.items()):
            stat=Path(path).stat()
            stream.write(json.dumps({'path':path,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                'source_id':row['source_id'],'sample_rate':row['sample_rate'],'duration':row['duration'],'manifest_wav_sha256':row['wav_sha256']})+'\n')
    receipt={'schema':1,'kind':'observed_full_input_audit_reuse_receipt_v1','passed':True,
        'source_loop_id':args.source_loop_id,'source_run_id':args.source_run_id,'actual_owned_worker_pid':identity['pid'],
        'worker_arguments_sha256':argument_sha,'worker_sha256':WORKER_SHA,'data_policy_sha256':POLICY_SHA,
        'prepare_helper_sha256':sha(__file__),'observed_completed_full_audit':True,
        'completion_evidence':'Pinned frozen sequential worker called full actual PCM auditor before entering acoustic cache; owned log prefix records both stages.',
        'original_returned_audit_JSON_separately_available':False,'new_PCM_byte_hashes_computed':False,
        'source_snapshots':snapshots,'input_files_sha256':{str(path.resolve()):sha(path) for path in inputs},
        'fingerprints_path':str(fingerprints),'fingerprints_sha256':sha(fingerprints),'unique_TRAIN_replay_heldout_WAV_files':len(files),
        'fingerprint_limitation':'Size and mtime_ns detect ordinary changes, not deliberate same-size same-timestamp content changes; no such source modifications authorized or performed.',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started}
    target=output/'reuse_receipt.json';target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'path':str(target),'sha256':sha(target),'WAV_files':len(files),'elapsed_seconds':receipt['elapsed_seconds']}))
if __name__=='__main__':main(parser().parse_args())
