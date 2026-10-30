"""Finite final-stat snapshot, only after DATA actual acquisition/header exits.

Preserves download-time full SHA authority; never claims to rehash 718GB here.
No running source, final marker, decoder environment or DATA worker is edited.
"""
import json,os,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/BPE_weak_then_clean_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require,scoped
from bpe_broad_source_identity_v1 import FinalSourceIdentity

def stable_metadata(path):
    path=scoped(path);before=path.stat();binding=record(path);after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Final metadata/index changed while hashing')
    return {**binding,'mtime_ns':after.st_mtime_ns}

def main():
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','Final metadata snapshot has no GPU permission')
    target=AREA/'final_source_identity_inventory_v1.json';require(not target.exists(),'Preserve finalized source signatures')
    source=AREA/'completed_source_inventory_v1.json';inventory_binding=record(source);inventory=json.loads(source.read_text(encoding='utf-8-sig'))
    require(inventory['kind']=='ASR18_completed_source_inventory_v1' and inventory['schema_version']==1 and inventory['sources_count']==784
        and inventory['new_sources_count']==773 and inventory['reused_sources_count']==11 and inventory['total_source_bytes']==718_862_913_959,'Complete fixed DATA inventory required')
    plan_binding=inventory['preparation_plan'];require(record(plan_binding['path'])==plan_binding,'Completed inventory plan changed')
    for role in ('acquisition','headers'):
        closed=inventory['actual_acquisition_and_header_closures'][role];proof_binding=closed['owned_actual_exit']
        require(record(proof_binding['path'])==proof_binding and record(closed['published_marker']['path'])==closed['published_marker'],'Actual source closure binding changed')
        proof=json.loads(Path(proof_binding['path']).read_text(encoding='utf-8-sig'))
        require(proof['kind']=='ASR18_source_bound_actual_tool_exit_v1' and proof['owner']=='/root/cpu_data_preparation'
            and proof['actual_exit_code']==0 and proof['actual_tool_chunk_id'] and proof['published_marker']==closed['published_marker'],'Native/tool closure required before stat snapshot')
    plan=json.loads(Path(plan_binding['path']).read_text(encoding='utf-8-sig'))
    jobs={str(Path(j['destination']).resolve()):j for j in plan['download_jobs']+plan['reused_archives']}
    entries=[]
    for old in inventory['sources']:
        path=scoped(old['destination']);job=jobs[str(path)];before=path.stat()
        require(before.st_size==old['source_size_bytes']==job['size'] and old['source_SHA256']==job['sha256'],'Source no longer matches completed inventory')
        receipt=stable_metadata(old['source_acquisition_receipt']['path'])
        require({k:receipt[k] for k in ('path','bytes','sha256')}==old['source_acquisition_receipt'],'Completed acquisition receipt changed')
        metadata=json.loads(Path(receipt['path']).read_text(encoding='utf-8-sig'))
        require(metadata['sha256']==job['sha256'] and metadata['revision']==job['revision'],'Download-time hash/pin differs')
        index=stable_metadata(ROOT/'indexes'/('yodas3_tar_'+path.stem+'.sqlite')) if path.suffix=='.tar' else None
        after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed during frozen snapshot')
        entries.append({'path':str(path),'sha256':job['sha256'],'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'receipt':receipt,'index':index})
    require(len(entries)==len({r['path'] for r in entries})==784 and sum(r['bytes'] for r in entries)==718_862_913_959,'Complete unique source population')
    require(sum(r['index'] is not None for r in entries)==51,'Every original archive index signature required')
    report={'kind':'ASR18_final_source_identity_inventory_v1','complete':True,'plan':plan_binding,'completed_source_inventory':inventory_binding,
        'sources':entries,'builder':record(__file__),'full_source_bytes_rehashed_here':False,
        'download_time_full_SHA_and_final_stat_binding':True,'no_source_data_modified':True,'TRAIN_or_full_geometry_admitted':False}
    write_json(target,report)
    verified=FinalSourceIdentity(record(target),plan_binding,jobs)
    for path in jobs:verified.verify(path)
    print(json.dumps({'passed':True,'final_source_identity':record(target),'source_files':len(entries)}),flush=True)

if __name__=='__main__':main()
