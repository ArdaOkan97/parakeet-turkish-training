"""Snapshot finalized original source signatures and bounded metadata hashes.

Full source SHA checks are the observed completed acquisition contract. This
does not reread718GB payloads or infer metadata/label/carrier completion.
"""
import argparse,datetime,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require,scoped
from bpe_broad_source_identity_v1 import FinalSourceIdentity
B=Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1')

def read_bound(pin):
    require(type(pin) is dict and set(pin)=={'path','bytes','sha256'} and type(pin['bytes']) is int
        and 0<pin['bytes']<=8*1024**2 and record(scoped(pin['path']))==pin,'Exact bounded metadata binding required')
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))

def bound4(path):
    path=scoped(path);before=(path.stat().st_size,path.stat().st_mtime_ns);value=record(path)
    require(before==(path.stat().st_size,path.stat().st_mtime_ns),'Metadata changed while hashing')
    return {**value,'mtime_ns':before[1]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder).resolve()
    require(f.is_relative_to(B.resolve()),'E-only source signature evidence');f.mkdir(exist_ok=False)
    target=B/'final_source_identity_inventory_v1.json';require(not target.exists(),'Preserve any existing finalized source snapshot')
    inventory_pin=record(B/'completed_source_inventory_v1.json');inventory=read_bound(inventory_pin)
    wrapper_pin=record(B/'completed_source_inventory_owned_tool_actual_CLOSED0_v1.json');wrapper=read_bound(wrapper_pin)
    require(wrapper['kind']=='ASR18_source_bound_actual_tool_exit_v1' and wrapper['owner']=='/root/cpu_data_preparation'
        and type(wrapper['actual_exit_code']) is int and wrapper['actual_exit_code']==0 and wrapper['actual_tool_chunk_id']
        and wrapper['published_marker']==inventory_pin,'Genuine observed full inventory producer0 required')
    require(inventory['kind']=='ASR18_completed_source_inventory_v1' and inventory['schema_version']==1
        and inventory['sources_count']==len(inventory['sources'])==784 and inventory['total_source_bytes']==718_862_913_959
        and inventory['new_sources_count']==773 and inventory['reused_sources_count']==11,'Exact full fixed assets required')
    plan_pin=inventory['preparation_plan'];plan=read_bound(plan_pin)
    jobs={str(scoped(v['destination'])):v for v in plan['download_jobs']+plan['reused_archives']}
    require(len(jobs)==784,'Every pinned new/reused source exactly once')
    closed=inventory['actual_acquisition_and_header_closures'];markers={}
    for role in ('acquisition','headers'):
        proof=read_bound(closed[role]['owned_actual_exit']);marker=read_bound(closed[role]['published_marker'])
        require(proof['kind']=='ASR18_source_bound_actual_tool_exit_v1' and proof['owner']=='/root/cpu_data_preparation'
            and type(proof['actual_exit_code']) is int and proof['actual_exit_code']==0 and proof['actual_tool_chunk_id']
            and proof['published_marker']==closed[role]['published_marker'] and type(marker['actual_exit_code']) is int
            and marker['actual_exit_code']==0 and marker['status']=='completed','Actual source/header closures required')
        markers[role]=marker
    require(markers['acquisition']['completed_files']==773 and markers['headers']['completed_archives']==len(markers['headers']['indexes'])==51,'All source/header stages required')
    indexes={str(scoped(v['index'])):v for v in markers['headers']['indexes']};require(len(indexes)==51,'Unique header indexes')
    entries=[];seen=set();metadata_bytes=0;source_signatures=[]
    for item in inventory['sources']:
        path=scoped(item['destination']);key=str(path);require(key not in seen and key in jobs,'Unique fixed source destination required');seen.add(key);job=jobs[key]
        require(item['source_SHA256']==job['sha256'] and item['source_size_bytes']==job['size'] and item['revision']==job['revision'],'Inventory differs from pinned plan')
        stat=path.stat();require(stat.st_size==job['size'],'Final compressed source size differs');source_signatures.append((key,stat.st_size,stat.st_mtime_ns))
        receipt_pin=item['source_acquisition_receipt'];receipt=read_bound(receipt_pin);receipt_path=path.with_suffix(path.suffix+'.receipt.json')
        require(Path(receipt_pin['path']).resolve()==receipt_path.resolve() and receipt['sha256']==job['sha256'] and receipt['revision']==job['revision'],'Exact source acquisition receipt required')
        receipt4=bound4(receipt_path);metadata_bytes+=receipt4['bytes'];index4=None
        if path.suffix=='.tar':
            index_path=path.parents[3]/'indexes'/('yodas3_tar_'+path.stem+'.sqlite');header=indexes[str(index_path.resolve())]
            require(header['archive_sha256']==job['sha256'],'Header index bound to another archive')
            index4=bound4(index_path);metadata_bytes+=index4['bytes'];require(index4['sha256']==header['index_sha256'],'Immutable completed header index changed')
        entries.append({'path':key,'sha256':job['sha256'],'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'receipt':receipt4,'index':index4})
    require(seen==set(jobs) and sum(v['bytes'] for v in entries)==718_862_913_959,'Whole fixed source inventory reconciles')
    for name,size,mtime in source_signatures:
        stat=Path(name).stat();require((stat.st_size,stat.st_mtime_ns)==(size,mtime),'Source changed during signature inventory')
    value={'kind':'ASR18_final_source_identity_inventory_v1','complete':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'owner':'/root/memory_diagnosis','plan':plan_pin,'completed_source_inventory':inventory_pin,'completed_source_inventory_actual0':wrapper_pin,
        'actual_acquisition_and_header_closures':closed,'sources':entries,'source_count':784,'total_original_source_bytes':718_862_913_959,
        'all_source_receipts_and_header_indexes_rehashed':True,'full_source_payloads_rehashed':False,
        'download_time_full_SHA256_verification_and_actual_closures_preserved':True,'source':record(__file__),
        'actual_optimizer_updates':0,'TRAIN_admitted':False,'production_enabled':False}
    provisional=f/'inventory_candidate.json';write_json(provisional,value)
    verifier=FinalSourceIdentity(record(provisional),plan_pin,jobs)
    for entry in entries:verifier.verify(entry['path'])
    # Publish only after all fixed entries and the actual runtime identity
    # verifier succeed. The separate outer actual0 proof follows process exit.
    write_json(target,value)
    write_json(f/'report.json',{'passed':True,'inventory':record(target),'source':record(__file__),'source_count':784,
        'actual_runtime_verifier_checked_all_sources':True,'source_payloads_rehashed':False,'metadata_bytes_read_at_least':metadata_bytes,
        'full_catalog_labels_carrier_geometry_finalization_claimed':False,'actual_optimizer_updates':0,'production_enabled':False})
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','pyarrow','transformers')),'Metadata-only source signatures')
    print(json.dumps({'passed':True,'source_count':784,'inventory':record(target),'source_payloads_rehashed':False}))
if __name__=='__main__':main()
