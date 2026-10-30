"""Continue the authorized compressed pilot without rebuilding Alania metadata."""
import json,pathlib
import prepare_alania_yodas_compressed_v1 as prep
from yodas_compressed_audio_v1 import validate_pilot

def main():
    root=prep.ROOT
    policy=json.loads((root/'storage_config.json').read_text(encoding='utf-8-sig'))
    assert pathlib.Path(policy['storage_root']).resolve()==root and not policy['allow_fallback_outside_root']
    status_path=root/'analysis/alania_yodas_preparation_v1/status.json'
    prep.STATE.update(json.loads(status_path.read_text(encoding='utf-8')))
    review=json.loads((root/'analysis/alania_yodas_preparation_v1/metadata_review.json').read_text(encoding='utf-8'))
    audio=json.loads((root/'datasets/yodas3/source_audio_inventory.json').read_text(encoding='utf-8'))
    mapping={pathlib.PurePosixPath(r['path']).stem:r for r in audio['files'] if r['type']=='file'}
    selected=[];hours=0
    for entry in review['per_shard']:
        selected.append(entry);hours+=entry['candidate_hours']
        if hours>=100 or len(selected)==3:break
    total=sum(mapping[x['shard']]['size'] for x in selected)
    if total>prep.MAX_STAGED_BYTES:raise ValueError('Initial30GB pilot cap exceeded')
    prep.write_json(root/'analysis/alania_yodas_preparation_v1/pilot_stage_plan.json',
       {'kind':'compressed_pilot_candidate_staging_plan_v1','selected_shards':selected,
        'metadata_screened_candidate_hours':hours,'compressed_bytes':total,
        'status':'candidates_only_not_accepted_TRAIN','accepted_TRAIN_hours':0,
        'minimum_free_bytes':prep.STOP_FREE_BYTES,'cap_bytes':prep.MAX_STAGED_BYTES,
        'do_not_duplicate_live_download':True})
    receipt_path=root/'datasets/yodas3/compressed_download_receipt.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    completed={pathlib.Path(x['path']).stem:x for x in receipt['files']}
    prep.status(phase='pilot_additional_compressed_staging',pilot_shards=[x['shard'] for x in selected],
       pilot_metadata_screened_candidate_hours=hours,pilot_compressed_bytes=total,accepted_TRAIN_hours=0)
    meta=prep.get_json(f'https://huggingface.co/api/datasets/{prep.YODAS}/tree/{prep.YODAS_REV}/data/tr/metadata?recursive=false&limit=100')
    for entry in selected:
        shard=entry['shard'];record=mapping[shard]
        if shard in completed:
            archive=prep.output_path(root/f'datasets/yodas3/audio/{shard}.tar')
            if archive.stat().st_size!=record['size']:raise ValueError('Previously verified archive size changed')
            continue
        md=next(x for x in meta if x['path']==f'data/tr/metadata/{shard}.parquet')
        prep.download(prep.YODAS,prep.YODAS_REV,md,root/f'datasets/yodas3/metadata/{shard}.parquet')
        prep.status(phase='downloading_compressed_pilot_audio',active_file=record['path'],
                    active_file_downloaded_bytes=0,active_file_expected_bytes=record['size'])
        completed[shard]=prep.download(prep.YODAS,prep.YODAS_REV,record,root/f'datasets/yodas3/audio/{shard}.tar')
        prep.write_json(receipt_path,{'created_at_utc':prep.now(),'files':list(completed.values()),
                        'no_transcode':True,'permanent_WAV_copy':False})
        validate_pilot(shard,12,report_path=root/f'analysis/alania_yodas_preparation_v1/on_demand_decode_validation_{shard}.json')
    prep.status(phase='compressed_pilot_staged_not_TRAIN_admitted',compressed_shards_completed=len(completed),
       compressed_audio_bytes=sum(x['bytes'] for x in completed.values()),
       pilot_metadata_screened_candidate_hours=hours,accepted_TRAIN_hours=0,
       next_stage='speaker/source-stratified audio/transcript QC, dedup and heldout exclusion')

if __name__=='__main__':
    try:main()
    except Exception as error:
        prep.status(phase='pilot_staging_failed_partial_preserved',error=repr(error),accepted_TRAIN_hours=0)
        raise
