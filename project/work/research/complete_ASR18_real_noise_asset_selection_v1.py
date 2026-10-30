"""Read-only asset hash completion plus supplied TRAIN selection; no downloads."""
import hashlib,json,os,sqlite3,struct,sys
from pathlib import Path
from ASR18_continuous_data_guard_v5 import DataIOGuard,ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from fastmss_real_noise_v1 import validate_registry

def main():
    guard=DataIOGuard('recorded_noise_asset_TRAIN_selection_completion',heavy=False,maximum_bytes=32_000_000);guard.wait();io=MetadataIO(guard)
    d=ROOT/'analysis/FastMSS_real_recorded_noise_listening_v1';target=d/'selected_TRAIN_utterances_v1.json';assert not target.exists()
    validated=validate_registry(d/'real_noise_registry_v1.json',guard=guard.wait)
    for a in validated['registry']['assets']:
        assert a['download_receipt']['http_status']==200 and a['download_receipt']['bytes']==a['audio']['bytes']
    metadata=sqlite3.connect((ROOT/'analysis/BPE_weak_then_clean_v1/cleaner_resource_accounting_v1/cleaner_resource_metadata_index_v1.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    metadata.execute('PRAGMA cache_size=-2048');source=P/'work/data/TR_expanded_WorldSpeech_index_v1'
    def row(ordinal):
        with (source/'offsets.u64').open('rb') as offsets,(source/'rows.jsonl').open('rb') as rows:
            offsets.seek((ordinal-1)*8);lo,hi=struct.unpack('<QQ',offsets.read(16));assert hi-lo<64000;rows.seek(lo);raw=rows.read(hi-lo);guard.pace(len(raw));r=json.loads(raw)
            assert r['language']=='tr' and r['dataset_split']=='train'
            return dict(cleaner_ordinal=ordinal,source_row=r,source_row_bytes_SHA256=hashlib.sha256(raw).hexdigest())
    selected=[];seen=set()
    for ordinal, in metadata.execute("SELECT ordinal FROM rows WHERE source_group='fleurs' AND duration BETWEEN 10 AND 18 ORDER BY ordinal"):
        r=row(ordinal);gender=r['source_row'].get('gender')
        if type(gender) is not int or gender not in (0,1) or gender in seen:continue
        seen.add(gender);r['display_voice_label']='FLEURS TRAIN · voice '+('A' if len(selected)==0 else 'B');selected.append(r)
        if len(seen)==2:break
    assert len(selected)==2
    extra=row(253799);extra['display_voice_label']='WorldSpeech TRAIN · separate recording';selected.append(extra);metadata.close()
    for i,r in enumerate(selected):r['id']='tr_'+str(i+1);r['noise_conditions']=[['cafe','traffic'],['fan','cafe'],['traffic','fan']][i]
    write_json(target,dict(kind='FastMSS_three_supplied_transcript_cleaner_TRAIN_clips_v1',schema_version=1,selection_before_mixing=True,
        supplied_transcript_only=True,recognizer_used=False,DEV_or_TEST_used=False,speaker_ID_independently_verified=False,
        source_gender_codes_preserved_without_name_mapping=True,source_index_provenance=io.binding(source/'index_provenance.json'),clips=selected))
    completed=dict(kind='FastMSS_real_noise_asset_and_TRAIN_selection_completion_v1',at_utc=now(),PID=os.getpid(),
        passed=True,registry=validated['binding'],selected_TRAIN=io.binding(target),no_download_or_audio_decode=True,
        original_acquisition_outcome=dict(owned_session=99523,native_PID=39104,actual_exit_code=1,actual_tool_chunk='f27a54',
            reason='All3 assets and registry completed; subsequent selection assertion expected text gender labels rather than preserved numeric FLEURS codes'),
        original_outcome_not_relabelled=True,production_enabled=False)
    write_json(d/'asset_selection_completion_v1.json',completed)
    print(dict(asset_selection_completion=io.binding(d/'asset_selection_completion_v1.json'),registry=validated['binding'],selected_TRAIN=io.binding(target),actual_owned_exit_still_required=True))
    assert not any(n in sys.modules for n in ('torch','numpy','soundfile','av','transformers'))

if __name__=='__main__':main()
