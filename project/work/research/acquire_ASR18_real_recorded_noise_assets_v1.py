"""Acquire three small public CC0 audio previews and pin three cleaner TRAIN rows."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import urllib.request

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO

P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

D=ROOT/'analysis/FastMSS_real_recorded_noise_listening_v1'
ASSETS=[
 dict(id='cafe',environment='Cafe / crowd',title='Coffee Shop.wav',creator='Trainscough5407',sound_id=404478,published_at='2017-10-12',description='Coffee Shop ambience',url='https://freesound.org/people/Trainscough5407/sounds/404478/',audio_url='https://cdn.freesound.org/previews/404/404478_5200244-hq.mp3'),
 dict(id='fan',environment='Indoor fan',title='Fan.wav',creator='Q.K.',sound_id=56236,published_at='2008-06-27',description='The hum of a fan blowing.',url='https://freesound.org/people/Q.K./sounds/56236/',audio_url='https://cdn.freesound.org/previews/56/56236_91374-hq.mp3'),
 dict(id='traffic',environment='Road traffic',title='traffic-cars-ambience.wav',creator='DotAfter',sound_id=610978,published_at='2021-12-08',description='Medium traffic with cars, buses, trucks and motorbikes.',url='https://freesound.org/people/DotAfter/sounds/610978/',audio_url='https://cdn.freesound.org/previews/610/610978_13474286-hq.mp3')]


def download(url,path,guard,maximum):
    assert not path.exists()
    size=0
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ASR18-recorded-noise-listening/1.0'}),timeout=30) as response,path.open('xb') as f:
        assert response.status==200
        while True:
            guard.wait();block=response.read(65536)
            if not block:break
            size+=len(block);assert size<=maximum;guard.pace(len(block));f.write(block)
        headers=dict(response.headers.items())
    return dict(http_status=200,bytes=size,response_headers=headers,retrieved_at_utc=now())


def main():
    guard=DataIOGuard('real_recorded_noise_asset_acquisition',heavy=False,maximum_bytes=40_000_000);guard.wait();io=MetadataIO(guard)
    import soundfile as sf
    folder=D/'assets';folder.mkdir(exist_ok=True)
    license_file=D/'CC0_primary_deed.html'
    if not license_file.exists():download('https://creativecommons.org/publicdomain/zero/1.0/',license_file,guard,1_000_000)
    output=D/'real_noise_registry_v1.json';assert not output.exists()
    entries=[]
    for item in ASSETS:
        evidence=D/(item['id']+'_primary_page.html');html=evidence.read_text(encoding='utf-8')
        assert item['audio_url'] in html and 'creativecommons.org/publicdomain/zero/1.0/' in html
        assert 'Creative Commons 0' in html and item['description'] in html
        assert 'AIGenerated' not in html
        path=folder/(item['id']+'_freesound_hq_preview.mp3')
        receipt=download(item['audio_url'],path,guard,16_000_000)
        with sf.SoundFile(path) as audio:
            assert audio.frames>0 and 1<=audio.channels<=2 and audio.samplerate in (22050,24000,32000,44100,48000,96000)
            geometry=dict(native_sample_rate=audio.samplerate,native_channels=audio.channels,native_frames=audio.frames,duration_s=audio.frames/audio.samplerate,format=audio.format,subtype=audio.subtype)
        entries.append(dict(id=item['id'],environment=item['environment'],title=item['title'],creator=item['creator'],
            source_page_url=item['url'],source_revision='Freesound sound ID '+str(item['sound_id'])+'; local evidence and audio SHA pins freeze this retrieved version',
            published_at=item['published_at'],retrieved_at_utc=receipt['retrieved_at_utc'],
            license=dict(spdx='CC0-1.0',url='https://creativecommons.org/publicdomain/zero/1.0/',audio_specific=True,evidence=io.binding(evidence),deed=io.binding(license_file)),
            audio=dict(**io.binding(path),**geometry,source_url=item['audio_url'],asset_variant='Public HQ lossy MP3 preview of uploader-recorded sound; original WAV not downloaded'),
            real_recording=True,recording_basis=item['description'],recording_authenticity_scope='Primary uploader description and dated source; no independent forensic claim',download_receipt=receipt))
        write_json(D/(item['id']+'_asset_receipt_v1.json'),entries[-1])
    registry=dict(kind='FastMSS_real_recorded_noise_asset_registry_v1',schema_version=1,owner='/root/cpu_data_preparation',created_at_utc=now(),PID=os.getpid(),
        assets=entries,production_enabled=False,original_recordings_preserved=True,Gaussian_or_synthetic_RIR_included=False,audio_specific_license_verified=True,
        intended_scope='Bounded listening preview and separately isolated real-noise module; no admitted training policy activated')
    write_json(output,registry)
    metadata=sqlite3.connect((ROOT/'analysis/BPE_weak_then_clean_v1/cleaner_resource_accounting_v1/cleaner_resource_metadata_index_v1.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
    metadata.execute('PRAGMA cache_size=-2048')
    source=P/'work/data/TR_expanded_WorldSpeech_index_v1'
    def row(ordinal):
        with (source/'offsets.u64').open('rb') as offsets,(source/'rows.jsonl').open('rb') as rows:
            offsets.seek((ordinal-1)*8);lo,hi=struct.unpack('<QQ',offsets.read(16));assert hi-lo<64000;rows.seek(lo);raw=rows.read(hi-lo);guard.pace(len(raw));r=json.loads(raw)
            assert r['language']=='tr' and r['dataset_split']=='train'
            return dict(cleaner_ordinal=ordinal,source_row=r,source_row_bytes_SHA256=hashlib.sha256(raw).hexdigest())
    selected=[];seen=set()
    for ordinal, in metadata.execute("SELECT ordinal FROM rows WHERE source_group='fleurs' AND duration BETWEEN 10 AND 18 ORDER BY ordinal"):
        r=row(ordinal);gender=r['source_row'].get('gender')
        if gender not in ('male','female') or gender in seen:continue
        seen.add(gender);r['display_voice_label']='FLEURS / source gender label: '+gender;selected.append(r)
        if len(seen)==2:break
    assert len(selected)==2
    extra=row(253799);extra['display_voice_label']='WorldSpeech / separate TRAIN recording; speaker identity unverified';selected.append(extra)
    metadata.close()
    for i,r in enumerate(selected):r['id']='tr_'+str(i+1);r['noise_conditions']=[['cafe','traffic'],['fan','cafe'],['traffic','fan']][i]
    write_json(D/'selected_TRAIN_utterances_v1.json',dict(kind='FastMSS_three_supplied_transcript_cleaner_TRAIN_clips_v1',schema_version=1,selection_before_mixing=True,
        supplied_transcript_only=True,recognizer_used=False,DEV_or_TEST_used=False,source_index_provenance=io.binding(source/'index_provenance.json'),clips=selected))
    assert not any(n in sys.modules for n in ('torch','transformers','faster_whisper'))
    print(dict(registry=io.binding(output),selected_TRAIN_manifest=io.binding(D/'selected_TRAIN_utterances_v1.json'),asset_count=3,audio_decode_for_geometry_only=True,actual_owned_tool_exit_still_required=True))


if __name__=='__main__':main()
