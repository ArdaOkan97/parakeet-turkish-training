"""Independent header/interval declarations against saved actual decodes."""
import argparse,copy,json,sys
from fractions import Fraction
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_runtime_declared_geometry_v1 import validate_declared_row,verify_actual_row
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only runtime geometry metadata fixture');f.mkdir(exist_ok=False)
    header_path=B/'FLAC_STREAMINFO_geometry_header_fixture_v1/report.json';header=json.loads(header_path.read_text());require(header['passed'],'Actual independent STREAMINFO prefix proof')
    require(all(record(v['path'])==v for v in header['existing_canonical_CPU_decode_reference_reports']),'Pinned actual decode reports unchanged')
    headers={v['unique_id']:v['geometry'] for v in header['observations']};cases=[];checks=[]
    for path in (A/'full14_FLAC_CPU_actual_01/decoded/report.json',A/'hard_U_rows_decode_01/decoded/report.json'):
        value=json.loads(path.read_text());require(value['passed'],'Actual full source decode report')
        for item in value['rows']:
            row=item['row'];actual=item['reader_evidence'];u=len(row['tr_metadata_labels'])
            if row['dataset']=='kiraat':
                info=headers[row['unique_id']];require(info['known_total_samples'],'Unknown samples cannot become a valid declaration')
                sr=info['sample_rate'];n=info['total_native_samples']
            else:sr=48000;n=round(Fraction(str(row['end']))*sr)-round(Fraction(str(row['start']))*sr)
            clean=(n*16000+sr-1)//sr;mel=clean//160;t=(mel+7)//8
            declared={'ordinal':row['ordinal'],'source_component':1 if row['dataset']=='kiraat' else 0,'native_sr':sr,'native_samples':n,
                'clean16k_samples':clean,'mel_frames':mel,'encoder_frames':t,'target_tokens':u,'status':'declared'}
            proof=verify_actual_row(declared,row,actual,full_target_tokens=u)
            cases.append({'source_id':row['source_id'],'unique_id':row['unique_id'],'dataset':row['dataset'],'declaration':declared,
                'proof':proof,'actual_full_decode_report':record(path)})
            if len(cases)==1:control=(declared,row,actual,u)
    require(len(cases)==16 and sum(v['dataset']=='alania_yodas3' for v in cases)==6,'All14+hard2 actual cases required')
    checks.append('Ten independent FLAC header and six rational48k interval declarations match all actual native/full16k/T observations')
    declared,row,actual,u=control
    def reject(call,name):
        try:call()
        except ValueError:checks.append(name);return
        raise ValueError('Expected fail-closed: '+name)
    reject(lambda:validate_declared_row({**declared,'status':'unknown_total_samples'},row,full_target_tokens=u),'Unknown declaration stops before draw; no exclusion')
    reject(lambda:validate_declared_row({**declared,'native_samples':0},row,full_target_tokens=u),'Unknown native samplecount zero rejected')
    reject(lambda:validate_declared_row({**declared,'native_sr':True},row,full_target_tokens=u),'Boolean cannot masquerade as sample rate')
    reject(lambda:validate_declared_row({**declared,'clean16k_samples':declared['clean16k_samples']+1},row,full_target_tokens=u),'Incorrect resample rounding rejected')
    reject(lambda:verify_actual_row(declared,row,{**actual,'native_samples':actual['native_samples']-1},full_target_tokens=u),'Native actual length mismatch stops before draw')
    reject(lambda:verify_actual_row(declared,row,{**actual,'encoder_frames':actual['encoder_frames']+1},full_target_tokens=u),'Actual encoder frame mismatch stops before draw')
    reject(lambda:verify_actual_row(declared,row,{**actual,'source_id':'intentional_wrong_original_source'},full_target_tokens=u),'Wrong actual source identity rejected')
    reject(lambda:validate_declared_row(declared,{**row,'ordinal':True},full_target_tokens=u),'Boolean catalog ordinal rejected')
    reject(lambda:validate_declared_row(declared,row,full_target_tokens=max(1,u-1)),'Truncated full target count rejected')
    flagged={**row,'original_flags_json':'{"eligible":false,"recommended":false,"LID":"other","disagreement":true}'}
    require(validate_declared_row(declared,flagged,full_target_tokens=u)==declared,'Flags must never introduce eligibility selection');checks.append('Original quality/LID/disagreement flags do not affect geometry eligibility')
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'Pure metadata comparisons');checks.append('No decoder/audio/numerical/model import or new waveform read')
    write_json(f/'report.json',{'passed':True,'kind':'ASR18_runtime_declared_geometry_saved_actual_control_v1','checks':checks,'cases':cases,
        'source':record(__file__),'runtime_helper':record(S/'bpe_runtime_declared_geometry_v1.py'),'independent_header_actual':record(header_path),
        'actual_new_decode_or_native_updates':0,'actual_training_draws_consumed':0,'full_final_envelope_or_runtime_integration_proved':False,
        'heldout_clearance_or_production_readiness_claimed':False,'future_final_index_source_identity_validation_remains_required':True})
    print(json.dumps({'passed':True,'actual_saved_decode_cases':len(cases),'checks':len(checks),'new_decode_or_native_updates':0}))
if __name__=='__main__':main()
