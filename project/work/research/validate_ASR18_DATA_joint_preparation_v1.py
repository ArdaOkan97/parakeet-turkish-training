"""DATA-only fail-closed and whole-scalar joint controls; no final DATA claim."""
import argparse,json,sqlite3,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from bpe_final_DATA_population_v1 import final_data_population,REQUIRED
from bpe_declared_joint_budget_audit_v1 import audit_index_joint_budget
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only finite control');f.mkdir(exist_ok=False)
    checks=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    source=(S/'bpe_final_DATA_population_v1.py').read_text()
    require('native_envelope_admission' not in source and 'validate_declared(' in source and 'validate_carriers(' in source
        and 'audit(bindings' in source and 'reference_clips' in source,'Separate actual source/data/declared/denominator checks without native circularity')
    checks.append('DATA foundation retains source/carrier/reference/declared/denominator checks with no circular native approval')
    require('geometry_admission' not in REQUIRED and 'declared_geometry_actual_closed' in REQUIRED and 'carrier_coverage_actual_closed' in REQUIRED,'Independent final actual DATA requirements')
    checks.append('Actual final DATA/carrier/geometry exits remain mandatory')
    base={'kind':'ASR18_final_data_population_for_bounded_preparation_v1','complete':True,'pass_count':2}
    reject('Empty DATA bindings cannot arm preparation',lambda:final_data_population({**base,'bindings':{}}))
    reject('Partial DATA/native masquerade rejected',lambda:final_data_population({**base,'bindings':{'geometry_admission':{}}}))
    report=json.loads((A/'runtime_geometry_CPU_actual_02/report.json').read_text(encoding='utf-8'));db=f/'fixture_geometry.sqlite';cases=report['cases']
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE geometry(ordinal INTEGER PRIMARY KEY,encoder_frames INTEGER,target_tokens INTEGER,clean16k_samples INTEGER,mel_frames INTEGER,status TEXT)')
        c.executemany('INSERT INTO geometry VALUES(?,?,?,?,?,?)',[(i+1,r['declaration']['encoder_frames'],r['declaration']['target_tokens'],r['declaration']['clean16k_samples'],r['declaration']['mel_frames'],'declared') for i,r in enumerate(cases)])
    tokens=sum(r['declaration']['target_tokens'] for r in cases)
    def audit():return audit_index_joint_budget(db,expected_rows=16,expected_tokens=tokens)
    valid=audit();require(valid['passed'] and valid['retained_rows']==16 and valid['unsupported_rows']==0
        and valid['maximum_declared_geometry']['encoder_frames']==1991 and valid['maximum_declared_geometry']['target_tokens']==574,'Actual saved hard geometry included')
    checks.append('All16 saved actual metadata rows including T1991/U574 satisfy unchanged conservative joint logical budget')
    reject('Partial row denominator rejected',lambda:audit_index_joint_budget(db,expected_rows=15,expected_tokens=tokens))
    reject('Partial token sum rejected',lambda:audit_index_joint_budget(db,expected_rows=16,expected_tokens=tokens-1))
    with sqlite3.connect(db) as c:c.execute("UPDATE geometry SET status='unresolved_unknown_samples' WHERE ordinal=1")
    blocked=audit();require(not blocked['passed'] and blocked['unsupported_rows']==1 and blocked['retained_rows']==16 and blocked['total_target_tokens']==tokens,'Unresolved row remains counted')
    checks.append('Unresolved declaration blocks while retaining every ordinal and all targets')
    with sqlite3.connect(db) as c:
        c.execute("UPDATE geometry SET status='declared' WHERE ordinal=1")
        c.execute('UPDATE geometry SET encoder_frames=100000,mel_frames=800000,clean16k_samples=128000000 WHERE ordinal=2')
    oversized=audit();require(not oversized['passed'] and oversized['unsupported_rows']==1 and oversized['retained_rows']==16,'Oversized logical fullDP cannot pass')
    checks.append('Oversized fullDP/joint budget blocks without dropping source/targets or widening cap')
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'No audio/model/numerical runtime');checks.append('No new audio/native update or production arm')
    write_json(f/'report.json',{'passed':True,'checks':checks,'check_count':len(checks),'positive_joint_logical_audit':valid,
        'unresolved_preserved':blocked,'oversized_preserved':oversized,'source':record(__file__),
        'DATA_verifier':record(S/'bpe_final_DATA_population_v1.py'),'joint_auditor':record(S/'bpe_declared_joint_budget_audit_v1.py'),
        'synthetic_contiguous_fixture_index_based_on_saved_actual_metadata':True,'final_positive_DATA_path_not_yet_exercised':True,
        'native_or_full_worker_forecast_approved':False,'actual_native_updates':0,'production_ready':False})
    print(json.dumps({'passed':True,'checks':len(checks),'native_updates':0}))
if __name__=='__main__':main()
