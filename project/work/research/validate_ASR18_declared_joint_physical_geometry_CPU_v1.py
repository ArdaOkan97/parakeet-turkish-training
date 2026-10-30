"""Finite scalar-only controls; never decode audio or publish final corpus fit."""
import argparse, json, sqlite3, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_declared_joint_budget_audit_v2 import audit_index_joint_budget
from asr18_physical_feature_geometry_v2 import from_samples
A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')


def main(folder):
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'Fresh E-only scalar fixture')
    folder.mkdir(); checks = []
    def check(name, value): require(value, name); checks.append(name)
    def reject(name, call):
        try: call()
        except (ValueError, KeyError): checks.append(name); return
        raise ValueError('Expected rejection: '+name)
    source = A/'runtime_geometry_CPU_actual_02/report.json'
    cases = read_json(source)['cases']; db = folder/'fixture_geometry.sqlite'
    rows = [(index+1, r['declaration']['encoder_frames'], r['declaration']['target_tokens'],
        r['declaration']['clean16k_samples'], r['declaration']['mel_frames'], 'declared') for index, r in enumerate(cases)]
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE geometry(ordinal INTEGER PRIMARY KEY,encoder_frames INTEGER,target_tokens INTEGER,clean16k_samples INTEGER,mel_frames INTEGER,status TEXT)')
        c.executemany('INSERT INTO geometry VALUES(?,?,?,?,?,?)', rows)
    tokens = sum(row[2] for row in rows)
    def audit(): return audit_index_joint_budget(db, expected_rows=len(rows), expected_tokens=tokens)
    actual = audit()
    check('All sixteen saved actual geometry metadata rows retained', actual['passed'] and actual['retained_rows'] == 16)
    check('Full T1991 U574 logical allocation policy unchanged', actual['largest_total_joint_budget']['budget']['T'] == 1991
        and actual['largest_total_joint_budget']['budget']['U'] == 574
        and actual['largest_total_joint_budget']['budget']['total_budget_bytes'] == 791167592)
    max_mel = max(row[4] for row in rows)
    check('Physical original feature+mask maximum includes one trailing frame', actual['maximum_declared_geometry']['original_feature_and_mask_bytes_per_row'] == (max_mel+1)*513)
    check('NPY bounds separately include two headers', actual['maximum_declared_geometry']['original_feature_and_mask_NPY_upper_bytes_per_row'] == (max_mel+1)*513+8192)
    for n, valid, physical, t in ((160,1,2,1),(319,1,2,1),(1279,7,8,1),(1280,8,9,1),(16000,100,101,13)):
        shape = from_samples(n)
        check('Exact scalar original physical/valid boundary '+str(n), shape['valid_mask_frames'] == valid
            and shape['physical_feature_frames'] == physical and shape['encoder_frames'] == t)
    reject('Partial row denominator rejected', lambda: audit_index_joint_budget(db, expected_rows=15, expected_tokens=tokens))
    reject('Partial token denominator rejected', lambda: audit_index_joint_budget(db, expected_rows=16, expected_tokens=tokens-1))
    reject('Boolean expected row count rejected', lambda: audit_index_joint_budget(db, expected_rows=True, expected_tokens=tokens))
    with sqlite3.connect(db) as c: c.execute("UPDATE geometry SET status='unresolved' WHERE ordinal=1")
    blocked = audit()
    check('Unknown declaration remains counted and blocks result', not blocked['passed'] and blocked['unsupported_rows'] == 1
        and blocked['retained_rows'] == 16 and blocked['total_target_tokens'] == tokens)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE geometry SET status='declared' WHERE ordinal=1")
        c.execute('UPDATE geometry SET encoder_frames=100000,mel_frames=800000,clean16k_samples=128000000 WHERE ordinal=2')
    oversized = audit()
    check('Full DP over budget remains counted without relaxed cap', not oversized['passed'] and oversized['unsupported_rows'] == 1
        and oversized['retained_rows'] == 16 and oversized['total_target_tokens'] == tokens)
    check('No numerical/audio/frontend/model runtime imported', not any(key in sys.modules for key in ('torch','numpy','scipy','pyarrow','transformers')))
    result = {'passed': True, 'check_count': len(checks), 'checks': checks, 'source': record(__file__),
        'auditor': record(S/'bpe_declared_joint_budget_audit_v2.py'), 'saved_actual_metadata': record(source),
        'positive_synthetic_index_audit': actual, 'unsupported_rows_preserved': [blocked, oversized],
        'actual_final_population_audit_run': False, 'actual_audio_decoded': False, 'actual_optimizer_updates': 0,
        'native_or_raw_lifetime_or_worker_fit_approved': False, 'production_enabled': False}
    write_json(folder/'report.json', result); print(json.dumps({'passed': True, 'checks': len(checks), 'actual_GPU_updates': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--folder', required=True)
    main(Path(parser.parse_args().folder))
