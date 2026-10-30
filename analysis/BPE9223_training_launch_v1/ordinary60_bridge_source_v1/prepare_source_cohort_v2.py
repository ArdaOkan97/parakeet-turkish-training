"""Exact source-only cohort with dependency hashes resolved before sealing."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

PROJECT = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA = Path(__file__).resolve().parent
COHORT = (
    ('outputs/parakeet_turkish/fastmss_BPE9223_plan_bridge_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_plan_bridge_v10.py'),
    ('outputs/parakeet_turkish/fastmss_BPE9223_feature_engine_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_feature_engine_v11.py'),
    ('outputs/parakeet_turkish/fastmss_BPE9223_cpu_common_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_cpu_common_v11.py'),
    ('outputs/parakeet_turkish/fastmss_BPE9223_feature_protocol_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_feature_protocol_v11.py'),
    ('work/research/produce_FastMSS_BPE9223_reference_CPU_v9.py', 'work/research/produce_FastMSS_BPE9223_reference_CPU_v11.py'),
    ('work/research/produce_FastMSS_BPE9223_frontend_CPU_v9.py', 'work/research/produce_FastMSS_BPE9223_frontend_CPU_v11.py'),
    ('outputs/parakeet_turkish/asr18_BPE9223_source_context_v6.py', 'outputs/parakeet_turkish/asr18_BPE9223_source_context_v8.py'),
    ('outputs/parakeet_turkish/asr18_BPE9223_CPU_window_configuration_v8.py', 'outputs/parakeet_turkish/asr18_BPE9223_CPU_window_configuration_v10.py'),
    ('work/research/run_ASR18_BPE9223_CPU_window_v11.py', 'work/research/run_ASR18_BPE9223_CPU_window_v13.py'),
    ('work/research/run_ASR18_BPE9223_CPU_leaf_observer_v9.py', 'work/research/run_ASR18_BPE9223_CPU_leaf_observer_v11.py'),
)
REPLACEMENTS = tuple((Path(old).stem, Path(new).stem) for old, new in COHORT)
OLD = "merged_transition = window_index == 3 and plan['initial_committed_updates'] == 60"
NEW = OLD + " and not same(prior['session'], session_binding)"
BRIDGE_OLD = '5be587feb5066f154d5c908c7b5a4410bf2b3fad0320a6794ade4f90c29faef4'
PROTOCOL_OLD = 'bd21ce30b41cf85ec5398ff4f340786c413477197bd235702fade8992bdaaab1'


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def names(text, replacements):
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main():
    assert not (AREA / 'source_handoff_v2.json').exists()
    assert all(not (PROJECT / new).exists() for _, new in COHORT[1:])
    prepared = []
    hashes = {}
    for old, new in COHORT:
        before = (PROJECT / old).read_text(encoding='utf-8')
        after = names(before, REPLACEMENTS)
        hash_change = None
        if old == COHORT[0][0]:
            assert after.count(OLD) == 1
            after = after.replace(OLD, NEW)
            assert (PROJECT / new).read_text(encoding='utf-8') == after, 'Reuse the exact already sealed corrected bridge10'
        if old == COHORT[1][0]:
            assert after.count(BRIDGE_OLD) == 1
            hash_change = (BRIDGE_OLD, hashes[COHORT[0][1]])
        if old == COHORT[7][0]:
            assert after.count(PROTOCOL_OLD) == 1
            hash_change = (PROTOCOL_OLD, hashes[COHORT[3][1]])
        if hash_change:
            after = after.replace(*hash_change)
        ast.parse(after)
        normalized = after
        if hash_change:
            normalized = normalized.replace(hash_change[1], hash_change[0])
        if old == COHORT[0][0]:
            normalized = normalized.replace(NEW, OLD)
        normalized = names(normalized, tuple((b, a) for a, b in REPLACEMENTS))
        assert normalized == before
        assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(ast.parse(before), include_attributes=False)
        hashes[new] = hashlib.sha256(after.encode('utf-8')).hexdigest()
        prepared.append((old, new, before, after, hash_change))
    rows = []
    for old, new, before, after, hash_change in prepared:
        target = PROJECT / new
        if old != COHORT[0][0]:
            with target.open('x', encoding='utf-8', newline='') as stream:
                stream.write(after)
        assert pin(target)['sha256'] == hashes[new]
        diff = AREA / (Path(old).stem + '_to_' + Path(new).stem + '_v2.diff.txt')
        with diff.open('x', encoding='utf-8', newline='') as stream:
            stream.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=old, tofile=new))
        rows.append({'previous': pin(PROJECT / old), 'source': pin(target), 'delta': pin(diff),
            'explicit_static_dependency_hash_change': None if hash_change is None else {'before':hash_change[0], 'after':hash_change[1]},
            'full_source_and_AST_inverse_equal': True,
            'numerical_body_unchanged_after_explicit_source_name_and_pin_normalization': True})
    value = {'kind':'ASR18_ordinary_same_session60_exact_CPU_dependency_source_handoff_v2',
        'prepared_by':'/root/label_bpe_audit', 'source_generator':pin(Path(__file__)),
        'scope':'source_preparation_only_unexecuted', 'cohort':rows, 'source_count':len(rows),
        'semantic_delta':{'before':OLD,'after':NEW,'effect':'Same-session60 stays strict ordinary path; different prior session retains all existing exact historical checks'},
        'static_dependency_checks':{'engine_BRIDGE_SHA':hashes[COHORT[0][1]],'factory_PROTOCOL_SHA256':hashes[COHORT[3][1]],'pins_computed_from_exact_final_dependency_bytes':True},
        'earlier_unadmitted_cohort_v1_preserved':True,
        'unchanged':['numerical bodies','labels/targets/geometry/normalizer/frontend','all resource limits and task/process0 semantics','all old sources and RUN04 artifacts'],
        'source_modules_imported_or_executed':False,'GPU_audio_model_data_jobs':0,'production_admission_claimed':False}
    output = AREA / 'source_handoff_v2.json'
    with output.open('x', encoding='utf-8', newline='') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps({'handoff':pin(output),'sources':[row['source'] for row in rows]}))


if __name__ == '__main__':
    main()
