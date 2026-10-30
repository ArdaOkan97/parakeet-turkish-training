"""Source-only exact successors for the ordinary same-session source60 boundary.

No generated module is imported or executed. Existing sources are immutable.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

PROJECT = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA = Path(__file__).resolve().parent
COHORT = (
    ('outputs/parakeet_turkish/fastmss_BPE9223_plan_bridge_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_plan_bridge_v10.py'),
    ('outputs/parakeet_turkish/fastmss_BPE9223_feature_engine_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_feature_engine_v10.py'),
    ('outputs/parakeet_turkish/fastmss_BPE9223_cpu_common_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_cpu_common_v10.py'),
    ('outputs/parakeet_turkish/fastmss_BPE9223_feature_protocol_v9.py', 'outputs/parakeet_turkish/fastmss_BPE9223_feature_protocol_v10.py'),
    ('work/research/produce_FastMSS_BPE9223_reference_CPU_v9.py', 'work/research/produce_FastMSS_BPE9223_reference_CPU_v10.py'),
    ('work/research/produce_FastMSS_BPE9223_frontend_CPU_v9.py', 'work/research/produce_FastMSS_BPE9223_frontend_CPU_v10.py'),
    ('outputs/parakeet_turkish/asr18_BPE9223_source_context_v6.py', 'outputs/parakeet_turkish/asr18_BPE9223_source_context_v7.py'),
    ('outputs/parakeet_turkish/asr18_BPE9223_CPU_window_configuration_v8.py', 'outputs/parakeet_turkish/asr18_BPE9223_CPU_window_configuration_v9.py'),
    ('work/research/run_ASR18_BPE9223_CPU_window_v11.py', 'work/research/run_ASR18_BPE9223_CPU_window_v12.py'),
    ('work/research/run_ASR18_BPE9223_CPU_leaf_observer_v9.py', 'work/research/run_ASR18_BPE9223_CPU_leaf_observer_v10.py'),
)
REPLACEMENTS = tuple((Path(old).stem, Path(new).stem) for old, new in COHORT)
OLD_PREDICATE = "merged_transition = window_index == 3 and plan['initial_committed_updates'] == 60"
NEW_PREDICATE = OLD_PREDICATE + " and not same(prior['session'], session_binding)"


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def replace_names(text, replacements):
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main():
    targets = [PROJECT / new for _, new in COHORT]
    assert all(not target.exists() for target in targets), 'Fresh successor source files only'
    assert not (AREA / 'source_handoff_v1.json').exists(), 'Do not replace a sealed handoff'
    prepared = []
    for old, new in COHORT:
        before = (PROJECT / old).read_text(encoding='utf-8')
        after = replace_names(before, REPLACEMENTS)
        if Path(old).name == 'fastmss_BPE9223_plan_bridge_v9.py':
            assert after.count(OLD_PREDICATE) == 1
            after = after.replace(OLD_PREDICATE, NEW_PREDICATE)
        ast.parse(after)
        normalized = after
        if Path(old).name == 'fastmss_BPE9223_plan_bridge_v9.py':
            assert normalized.count(NEW_PREDICATE) == 1
            normalized = normalized.replace(NEW_PREDICATE, OLD_PREDICATE)
        normalized = replace_names(normalized, tuple((new_name, old_name) for old_name, new_name in REPLACEMENTS))
        assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(ast.parse(before), include_attributes=False), 'Only declared predicate/import/source-pin path changes'
        assert normalized == before, 'Exact source inverse beyond the declared changes'
        prepared.append((old, new, before, after))
    rows = []
    for old, new, before, after in prepared:
        target = PROJECT / new
        with target.open('x', encoding='utf-8', newline='') as stream:
            stream.write(after)
        diff = AREA / (Path(old).stem + '_to_' + Path(new).stem + '.diff.txt')
        with diff.open('x', encoding='utf-8', newline='') as stream:
            stream.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=old, tofile=new))
        rows.append({'previous': pin(PROJECT / old), 'source': pin(target), 'delta': pin(diff),
                     'full_AST_inverse_equal': True, 'full_source_inverse_equal': True,
                     'numerical_body_unchanged_after_explicit_source_name_normalization': True})
    value = {
        'kind': 'ASR18_ordinary_same_session60_bridge_and_exact_CPU_dependency_source_handoff_v1',
        'prepared_by': '/root/label_bpe_audit',
        'source_generator': pin(Path(__file__)),
        'scope': 'source_preparation_only_unexecuted',
        'semantic_delta': {'file': str((PROJECT / COHORT[0][1]).resolve()), 'before': OLD_PREDICATE,
                           'after': NEW_PREDICATE,
                           'effect': 'Ordinary same-session source60/window3 uses the normal strict path; a different prior session retains every existing exact historical session2 check'},
        'cohort': rows,
        'source_count': len(rows),
        'unchanged': ['all numerical bodies', 'labels/targets/geometry/frontend/normalizer',
                      'resource caps/thread/guard policy', 'task vs real leaf process0 semantics',
                      'historical cross-session source60 validation', 'all old source files',
                      'all RUN04/checkpoint/journal/cache files'],
        'GPU_audio_model_data_jobs': 0,
        'source_modules_imported_or_executed': False,
        'production_admission_claimed': False,
    }
    output = AREA / 'source_handoff_v1.json'
    with output.open('x', encoding='utf-8', newline='') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'handoff': pin(output), 'sources': [row['source'] for row in rows]}))


if __name__ == '__main__':
    main()
