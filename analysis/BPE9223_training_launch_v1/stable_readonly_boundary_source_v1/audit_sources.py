"""Parse-only closure audit: never import or execute numerical modules."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

AREA = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    pairs = json.loads((AREA / 'source_pairs.json').read_text(encoding='utf-8-sig'))
    renames = {Path(p['new']).stem: Path(p['old']).stem for p in pairs}

    class Normalize(ast.NodeTransformer):
        def visit_Import(self, node):
            for item in node.names:
                item.name = renames.get(item.name, item.name)
            return node

        def visit_ImportFrom(self, node):
            node.module = renames.get(node.module, node.module)
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for new, old in renames.items():
                    node.value = node.value.replace(new, old)
            return node

    def dump(node):
        return ast.dump(Normalize().visit(node), include_attributes=False)

    entries = []
    equal_count = 0
    for pair in pairs:
        old, new = map(Path, (pair['old'], pair['new']))
        old_text, new_text = [p.read_text(encoding='utf-8-sig') for p in (old, new)]
        trees = [ast.parse(text) for text in (old_text, new_text)]
        compile(trees[1], str(new), 'exec')  # Compilation only, never exec.
        defs = [{n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))} for tree in trees]
        assert defs[0].keys() == defs[1].keys()
        changed = [name for name in defs[0] if dump(defs[0][name]) != dump(defs[1][name])]
        allowed = {'configure_session'} if new.name == 'fastmss_BPE9223_resource_CPU_boundary_v2.py' else set()
        assert set(changed) == allowed, (new.name, changed)
        unchanged = [name for name in defs[0] if name not in changed]
        equal_count += len(unchanged)
        assignments = [{n.targets[0].id: n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)} for tree in trees]
        assert assignments[0].keys() == assignments[1].keys()
        changed_assignments = [name for name in assignments[0] if dump(assignments[0][name]) != dump(assignments[1][name])]
        allowed_assignments = {'BRIDGE_SHA'} if new.name == 'fastmss_BPE9223_feature_engine_v9.py' else {'PROTOCOL_SHA256'} if new.name == 'asr18_BPE9223_CPU_window_configuration_v8.py' else set()
        assert set(changed_assignments) == allowed_assignments, (new.name, changed_assignments)
        delta = AREA / (new.stem + '.diff.txt')
        delta.write_text(''.join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True), fromfile=str(old), tofile=str(new))), encoding='utf-8', newline='\n')
        entries.append(dict(old=record(old), new=record(new), delta=record(delta),
            normalized_equal_functions_and_classes=unchanged, changed_metadata_definitions=changed,
            changed_fixed_source_hashes=changed_assignments, syntax_compilation_passed=True))
    sources = {Path(e['new']['path']).name: e['new'] for e in entries}
    assert sources['fastmss_BPE9223_plan_bridge_v9.py']['sha256'] in Path(sources['fastmss_BPE9223_feature_engine_v9.py']['path']).read_text()
    assert sources['fastmss_BPE9223_feature_protocol_v9.py']['sha256'] in Path(sources['asr18_BPE9223_CPU_window_configuration_v8.py']['path']).read_text()
    report = dict(kind='ASR18_stable_readonly_boundary_numerical_closure_AST_audit_v1', passed=True,
        producer='/root/label_bpe_audit', source_only=True, numerical_modules_imported=False,
        actual_audio_or_model_or_GPU_jobs=0, normalized_equal_definition_count=equal_count,
        changed_metadata_function='boundary2.configure_session only',
        original_verify_resource_boundary_AST_equal=True, unchanged_resource_domains=True,
        unchanged_numerical_bodies_after_named_source_identity_normalization=True, entries=entries,
        limitations=['This is source identity/AST evidence, not actual API execution, memory fit, audio/frontend/native success or production admission.',
            'Boundary2 configure_session remains subject to ROOT exact source and genuine current authority review.'])
    out = AREA / 'source_AST_audit_v1.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, report=record(out), equal_definitions=equal_count, sources=sources), ensure_ascii=False))


if __name__ == '__main__':
    main()
