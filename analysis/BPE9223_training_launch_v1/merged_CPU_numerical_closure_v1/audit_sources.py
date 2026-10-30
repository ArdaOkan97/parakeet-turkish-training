"""Source-only AST and exact delta audit; imports no training modules."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

AREA = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


RENAMES = {
    "fastmss_BPE9223_plan_bridge_v8": "fastmss_BPE9223_plan_bridge_v7",
    "fastmss_BPE9223_feature_engine_v8": "fastmss_BPE9223_feature_engine_v7",
    "fastmss_BPE9223_cpu_common_v8": "fastmss_BPE9223_cpu_common_v7",
    "fastmss_BPE9223_feature_protocol_v8": "fastmss_BPE9223_feature_protocol_v7",
    "produce_FastMSS_BPE9223_reference_CPU_v8": "produce_FastMSS_BPE9223_reference_CPU_v7",
    "produce_FastMSS_BPE9223_frontend_CPU_v8": "produce_FastMSS_BPE9223_frontend_CPU_v7",
    "asr18_BPE9223_source_context_v5": "asr18_BPE9223_source_context_v4",
    "asr18_BPE9223_CPU_window_configuration_v7": "asr18_BPE9223_CPU_window_configuration_v6",
    "run_ASR18_BPE9223_CPU_leaf_observer_v8.py": "run_ASR18_BPE9223_CPU_window_v7.py",
}


class Normalize(ast.NodeTransformer):
    def visit_Import(self, node):
        for item in node.names:
            item.name = RENAMES.get(item.name, item.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = RENAMES.get(node.module, node.module)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for new, old in RENAMES.items():
                node.value = node.value.replace(new, old)
        return node


def dump(node):
    return ast.dump(Normalize().visit(node), include_attributes=False)


ALLOWED_CHANGED_DEFINITIONS = {
    "fastmss_BPE9223_plan_bridge_v8.py": {"bridge_plan"},
    "fastmss_BPE9223_feature_protocol_v8.py": {"verify_source_contract"},
}
ALLOWED_CHANGED_ASSIGNS = {
    "fastmss_BPE9223_feature_engine_v8.py": {"BRIDGE_SHA"},
    "asr18_BPE9223_CPU_window_configuration_v7.py": {"PROTOCOL_SHA256"},
}


def main():
    pairs = json.loads((AREA / "source_pairs.json").read_text(encoding="utf-8-sig"))
    entries = []
    equivalent_total = 0
    for pair in pairs:
        old_path, new_path = map(Path, (pair["old"], pair["new"]))
        old_text, new_text = (p.read_text(encoding="utf-8-sig") for p in (old_path, new_path))
        old_ast, new_ast = ast.parse(old_text), ast.parse(new_text)
        compile(new_ast, str(new_path), "exec")  # Compile syntax only; never execute/import.
        old_defs = {n.name: n for n in old_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        new_defs = {n.name: n for n in new_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        assert old_defs.keys() == new_defs.keys()
        changed = [name for name in old_defs if dump(old_defs[name]) != dump(new_defs[name])]
        assert set(changed) == ALLOWED_CHANGED_DEFINITIONS.get(new_path.name, set()), (new_path.name, changed)
        same_defs = [name for name in old_defs if name not in changed]
        equivalent_total += len(same_defs)
        old_assigns = {n.targets[0].id: n for n in old_ast.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        new_assigns = {n.targets[0].id: n for n in new_ast.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        assert old_assigns.keys() == new_assigns.keys()
        changed_assigns = [name for name in old_assigns if dump(old_assigns[name]) != dump(new_assigns[name])]
        assert set(changed_assigns) == ALLOWED_CHANGED_ASSIGNS.get(new_path.name, set()), (new_path.name, changed_assigns)
        diff = "".join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True), fromfile=str(old_path), tofile=str(new_path)))
        diff_path = AREA / (new_path.stem + ".diff.txt")
        diff_path.write_text(diff, encoding="utf-8", newline="\n")
        entries.append({"old": record(old_path), "new": record(new_path), "delta": record(diff_path),
            "syntax_compile_without_execution": True, "normalized_equivalent_definitions": same_defs,
            "changed_metadata_definitions": changed, "changed_source_pin_assignments": changed_assigns})
    sources = {Path(e["new"]["path"]).name: e["new"] for e in entries}
    engine = Path(sources["fastmss_BPE9223_feature_engine_v8.py"]["path"]).read_text()
    factory = Path(sources["asr18_BPE9223_CPU_window_configuration_v7.py"]["path"]).read_text()
    assert sources["fastmss_BPE9223_plan_bridge_v8.py"]["sha256"] in engine
    assert sources["fastmss_BPE9223_feature_protocol_v8.py"]["sha256"] in factory
    assert 128_000_000 + 1_800_000_000 + 2 * 33_554_432 == 1_995_108_864 < 2_000_000_000
    report = {"kind": "ASR18_merged_CPU_numerical_closure_source_AST_audit_v1", "passed": True,
        "source_only": True, "numerical_modules_imported": False, "audio_or_model_or_GPU_jobs": False,
        "normalized_equal_definition_count": equivalent_total, "entries": entries,
        "explicit_metadata_deltas": ["Exact successor import/source paths and engine/factory SHA pins",
            "One pinned session2 to session3 handoff at window3/source60, preserving original prior bridge7",
            "Public source resource contract binds merged128000000 parent plus unchanged1800000000 leaf and two33554432 shims"],
        "resource_domains": {"merged_live_parent_bytes": 128_000_000, "leaf_bytes": 1_800_000_000,
            "inner_shim_bytes": 33_554_432, "outer_shim_bytes": 33_554_432,
            "inner_family_bytes": 1_961_554_432, "whole_family_bytes": 1_995_108_864},
        "limits": ["AST equivalence proves unchanged parsed source bodies after named identity substitutions; it is not runtime equality or actual resource fit",
            "Changed bridge/session and public resource validation definitions require ROOT source/config review",
            "No merged-task completion or process exit proof is created by this source-only audit"]}
    out = AREA / "source_AST_audit_v1.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "report": record(out), "equivalent_definitions": equivalent_total,
        "sources": sources}, ensure_ascii=False))


if __name__ == "__main__":
    main()
