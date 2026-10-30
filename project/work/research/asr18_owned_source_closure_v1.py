"""Hashable owned Python import closure, without importing model/runtime code."""
import ast
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
ROOTS = (PROJECT/'outputs/parakeet_turkish', PROJECT/'work/research')

def owned_closure(entries):
    pending = [Path(value).resolve() for value in entries if Path(value).suffix == '.py']
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file() or not any(path.is_relative_to(root) for root in ROOTS):
            raise ValueError('Owned source closure entry outside declared source roots: '+str(path))
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split('.')[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module' and node.args:
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    names.add(node.args[0].value.split('.')[0])
        for name in names:
            matches = [root/(name+'.py') for root in ROOTS if (root/(name+'.py')).is_file()]
            if len(matches) > 1:
                raise ValueError('Ambiguous owned import resolution: '+name)
            pending.extend(matches)
    return sorted(found)
