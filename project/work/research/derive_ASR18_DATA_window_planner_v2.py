"""Create a NEW final-metadata lifetime guard successor; preserve executed v1."""
import ast,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require

source=(S/'bpe_final_DATA_window_planner_v1.py').read_text(encoding='utf-8')
source=source.replace('from bpe_declared_geometry_index_v1 import DeclaredGeometryIndex,open_final,signature',
    'from bpe_declared_geometry_index_v1 import DeclaredGeometryIndex,open_final,signature\nfrom bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks')
source=source.replace("bindings=validated['bindings'];self.population=validated['retained_rows'];self.signatures=[]",
    "bindings=validated['bindings'];self.population=validated['retained_rows'];self.signatures=[]\n        self.metadata_locks=FinalMetadataReadLocks([bindings[k]['path'] for k in ('final_catalog','final_labels','declared_geometry_index')])")
source=source.replace('for connection in self.connections:connection.close()\n            raise',
    'for connection in self.connections:connection.close()\n            self.metadata_locks.close()\n            raise')
source=source.replace("if hasattr(self,'geometry'):self.geometry.close()", "if hasattr(self,'geometry'):self.geometry.close()\n        self.metadata_locks.close()")
ast.parse(source);target=S/'bpe_final_DATA_window_planner_v2.py';require(not target.exists(),'Preserve existing successor');target.write_text(source,encoding='utf-8',newline='\n')
test=(P/'work/research/validate_ASR18_final_DATA_window_planner_v1.py').read_text(encoding='utf-8').replace(
    'from bpe_final_DATA_window_planner_v1 import','from bpe_final_DATA_window_planner_v2 import').replace(
    "'bpe_final_DATA_window_planner_v1.py'","'bpe_final_DATA_window_planner_v2.py'")
test=test.replace("# Only owned fixture metadata changes. Runtime source signatures must\n        # poison this resolver before an altered row can become a draw.\n        with sqlite3.connect(paths['final_catalog']) as c:c.execute('UPDATE clips SET text=? WHERE ordinal=?',('changed',first))\n        reject('Changed finalized DB signature blocks before prepared draw',lambda:resolver.row(first))\n        reject('Poisoned source resolver cannot be reused',lambda:resolver.row(rows[1]['ordinal']))",
    "# Read-only Windows lifetime handles must deny mutation and replacement\n        # even when filesystem mtime would be deferred until handle closure.\n        for name in ('final_catalog','final_labels','declared_geometry_index'):\n            try:\n                with paths[name].open('r+b') as stream:pass\n            except PermissionError:checks.append('Write handle denied while validated '+name+' is in use')\n            else:raise ValueError('A finalized DB remained writable under its read-only lifetime lock')\n        require(resolver.row(first)['broad_original']['text']==rows[0]['text'],'Original cached row preserved under lock')\n        checks.append('Final original row remains exact after denied writers')")
test=test.replace('finally:resolver.close()',"finally:resolver.close()\n    # Owned fixture mutation after lock release must fail the original full hash\n    # on a fresh resolver, before a provisional row can be prepared.\n    with sqlite3.connect(paths['final_catalog']) as c:c.execute('UPDATE clips SET text=? WHERE ordinal=?',('changed',rows[0]['ordinal']))\n    reject('Changed final DB rejected by full hash on next lifetime',lambda:FinalBroadRows(protocol))")
ast.parse(test);out=P/'work/research/validate_ASR18_final_DATA_window_planner_v2.py';require(not out.exists(),'Preserve control successor');out.write_text(test,encoding='utf-8',newline='\n')
