"""Import the pinned official NeMo loss without the unrelated ASR model stack.

No kernel/wrapper source is modified. Three parent package initializers are
replaced only in this fresh diagnostic process by namespace packages pointing
at the installed paths. The actual rnnt_loss initializer and every loss module
execute normally. This process must never load NeMo ASR models/data afterward.
"""
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sys
from types import ModuleType


def require(ok,msg):
    if not ok:raise ValueError(msg)


def prepare():
    prefixes=('nemo.collections.asr','nemo.collections.asr.parts','nemo.collections.asr.parts.numba')
    require(all(name not in sys.modules for name in prefixes),'Minimal import requires fresh NeMo ASR namespace')
    nemo=importlib.import_module('nemo')
    importlib.import_module('nemo.collections')
    require(all(name not in sys.modules for name in prefixes),'Parent import populated NeMo ASR namespace before replacement')
    root=Path(nemo.__file__).resolve().parent
    project=Path(__file__).resolve().parents[2]
    require(root.is_relative_to(project),'Existing pinned project NeMo required')
    manifest=Path(__file__).with_name('native_tdt_minimal_import_manifest_v1.json')
    pins=json.loads(manifest.read_text())
    for relative,digest in pins.items():
        require(hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,'Pinned official loss changed: '+relative)
    for name in prefixes:
        path=root.joinpath(*name.split('.')[1:])
        require(path.is_dir(),'Installed native package missing')
        package=ModuleType(name)
        package.__package__=name;package.__path__=[str(path)]
        package.__file__=str(path/'__init__.py')
        package.__spec__=ModuleSpec(name,loader=None,is_package=True)
        package.__spec__.submodule_search_locations=[str(path)]
        sys.modules[name]=package
        parent,child=name.rsplit('.',1);setattr(sys.modules[parent],child,package)
    module=importlib.import_module('nemo.collections.asr.parts.numba.rnnt_loss')
    require(module.TDTLossNumba.__module__=='nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch',
            'Actual official NeMo TDT class required')
    require(not any(name.startswith(('nemo.collections.asr.models','nemo.collections.asr.data')) for name in sys.modules),
            'Unrelated ASR stack imported')
    return {'kind':'official_TDT_minimal_parent_namespace_import_v1','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'pinned_official_loss_files':pins,'skipped_parent_initializers':list(prefixes),
            'actual_class_module':module.TDTLossNumba.__module__,'official_sources_modified':False,
            'full_NeMo_model_support_claimed':False}
