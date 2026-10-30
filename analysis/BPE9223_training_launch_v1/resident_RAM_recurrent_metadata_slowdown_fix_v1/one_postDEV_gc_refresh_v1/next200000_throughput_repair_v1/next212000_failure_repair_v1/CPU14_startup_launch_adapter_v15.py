"""INACTIVE E-only launch adapter; ORIGINAL CPU14 methods and binary transport.
Only a private runpy namespace gets a subprocess proxy; no installed/global patch.
"""
import hashlib
import json
from pathlib import Path
import runpy

ORIGINAL=Path(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish/fastmss_resident_feature_stream_v14.py")
ORIGINAL_SHA="e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e"
ROOT=Path(r"E:/gpu_training_scratchpad").resolve()
WORKER_PYTHON=Path(r"E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe").resolve()
CONTEXT=None
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert pin(ORIGINAL)["sha256"]==ORIGINAL_SHA
_exports=runpy.run_path(str(ORIGINAL),run_name="CPU14_private_launch_adapter_definitions")
ResidentFeatureProducer=_exports["ResidentFeatureProducer"]
_namespace=ResidentFeatureProducer.__init__.__globals__
_original_subprocess=_namespace["subprocess"]
def configure_startup_receipts(policy,configuration_pin,*,startup_scope_path=None):
    global CONTEXT
    assert CONTEXT is None,"One immutable per-parent launch context"
    assert policy["kind"]=="ASR18_original_CPU14_startup_receipts_launch_adapter_v1"
    assert policy["adapter_source"]==pin(__file__)
    assert policy["original_CPU14_source"]==pin(ORIGINAL)
    assert policy["worker_python_path"]==str(WORKER_PYTHON)
    assert policy["additional_host_reservation_bytes"]==8388608
    assert policy["maximum_receipt_bytes"]==1048576 and policy["four_coarse_hooks_once"]is True
    assert policy["no_per_update_observer"]is True
    assert pin(configuration_pin["path"])==configuration_pin
    runner=Path(policy["runner_source"]["path"]).resolve()
    assert runner.is_relative_to(ROOT)and pin(runner)==policy["runner_source"]
    scope_path=Path(startup_scope_path or policy["execution_review_path"]).resolve()
    assert scope_path.is_relative_to(ROOT)
    CONTEXT=dict(policy=dict(policy),configuration=dict(configuration_pin),runner=runner,scope_path=scope_path)
class _OwnedLaunchSubprocess:
    def __getattr__(self,name):return getattr(_original_subprocess,name)
    def Popen(self,argv,*positional,**keywords):
        assert CONTEXT is not None,"Typed startup policy must be configured before CPU launch"
        context=CONTEXT;configuration=context["configuration"]
        expected=[str(WORKER_PYTHON),"-Xutf8","-B",str(ORIGINAL.resolve()),
            "--worker","--configuration",configuration["path"],"--private-limit","8000000000"]
        actual=list(argv)
        assert len(actual)==len(expected)
        assert Path(actual[0]).resolve()==WORKER_PYTHON and Path(actual[3]).resolve()==ORIGINAL.resolve()
        assert actual[1:3]==expected[1:3]and actual[4:]==expected[4:]
        assert pin(configuration["path"])==configuration
        assert not positional and set(keywords)=={"stdin","stdout","stderr","env","bufsize","creationflags"}
        assert keywords["stdin"]==_original_subprocess.PIPE and keywords["stdout"]==_original_subprocess.PIPE
        assert keywords["env"]["CUDA_VISIBLE_DEVICES"]=="-1"and keywords["bufsize"]==0
        folder=Path(keywords["stderr"].name).resolve().parent
        assert folder.is_relative_to(ROOT)and folder.is_dir()
        receipt=folder/"early_startup_receipts";assert not receipt.exists()
        command=[actual[0],"-Xutf8","-B",str(context["runner"]),
            "--ROOT-startup-scope",str(context["scope_path"]),"--original-source",str(ORIGINAL.resolve()),
            "--receipt-directory",str(receipt),"--expected-configuration-pin",
            json.dumps(configuration,sort_keys=True,separators=(",",":")), *actual[4:]]
        # Original Popen called EXACTLY ONCE; same pipes/env/private limit/credit.
        return _original_subprocess.Popen(command,**keywords)
_namespace["subprocess"]=_OwnedLaunchSubprocess()
