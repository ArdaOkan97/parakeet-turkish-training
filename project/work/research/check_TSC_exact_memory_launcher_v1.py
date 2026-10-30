"""Stdlib exact-only memory admission/unchanged lifecycle source fixture."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import launch_with_memory_guard_TSC_resume_v1 as helper


def main(args):
    report=args.output.resolve()
    assert report.is_relative_to(ROOT) and not report.exists()
    scratch=report.parent/(report.stem+'_fixture');scratch.mkdir(parents=True,exist_ok=False)
    checks=[]
    def check(value,name):
        assert value,name
        checks.append(name)
    def reject(fn,name):
        try:fn()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    assets=('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest',
            '--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof','--resume-exact')
    argv=['--run-id','TSC_fixture_exact_same_ID']
    for flag in assets:
        path=scratch/(flag.removeprefix('--')+'.fixture')
        path.write_text('Synthetic admission bytes; not a corpus/checkpoint/approval',encoding='utf-8')
        argv += [flag,str(path)]
    records=helper.tsc_argument_records(['--',*argv])
    check(len(records)==9 and 'resume_exact' in records and 'resume_weights' not in records,'exact_checkpoint_and_eight_original_asset_records')
    check(all(item['sha256']==helper.file_sha(item['path']) for item in records.values()),'all_actual_fixture_asset_bytes_bound')
    reject(lambda:helper.tsc_argument_records(argv+['--resume-weights',str(scratch/'resume-exact.fixture')]),'weights_plus_exact_rejected')
    reject(lambda:helper.tsc_argument_records(argv+['--resume-runtime-migration',str(scratch/'resume-exact.fixture')]),'migration_rejected')
    absent=argv[:argv.index('--resume-exact')]
    reject(lambda:helper.tsc_argument_records(absent),'missing_fullstate_rejected')
    reject(lambda:helper.tsc_argument_records(argv+['--resume-exact',str(scratch/'resume-exact.fixture')]),'duplicate_fullstate_rejected')
    fresh=[value.replace('--resume-exact','--resume-weights') for value in argv]
    reject(lambda:helper.tsc_argument_records(fresh),'initial_warmstart_wrong_route_rejected')
    old=ast.parse((HERE/'launch_with_memory_guard_tsc_v1.py').read_text(encoding='utf-8'))
    new=ast.parse((HERE/'launch_with_memory_guard_TSC_resume_v1.py').read_text(encoding='utf-8'))
    for name in ('safe_sample','creation_identity','validate_capacity','LiveTelemetrySampler','UpdateTelemetry','telemetry_reclaimer_class','argument_identity','parser'):
        before=next(node for node in old.body if getattr(node,'name',None)==name)
        after=next(node for node in new.body if getattr(node,'name',None)==name)
        check(ast.dump(before)==ast.dump(after),'unchanged_'+name)
    check(helper.validate_capacity(16e9,8)['cap_bytes']==8_000_000_000,'same_cap8')
    reject(lambda:helper.validate_capacity(16e9,9),'cap9_rejected')
    # Execute the actual wrapper's pre-Torch validate-only dispatch.
    options=SimpleNamespace(provenance=scratch/'io.json',cache_provenance=scratch/'cache.json',memory_provenance=scratch/'memory.json',
          worker_identity=scratch/'identity.json',cap_gb=8.,threshold_gb=4.,max_events=128,retry_seconds=30.,validate_only=True,arguments=['--',*argv])
    helper.main(options)
    check('torch' not in sys.modules,'actual_validate_only_imported_no_Torch')
    check(not any((scratch/name).exists() for name in ('io.json','cache.json','memory.json','identity.json')),'validate_only_generated_no_runtime_evidence')
    encoded={'schema':1,'kind':'TSC_exact_memory_launcher_stdlib_admission_verification_v1','passed':True,'checks':len(checks),
       'check_names':checks,'helper_sha256':helper.file_sha(helper.__file__),'fixture_sha256':helper.file_sha(__file__),
       'old_initial_memory_launcher_sha256':helper.file_sha(HERE/'launch_with_memory_guard_tsc_v1.py'),
       'actual_approved_TSC_assets_or_checkpoint_read':False,'Torch_NeMo_GPU_Win32_or_network_used':False,
       'scope':'Synthetic bytes test actual stdlib admission and source equality only; root must observe actual resumed hardware/full-state identity.'}
    with report.open('x',encoding='utf-8') as stream:json.dump(encoded,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'report':str(report)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
