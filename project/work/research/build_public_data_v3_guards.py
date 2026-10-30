"""Derive small versioned fixtures; retain every original fixture and report."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'work/research'
def write(name,text):
    p=R/name
    if p.exists():raise FileExistsError(p)
    p.write_text(text,encoding='utf-8',newline='\n')
text=(R/'test_public_data_policy_v2_cpu.py').read_text(encoding='utf-8').replace('_v2','_v3')
text=text.replace("check('Torch-not-imported-by-policy-tests'", "check('explicit-ungrouped-TDT-policy',policy.POLICY['tokenizer_decode_group_tokens'] is False)\ncheck('new-trainer-explicit-ungrouped-TDT', 'skip_special_tokens=True, group_tokens=False' in new.read_text(encoding='utf-8'))\ncheck('Torch-not-imported-by-policy-tests'")
write('test_public_data_policy_v3_cpu.py',text)
text=(R/'verify_data_v2_trainer_core_cpu.py').read_text(encoding='utf-8').replace('_v2','_v3')
write('verify_data_v3_trainer_core_cpu.py',text)
text=(R/'test_memory_guard_v4_cpu.py').read_text(encoding='utf-8').replace('launch_with_memory_guard_v4','launch_with_memory_guard_v5').replace('launch_with_cache_reclaim_v2','launch_with_cache_reclaim_v3').replace('memory_guard_v4_final','memory_guard_v5_final')
text=text.replace("    check('final-live-sidecar-hash'", "    check('separate-global-and-self-commit-live-observations',live['sample']['system_memory']['API'].startswith('GetPerformanceInfo') and live['sample']['calling_process_commit_memory']['pid']==os.getpid())\n    check('new-telemetry-dependency-bound',live['resource_telemetry_sha256']==launch.RESOURCE_TELEMETRY_SHA256)\n    check('final-live-sidecar-hash'")
write('test_memory_guard_v5_cpu.py',text)
