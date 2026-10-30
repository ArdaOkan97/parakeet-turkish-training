"""Synthetic CPU fixture of adaptive audit's pinned strong TEST branch."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import verify_adaptive_results as worker
import benchmark_strong_tr_test as strong

# Reproduce the reviewed prerequisite fixtures in a separate owned directory.
# They contain fake model bytes and synthetic transcripts only.
source=(ROOT/'work/verify_strong_test_guards_cpu.py').read_text(encoding='utf-8')
source=source.replace("work/strong_test_cpu_fixture","work/adaptive_strong_test_cpu_fixture")
source=source.replace('strong_test_guard_cpu_verification.json','adaptive_strong_prerequisite_cpu_verification.json')
context={'__file__':str(ROOT/'work/verify_strong_test_guards_cpu.py'),'__name__':'adaptive_fixture_prerequisite'}
exec(compile(source,str(ROOT/'work/verify_strong_test_guards_cpu.py'),'exec'),context)
fixture=context['fixture'];out=context['out'];freeze=context['freeze'];run=context['run'];selected=context['selected'];testout=context['testout']
(out/'benchmark_strong_tr_test.py').write_bytes((OUTPUT/'benchmark_strong_tr_test.py').read_bytes())
frozen=strong.read(freeze)
frozen['baseline_provenance']={'summary_sha256':strong.sha256(context['devsummary'])}
strong.write_json(freeze,frozen)
f,fa=strong.verified_frozen_selection(freeze)
summary=strong.read(testout/'summary.json')
summary.update(all_full_prefixes_verified=True,all_terminal_eos_verified=True,
               frozen_selection_before_inference=fa,created_at_utc=datetime.now(timezone.utc).isoformat())
strong.write_json(testout/'summary.json',summary)
comparison=strong.cpu_compare(context['args'],f,fa,context['examples'],context['inputaudit'],context['assets'])
comparison_path=context['args'].comparison_output
worker.ROOT=fixture;worker.OUTPUT=out;worker.DATA=fixture/'work/data'
saved={'tr':{'test':(context['strong'],context['adapted'])}}
dev=strong.read(context['devsummary'])
check=worker.Checks();result=worker.strong_test(comparison_path,freeze,f,saved,dev,check)
expected_gap=100*(comparison['adapted_asr']['wer']-comparison['baseline_asr']['wer'])
tests=[{'name':'complete synthetic100-record strong TEST comparison accepted','passed':result['available'],'checks':len(check.items)},
       {'name':'synthetic positive gap remains worse and not within2pp','passed':result['candidate_minus_strong_wer_gap_percentage_points']==expected_gap and expected_gap>2 and result['small_test_within_two_percentage_points'] is False}]
original=comparison_path.read_bytes()
fixtures=[('wrong frozen run',lambda x:x.__setitem__('run_id','another_winner')),
          ('wrong frozen step',lambda x:x.__setitem__('selected_step',8)),
          ('wrong freeze SHA',lambda x:x.__setitem__('frozen_selection_sha256','0'*64)),
          ('wrong pinned model revision',lambda x:x.__setitem__('baseline_revision','wrong')),
          ('wrong decoder protocol',lambda x:x['baseline_protocol'].__setitem__('num_beams',5)),
          ('TEST-driven selection',lambda x:x.__setitem__('selection_split','FLEURS test')),
          ('wrong baseline metrics',lambda x:x['baseline_asr'].__setitem__('wer',.01)),
          ('wrong candidate metrics',lambda x:x['adapted_asr'].__setitem__('word_errors',101)),
          ('wrong bootstrap point',lambda x:x['paired_sentence_bootstrap']['wer'].__setitem__('delta_percentage_points',1.)),
          ('wrong bootstrap bounds',lambda x:x['paired_sentence_bootstrap']['wer']['ci_percentage_points'].__setitem__(0,-1.)),
          ('wrong delta direction value',lambda x:x.__setitem__('wer_delta_percentage_points',-17.)),
          ('wrong prediction provenance',lambda x:x['provenance'].__setitem__('strong_predictions_sha256','0'*64)),
          ('wrong selected export provenance',lambda x:x['frozen_selection'].__setitem__('selected_finalization_provenance_sha256','0'*64))]
for name,mutate in fixtures:
    value=copy.deepcopy(comparison);mutate(value);strong.write_json(comparison_path,value)
    try:worker.strong_test(comparison_path,freeze,f,saved,dev,worker.Checks())
    except Exception as error:tests.append({'name':name,'passed':True,'failure':str(error)})
    else:tests.append({'name':name,'passed':False})
comparison_path.write_bytes(original)
payload={'passed':all(x['passed'] for x in tests),'gpu_used':False,
         'scope':'Synthetic100-record frozen-winner strong comparison and fake model assets; no real strong TEST inference, no final adaptive quality result.',
         'auditor_sha256':worker.file_sha(OUTPUT/'verify_adaptive_results.py'),'test_count':len(tests),'tests':tests,
         'synthetic_gap_percentage_points':result['candidate_minus_strong_wer_gap_percentage_points']}
target=OUTPUT/'adaptive_strong_auditor_cpu_verification.json'
target.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'passed':payload['passed'],'tests':len(tests),'output':str(target)},indent=2))
raise SystemExit(0 if payload['passed'] else 1)
