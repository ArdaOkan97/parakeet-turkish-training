"""Stdlib AST/field fixtures only; never import Trackio or read live logs."""
from __future__ import annotations
import ast
import copy
from datetime import datetime, timezone
import io
import time
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def run():
    path = OUT / "track_continuation_v8.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old = ast.parse((OUT / "track_continuation_v7.py").read_text(encoding="utf-8"))
    tests = 0; rejections = 0
    def check(value):
        nonlocal tests
        assert value
        tests += 1
    def rejected(call):
        nonlocal tests, rejections
        try: call()
        except ValueError: tests += 1; rejections += 1
        else: raise AssertionError("Invalid synthetic fields accepted")
    functions = {node.name:node for node in tree.body if isinstance(node, ast.FunctionDef)}
    previous = {node.name:node for node in old.body if isinstance(node, ast.FunctionDef)}
    for name in ("finite_number", "resume_context", "values_for", "acquisition_context", "acquisition_values_for", "continued_acquisition_context", "continued_acquisition_values_for", "adaptation_context", "adaptation_values_for"):
        check(ast.dump(functions[name], include_attributes=False) == ast.dump(previous[name], include_attributes=False))
    orig_tree = ast.parse((OUT / "track_experiment.py").read_text(encoding="utf-8"))
    metric = next(node for node in orig_tree.body if isinstance(node, ast.FunctionDef) and node.name == "metrics_for")
    orig = {}
    exec(compile(ast.Module(body=[metric], type_ignores=[]), "actual_metrics_for_AST", "exec"), orig)
    files = {}
    class FakePath:
        def __init__(self,value): self.value = value
        def __truediv__(self,value): return FakePath(self.value+"/"+str(value))
        def __hash__(self): return hash(self.value)
        def __eq__(self,other): return isinstance(other,FakePath) and self.value==other.value
        def exists(self): return self in files
        def resolve(self): return self
        def is_relative_to(self,parent): return self.value.startswith(parent.value+'/') or self==parent
        @property
        def parent(self): return FakePath(self.value.rsplit('/',1)[0])
        def mkdir(self,**kwargs): pass
        def __str__(self): return self.value
        def open(self,mode,encoding=None):
            assert mode=='x' and self not in files
            target=self
            class Sink(io.StringIO):
                def __exit__(self,*args):
                    if args[0] is None: files[target]=json.loads(self.getvalue())
                    return super().__exit__(*args)
            return Sink()
    root = FakePath("synthetic_no_real_io")
    ns = {"math":math, "ROOT":root, "OUTPUT":root/"outputs", "original":SimpleNamespace(metrics_for=orig["metrics_for"]),
        "read_json":lambda p,default: copy.deepcopy(files.get(p,default))}
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target,ast.Name) and target.id in ("ACQUISITION_PHASE","ADAPTATION_PHASE","OLD_ADAPTATION_IDENTITY","LOW_LR_ADAPTATION_IDENTITY","CONTINUED_ACQUISITION_PHASE","CONTINUED_ACQUISITION_IDENTITY","SCHEDULED_ACQUISITION_PHASE","SCHEDULED_ACQUISITION_IDENTITY","SCHEDULED_UTILITY_SHA") for target in node.targets)]
    chosen = [functions[name] for name in ("finite_number", "resume_context", "values_for", "acquisition_context",
        "acquisition_values_for", "continued_acquisition_context", "continued_acquisition_values_for", "adaptation_context", "adaptation_values_for", "scheduled_row_lr", "scheduled_acquisition_context", "scheduled_acquisition_values_for", "mirror_state", "development_cycle", "cycle_rows")]
    exec(compile(ast.Module(body=constants+chosen,type_ignores=[]),str(path),"exec"),ns)
    def record(step, **kwargs):
        value={"step":step,"tdt_loss":1.,"replay_kl":.01,"gradient_norm":.5,
            "elapsed_seconds":1.,"peak_allocated_vram_gb":4.,"head_lr":0.,"lora_lr":0.,
            "audio_seen_seconds":1000.,"train_wall_seconds":100.,
            "turkish_audio_seen_seconds":250.,"replay_audio_seen_seconds":750.}
        value.update(kwargs); return value
    # Unchanged original exact-resume detection/interval rate remains exercised.
    history=[record(10),record(11,audio_seen_seconds=1100.,train_wall_seconds=102.)]
    exact=ns["resume_context"]({"warm_start":{"mode":"exact_resume","source_step":10}},history)
    check(ns["values_for"](history[1],exact)["resources/post_resume_audio_minutes_per_wall_minute"]==50.)
    check(ns["resume_context"]({"warm_start":{"mode":"declared_optimization","source_step":10}},history) is None)
    # The v4 acquisition fields and denominator behavior are unchanged.
    acq_config={"phase":ns["ACQUISITION_PHASE"],"source_selected_step":10,"inherited_history_steps":10,
        "config_identity":{"acquisition_phase":{"kind":ns["ACQUISITION_PHASE"],"source_selected_step":10,
            "active_optimizer_tables":3,"acquisition_lr":1e-4,"source_Adam_retained_dormant":True,"unfreeze_allowed":False}}}
    acq_row=record(11,acquisition_updates_completed=1,acquisition_lr=1e-4,new_token_update_multiplier=1,
        old_six_character_update_multiplier_applied=False,audio_seen_seconds=1100.,train_wall_seconds=102.,
        turkish_audio_seen_seconds=270.,replay_audio_seen_seconds=830.)
    acq=ns["acquisition_context"](acq_config,[history[0],acq_row])
    check(ns["acquisition_values_for"](acq_row,acq)["resources/post_acquisition_audio_minutes_per_wall_minute"]==50.)
    phase=ns["ADAPTATION_PHASE"]
    config={"phase":phase,"source_phase1_step":12,"original_source_selected_step":10,
        "selected_acquisition_updates":2,"inherited_history_steps":12,
        "config_identity":{"adaptation_phase":{"kind":phase,"source_phase1_step":12,
            "original_source_selected_step":10,"selected_acquisition_updates":2,"active_parameter_tensors":195,
            "row_lr":1e-4,"lora_lr":3e-4,"old15_head_moments_active":False,
            "original207_Adam_artifact_retained_dormant":True,"further_unfreeze_allowed":False},
            "training_arguments":{"lora_lr":3e-4},
            "adaptation_optimizers":{"fixed_shared_head_tensors":12,"kind":ns["OLD_ADAPTATION_IDENTITY"],
                "row_group":{"lr":1e-4},"lora_group":{"lr":3e-4}}}}
    baseline=record(12,acquisition_updates_completed=2)
    row=record(13,adaptation_updates_completed=1,source_phase1_step=12,original_source_selected_step=10,
        selected_acquisition_updates=2,row_Adam_clock=3,LoRA_Adam_clock=11,row_lr=1e-4,lora_lr=3e-4,
        new_token_update_multiplier=1,old_six_character_update_multiplier_applied=False,
        audio_seen_seconds=1110.,train_wall_seconds=102.,turkish_audio_seen_seconds=260.,replay_audio_seen_seconds=850.)
    context=ns["adaptation_context"](config,[baseline,row])
    values=ns["adaptation_values_for"](row,context)
    check(values["phase/adaptation_updates_completed"]==1 and values["phase/row_Adam_clock"]==3
        and values["phase/LoRA_Adam_clock"]==11)
    check(values["phase/active_parameter_tensors"]==195 and values["phase/fixed_shared_head_tensors"]==12)
    check(values["resources/post_adaptation_audio_minutes_per_wall_minute"]==55.
        and values["resources/post_adaptation_turkish_audio_minutes_per_wall_minute"]==5.
        and values["resources/post_adaptation_replay_audio_minutes_per_wall_minute"]==50.)
    zero=ns["adaptation_values_for"](baseline,context)
    check(zero["phase/row_Adam_clock"]==2 and zero["phase/LoRA_Adam_clock"]==10
        and not any(key.startswith("resources/post_adaptation") for key in zero))
    for key,bad in (("row_Adam_clock",13),("LoRA_Adam_clock",13),("adaptation_updates_completed",2),
        ("source_phase1_step",10),("row_lr",1e-5),("head_lr",1e-5),
        ("lora_lr",0.),("old_six_character_update_multiplier_applied",True),
        ("train_wall_seconds",100.),("audio_seen_seconds",float("nan")),("replay_audio_seen_seconds",800.)):
        altered=copy.deepcopy(row);altered[key]=bad
        rejected(lambda:ns["adaptation_values_for"](altered,context))
    bad=copy.deepcopy(config);bad["source_phase1_step"]=11
    rejected(lambda:ns["adaptation_context"](bad,[baseline,row]))
    rejected(lambda:ns["adaptation_context"](config,[row]))
    # Only the explicitly declared low-LR optimizer identity can use3e-5.
    low=copy.deepcopy(config);low["config_identity"]["adaptation_phase"]["lora_lr"]=3e-5
    low_optimizer=low["config_identity"]["adaptation_optimizers"]
    low_optimizer.update(kind=ns["LOW_LR_ADAPTATION_IDENTITY"],source_original_LoRA_lr=3e-4,
        active_LoRA_lr=3e-5,source_LoRA_moments_reset=False)
    low_optimizer["lora_group"]["lr"]=3e-5
    low_row=copy.deepcopy(row);low_row["lora_lr"]=3e-5
    low_context=ns["adaptation_context"](low,[baseline,low_row])
    low_values=ns["adaptation_values_for"](low_row,low_context)
    check(low_values["phase/lora_lr"]==3e-5 and low_values["phase/row_lr"]==1e-4)
    check(low_values["phase/row_Adam_clock"]==3 and low_values["phase/LoRA_Adam_clock"]==11)
    check(low_values["resources/post_adaptation_audio_minutes_per_wall_minute"]==55.)
    check(ns["adaptation_values_for"](baseline,low_context)["phase/lora_lr"]==3e-5)
    for key,bad in (("kind","unknown_kind"),("source_original_LoRA_lr",3e-5),
        ("active_LoRA_lr",3e-4),("source_LoRA_moments_reset",True)):
        altered=copy.deepcopy(low);altered["config_identity"]["adaptation_optimizers"][key]=bad
        rejected(lambda:ns["adaptation_context"](altered,[baseline,low_row]))
    altered=copy.deepcopy(low);altered["config_identity"]["adaptation_phase"]["lora_lr"]=3e-4
    rejected(lambda:ns["adaptation_context"](altered,[baseline,low_row]))
    altered=copy.deepcopy(config);altered["config_identity"]["adaptation_phase"]["lora_lr"]=3e-5
    rejected(lambda:ns["adaptation_context"](altered,[baseline,low_row]))
    altered=copy.deepcopy(low);altered["config_identity"]["training_arguments"]["lora_lr"]=3e-5
    rejected(lambda:ns["adaptation_context"](altered,[baseline,low_row]))
    rejected(lambda:ns["adaptation_values_for"](row,low_context))
    # Only actually published new-run source300/fresh phase2 DEV are queried.
    def summary(wer):return {"validation_tr":{"asr":{"wer":wer}}}
    run_id="synthetic_phase2"
    folders=[(0,.9),(10,.8),(12,.4),(13,.3)]
    for step,wer in folders:files[root/"outputs"/f"results/{run_id}_step_{step}/summary.json"]=summary(wer)
    prior=record(10,validation={"retention_gate_passed":False,"retention_wer_deltas":{"de":.03}})
    baseline["validation"]={"retention_gate_passed":True,"retention_wer_deltas":{"de":0.}}
    row["validation"]={"retention_gate_passed":True,"retention_wer_deltas":{"de":0.}}
    development=ns["development_cycle"](run_id,[prior,baseline,row],config)
    check(set(development)=={12,13})
    rows=dict(ns["cycle_rows"](config,[prior,baseline,row],development))
    check(rows[10]["phase/inherited_history"]==1 and not any(key.startswith(("retention/","dev/")) for key in rows[10]))
    check(rows[12]["dev/wer_tr_percent"]==40. and rows[13]["dev/wer_tr_percent"]==30.)
    # New continued-row phase keeps LOCAL M distinct from inherited Adam K+M.
    continued_phase=ns["CONTINUED_ACQUISITION_PHASE"]
    row_config={"phase":continued_phase,"source_selected_step":12,"original_source_selected_step":10,
        "source_acquisition_updates":2,"inherited_history_steps":12,
        "config_identity":{"acquisition_phase":{"kind":continued_phase,"source_selected_step":12,
            "source_acquisition_updates":2,"source_row_lr":1e-4,"active_optimizer_tables":3,
            "acquisition_lr":1e-3,"source_Adam_retained_dormant":True,"unfreeze_allowed":False},
            "acquisition_optimizer":{"kind":ns["CONTINUED_ACQUISITION_IDENTITY"],"source_step":10,
                "source_acquisition_updates":2,"source_row_lr":1e-4,"active_row_lr":1e-3,
                "source_moments_or_clock_reset":False,"encoder_LoRA_active":False}}}
    continued_row=copy.deepcopy(row)
    continued_row.update(acquisition_updates_completed=1,source_acquisition_updates=2,row_Adam_clock=3,
        acquisition_lr=1e-3,active_row_lr=1e-3,source_row_lr=1e-4,lora_lr=0)
    for active_lr in (1e-3,3e-4):
        c=copy.deepcopy(row_config);r=copy.deepcopy(continued_row)
        c["config_identity"]["acquisition_phase"]["acquisition_lr"]=active_lr
        c["config_identity"]["acquisition_optimizer"]["active_row_lr"]=active_lr
        r.update(acquisition_lr=active_lr,active_row_lr=active_lr)
        context_row=ns["continued_acquisition_context"](c,[baseline,r])
        got=ns["continued_acquisition_values_for"](r,context_row)
        check(got["phase/acquisition_updates_completed"]==1 and got["phase/row_Adam_clock"]==3)
        check(got["phase/acquisition_lr"]==active_lr and got["phase/lora_active"]==0 and got["phase/frozen_parameter_tensors"]==204)
        check(got["resources/post_acquisition_audio_minutes_per_wall_minute"]==55.)
        initial=ns["continued_acquisition_values_for"](baseline,context_row)
        check(initial["phase/acquisition_updates_completed"]==0 and initial["phase/row_Adam_clock"]==2)
    context_row=ns["continued_acquisition_context"](row_config,[baseline,continued_row])
    for key,bad in (("acquisition_updates_completed",3),("source_acquisition_updates",0),("row_Adam_clock",1),
        ("acquisition_lr",1e-4),("active_row_lr",3e-4),("source_row_lr",1e-3),("lora_lr",3e-5),
        ("old_six_character_update_multiplier_applied",True)):
        changed=copy.deepcopy(continued_row);changed[key]=bad
        rejected(lambda:ns["continued_acquisition_values_for"](changed,context_row))
    for key,bad in (("kind","unknown"),("source_step",9),("source_acquisition_updates",0),
        ("source_row_lr",1e-3),("source_moments_or_clock_reset",True),("encoder_LoRA_active",True)):
        changed=copy.deepcopy(row_config);changed["config_identity"]["acquisition_optimizer"][key]=bad
        rejected(lambda:ns["continued_acquisition_context"](changed,[baseline,continued_row]))
    old_phase1=record(11,acquisition_updates_completed=1,validation={"retention_gate_passed":False,"retention_wer_deltas":{}})
    fake_run="synthetic_continued_row"
    for step,wer in ((0,.99),(11,.9),(12,.4),(13,.35)):
        files[root/"outputs"/f"results/{fake_run}_step_{step}/summary.json"]=summary(wer)
    dev=ns["development_cycle"](fake_run,[old_phase1,baseline,continued_row],row_config)
    check(set(dev)=={12,13})
    new_rows=dict(ns["cycle_rows"](row_config,[old_phase1,baseline,continued_row],dev))
    check(new_rows[11]["phase/inherited_history"]==1 and not any(k.startswith(("dev/","retention/")) for k in new_rows[11]))
    check(new_rows[13]["phase/row_Adam_clock"]==3 and new_rows[13]["dev/wer_tr_percent"]==35.)
    # Scheduled phase uses static peak + exact dynamic localN, never AdamK+N.
    from_schedule=ast.parse((OUT/'bpe_newrows_scheduled_acquisition_v1.py').read_text(encoding='utf-8'))
    pure=[node for node in from_schedule.body if (isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id in ('ALLOWED_PEAKS','WARMUP_UPDATES','HORIZON_UPDATES','MIN_LR_RATIO') for target in node.targets))
        or (isinstance(node,ast.FunctionDef) and node.name in ('require','schedule_identity','lr_for_update'))]
    # require is imported in the actual module, so provide it explicitly.
    oracle={'math':math,'require':lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg))}
    exec(compile(ast.Module(body=pure,type_ignores=[]),'actual_schedule_AST','exec'),oracle)
    for peak in (1e-3,3e-4):
        schedule=oracle['schedule_identity'](peak)
        c=copy.deepcopy(row_config);c['phase']=ns['SCHEDULED_ACQUISITION_PHASE']
        phase=c['config_identity']['acquisition_phase'];optimizer=c['config_identity']['acquisition_optimizer']
        phase.update(kind=c['phase'],peak_row_lr=peak,acquisition_lr=peak,row_schedule=schedule)
        optimizer.update(kind=ns['SCHEDULED_ACQUISITION_IDENTITY'],utility_sha256=ns['SCHEDULED_UTILITY_SHA'],
            peak_row_lr=peak,active_row_lr=peak,base_lr=peak,row_schedule=schedule,
            dynamic_applied_lr_is_not_scientific_identity=True)
        scheduled_context=ns['scheduled_acquisition_context'](c,[baseline])
        zero=ns['scheduled_acquisition_values_for'](baseline,scheduled_context)
        check(zero['phase/scheduler_local_N']==0 and zero['phase/row_Adam_clock']==2
            and zero['phase/current_row_lr']==1e-4 and 'phase/last_applied_row_lr' not in zero)
        for n in (1,50,51,250,1000):
            applied=oracle['lr_for_update'](n,peak)
            check(ns['scheduled_row_lr'](n,peak)==applied)
            r=copy.deepcopy(continued_row)
            r.update(step=12+n,acquisition_updates_completed=n,scheduled_acquisition_updates=n,
                row_Adam_clock=2+n,peak_row_lr=peak,acquisition_lr=applied,active_row_lr=applied,
                current_row_lr=applied,last_applied_row_lr=applied,scheduled_next_update=(n+1 if n<1000 else None))
            got=ns['scheduled_acquisition_values_for'](r,scheduled_context)
            check(got['phase/scheduler_local_N']==n and got['phase/row_Adam_clock']==2+n
                and got['phase/schedule_peak_row_lr']==peak and got['phase/current_row_lr']==applied)
            check(got['resources/post_acquisition_audio_minutes_per_wall_minute']==55.
                and got['resources/post_acquisition_turkish_audio_minutes_per_wall_minute']==5.)
        for key,bad in (('scheduled_acquisition_updates',999),('row_Adam_clock',1000),('acquisition_lr',peak),
            ('current_row_lr',peak),('last_applied_row_lr',peak),('scheduled_next_update',1001),
            ('peak_row_lr',1e-4),('lora_lr',3e-5),('source_acquisition_updates',0)):
            altered=copy.deepcopy(r);altered[key]=bad
            rejected(lambda:ns['scheduled_acquisition_values_for'](altered,scheduled_context))
        for key,bad in (('utility_sha256','0'*64),('source_moments_or_clock_reset',True),
            ('row_schedule',{**schedule,'horizon_updates':1001}),('kind','unknown')):
            altered=copy.deepcopy(c);altered['config_identity']['acquisition_optimizer'][key]=bad
            rejected(lambda:ns['scheduled_acquisition_context'](altered,[baseline]))
        schedule_run='synthetic_scheduled_'+str(peak)
        for step,wer in ((0,.9),(11,.8),(12,.4),(1012,.3)):
            files[root/'outputs'/f'results/{schedule_run}_step_{step}/summary.json']=summary(wer)
        d=ns['development_cycle'](schedule_run,[old_phase1,baseline,r],c)
        check(set(d)=={12,1012})
        mirror=dict(ns['cycle_rows'](c,[old_phase1,baseline,r],d))
        check(mirror[11]['phase/inherited_history']==1
            and not any(key.startswith(('retention/','dev/')) for key in mirror[11]))
        check(mirror[1012]['phase/scheduler_local_N']==1000 and mirror[1012]['phase/row_Adam_clock']==1002
            and mirror[1012]['dev/wer_tr_percent']==30.)
    for n in (-1,1001,True):rejected(lambda:ns['scheduled_row_lr'](n,1e-3))
    rejected(lambda:ns['scheduled_row_lr'](1,1e-4))
    # Own state wins; otherwise ALL v7 flags persist; old fallback only if absent.
    v5=root/"work/trackio_continuation_mirror_state_v8.json"
    v4=root/"work/trackio_continuation_mirror_state_v7.json"
    v3=root/"work/trackio_continuation_mirror_state_v3.json"
    args=SimpleNamespace(state_path=v5,source_state_path=v4)
    files[v4]={"phase1":{"reference_base_v3":True,"10":{"dev/wer_tr_percent":40.}}}
    check(ns["mirror_state"](args)==files[v4])
    files[v5]={"phase2":{"reference_base_v3":True}}
    check(ns["mirror_state"](args)==files[v5]);del files[v5];del files[v4]
    files[v3]={"original":{"reference_base_v3":True,"finished":"synthetic_flag"}}
    check(ns["mirror_state"](args)==files[v3])
    check("trackio" not in sys.modules and "torch" not in sys.modules and "transformers" not in sys.modules)
    # New optional controls: AST-extracted actual startup/main/follow functions.
    # Identity and stop files exist solely in this memory-backed fake filesystem.
    calls=[]
    ns.update(datetime=datetime,timezone=timezone,hashlib=hashlib,json=json,Path=Path,
        __file__=str(path),os=SimpleNamespace(getpid=lambda:4242,getppid=lambda:3131),
        sys=SimpleNamespace(executable='SYNTHETIC_MONITOR_ENV_PYTHON'),time=time,
        write_json=lambda p,v:files.__setitem__(p,copy.deepcopy(v)),
        trackio=SimpleNamespace(finish=lambda:calls.append('finish')))
    names=('stop_requested','scoped_control_path','start_controls','publish_control_run','finish_controls','follow','main')
    exec(compile(ast.Module(body=[functions[name] for name in names],type_ignores=[]),str(path),'exec'),ns)
    control_args=SimpleNamespace(project='parakeet-turkish',follow=True,state_path=v5,source_state_path=v4,
        stop_file=root/'control/stop.txt',identity_file=root/'control/identity.json')
    actual_follow=ns['follow']
    def stop_at_first_cycle(args,control=None):
        files[args.stop_file]='SYNTHETIC_ROOT_STOP'
        return actual_follow(args,control)
    ns['follow']=stop_at_first_cycle
    check(ns['main'](control_args)==0)
    identity=files[control_args.identity_file]
    check(identity['status']=='stopped' and identity['stop_reason']=='cooperative_stop_file'
        and identity['trackio_finish_returned'] is True and calls==['finish'])
    check(identity['pid']==4242 and identity['ppid']==3131
        and identity['script_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        and identity['started_at_utc'] and identity['finished_at_utc'])
    check(identity['runstate_identity']['state_path']==str(v5)
        and identity['runstate_identity']['source_state_path']==str(v4)
        and identity['runstate_identity']['active_run_id'] is None)
    # Stale stop and identity rejection happens before any new identity write.
    stale=SimpleNamespace(**vars(control_args));stale.identity_file=root/'control/unused_identity.json'
    rejected(lambda:ns['start_controls'](stale));check(stale.identity_file not in files)
    files.pop(control_args.stop_file)
    rejected(lambda:ns['start_controls'](control_args))
    outside=SimpleNamespace(**vars(control_args));outside.identity_file=FakePath('outside_workspace/identity.json')
    rejected(lambda:ns['start_controls'](outside))
    same=SimpleNamespace(**vars(control_args));same.identity_file=same.stop_file
    rejected(lambda:ns['start_controls'](same))
    # Optional old CLI controls absent stays compatible and writes nothing.
    optional=SimpleNamespace(project='parakeet-turkish',state_path=v5,source_state_path=v4)
    check(ns['start_controls'](optional)=={'path':None,'identity':None})
    # Final failure metadata cannot claim Trackio finished successfully.
    fresh=SimpleNamespace(**vars(control_args));fresh.identity_file=root/'control/failed_identity.json'
    control=ns['start_controls'](fresh)
    ns['finish_controls'](control,'failed','trackio_finish_failed','synthetic failure',False)
    check(files[fresh.identity_file]['status']=='failed'
        and files[fresh.identity_file]['trackio_finish_returned'] is False)
    check('trackio' not in sys.modules and 'torch' not in sys.modules)
    report={"schema":1,"kind":"inactive_track_continuation_v8_stdlib_field_and_control_verification_v1",
        "passed":True,"assertions":tests,"rejections":rejections,
        "source":{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()},
        "preserved_v7":{"path":str(OUT/"track_continuation_v7.py"),"sha256":hashlib.sha256((OUT/"track_continuation_v7.py").read_bytes()).hexdigest()},
        "scheduled_utility":{"path":str(OUT/"bpe_newrows_scheduled_acquisition_v1.py"),
            "sha256":hashlib.sha256((OUT/"bpe_newrows_scheduled_acquisition_v1.py").read_bytes()).hexdigest()},
        "fixture":{"path":str(Path(__file__).resolve()),"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "scope":"AST-extracted actual metric/bridge/startup/stop functions and synthetic memory-backed fields/state/controls only.",
        "cooperative_controls":"Optional scoped fresh stop/identity; actual startup/main tested with stdlib Trackio/os stubs only.",
        "legacy_exact_resume_and_phase1_function_AST_unchanged":True,
        "phase2_rates_denominator":"Cumulative audio/TR/replay deltas divided by actual cumulative optimizer train-wall delta from chosenS+K.",
        "phase2_Adam_clocks":"rowK+N and LoRAS+N, separately from cumulativeS+K+N.",
        "Trackio_imported":False,"database_or_shared_log_reads_or_writes":False,
        "actual_phase2_worker_or_ASR_executed":False,"Tensor_or_GPU_execution":False,
        "mirror_started":False,"existing_live_v7_or_prior_mirrors_modified":False}
    destination=OUT/"track_continuation_v8_field_CPU_verification_v1.json"
    with destination.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"report":str(destination),"sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_sha256":report["source"]["sha256"],"assertions":tests,"rejections":rejections,
        "Trackio_imported":False,"actual_shared_state_or_database_access":False}))


if __name__=="__main__":run()
