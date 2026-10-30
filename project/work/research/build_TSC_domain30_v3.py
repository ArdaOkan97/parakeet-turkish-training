"""NEW source assembly only; every frozen source remains byte-identical."""
import ast
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]/"outputs/parakeet_turkish"
PINS={"turkish_domain_sampler_v2.py":"822d0ddb608d719112e4a4285ad42a76ef7f1e4ede0cf4d97211fe38f97c62fd",
      "train_lora_batched_tsc_v2.py":"e3df50a3b308a4db2cc39ca83c1a85772709c305576b03d812b6112a70b52484",
      "tsc_training_inputs_v2.py":"e1175f6a5e8d4cb8371052afe16b3a539d598e87ec0e1ceb4d2da0314274fbcf"}
for name,digest in PINS.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
def save(name,text):
    with (HERE/name).open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
def change(text,old,new):
    assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)

sampler=(HERE/"turkish_domain_sampler_v2.py").read_text(encoding="utf-8")
sampler=sampler.replace("from turkish_domain_sampler_v1 import LANGUAGES, fingerprint, turkish_domain",
 "from turkish_domain_sampler_v1 import LANGUAGES, fingerprint, turkish_domain as original_domain\nfrom collections import Counter")
header='''
TSC_REPO = "issai/Turkish_Speech_Corpus"
TSC_REVISION = "d79ea01392882322e8f46657cb635236af25b6d4"


def turkish_domain(row):
    if row.get("dataset") == TSC_REPO:
        if (row.get("dataset_revision") != TSC_REVISION or row.get("dataset_split") != "train"
                or row.get("training_role") != "train" or row.get("language") != "tr"):
            raise ValueError("Only approved pinned TSC TRAIN enters the CV+TSC stream")
        return "cv_tsc"
    return "fleurs" if original_domain(row) == "fleurs" else "cv_tsc"


def source_group(row):
    return "tsc" if row.get("dataset") == TSC_REPO else ("fleurs" if turkish_domain(row)=="fleurs" else "cv")


def domain_horizon(fleurs_count, other_count, epochs=1., tr_per_update=8):
    """Smallest update count meeting both exact cumulative floor quotas."""
    if (type(fleurs_count) is not int or type(other_count) is not int or min(fleurs_count,other_count)<1
            or type(tr_per_update) is not int or tr_per_update<1 or not math.isfinite(epochs) or epochs<=0):
        raise ValueError("Positive actual pools/exposure/TR quota required")
    nf,nc=math.ceil(epochs*fleurs_count),math.ceil(epochs*other_count)
    low,high=0,math.ceil(max(nf/.3,nc/.7)/tr_per_update)+2
    while low<high:
        middle=(low+high)//2;draws=middle*tr_per_update;actual_f=draws*3//10
        if actual_f>=nf and draws-actual_f>=nc:high=middle
        else:low=middle+1
    return low


'''
sampler=change(sampler,"class DomainBalancedBucketSampler:",header+"class TSCDomainBalancedBucketSampler:")
sampler=sampler.replace('"cv"','"cv_tsc"')
# The new helper's sub-source label remains CV, not the combined domain label.
sampler=sampler.replace('else "cv_tsc")','else "cv")')
sampler=sampler.replace('"schema": 2','"schema": 3').replace('state.get("schema") != 2','state.get("schema") != 3')
sampler=sampler.replace("Turkish_domain_independent_replay_sampler_v2","TSC_FLEURS30_CV_TSC70_independent_replay_sampler_v3")
sampler=change(sampler,'        self.domain_pool_sizes = {name: len(rows) for name, rows in by_domain.items()}',
 '''        self.domain_pool_sizes = {name: len(rows) for name, rows in by_domain.items()}
        self.source_groups = {sid:source_group(row) for sid,row in self.tr_by_id.items()}
        self.group_pool_sizes = dict(Counter(self.source_groups.values()))
        self.group_draws = {name:0 for name in self.group_pool_sizes}
        self.group_seen = {name:set() for name in self.group_pool_sizes}
        self.group_audio_seconds = {name:0. for name in self.group_pool_sizes}''')
sampler=change(sampler,'            "turkish_pool_fingerprint": fingerprint(tr), "replay_pool_fingerprint": fingerprint(replay)}',
 '''            "turkish_pool_fingerprint": fingerprint(tr), "replay_pool_fingerprint": fingerprint(replay),
            "sampler_family":"TSC_domain30_v3", "second_domain":"uniform_CV_plus_TSC",
            "source_group_pool_counts":self.group_pool_sizes.copy()}''')
sampler=change(sampler,'            self.domain_audio_seconds[domain] += float(row["duration"])',
 '''            self.domain_audio_seconds[domain] += float(row["duration"])
            group=self.source_groups[row["source_id"]]
            self.group_draws[group]+=1;self.group_seen[group].add(row["source_id"])
            self.group_audio_seconds[group]+=float(row["duration"])''')
sampler=change(sampler,'        return {"domain": domains, "turkish_draws": self.tr_draws,',
 '''        groups={name:{"draws":self.group_draws[name],"unique_clips_seen":len(self.group_seen[name]),
            "pool_clips":count,"repeat_visits":self.group_draws[name]-len(self.group_seen[name]),
            "audio_seen_seconds":self.group_audio_seconds[name],"fraction_unique_seen":len(self.group_seen[name])/count}
            for name,count in self.group_pool_sizes.items()}
        return {"domain": domains, "source_groups":groups, "turkish_draws": self.tr_draws,''')
tree=ast.parse(sampler); node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="updates_for_epochs")
lines=sampler.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno]=['''    def updates_for_epochs(self, epochs):
        if self.sampling_mode == "uniform":
            if not math.isfinite(epochs) or epochs<=0:raise ValueError("Positive finite exposure required")
            return math.ceil(epochs*len(self.tr_by_id)/self.tr_per_update)
        return domain_horizon(self.domain_pool_sizes["fleurs"],self.domain_pool_sizes["cv_tsc"],epochs,self.tr_per_update)
''']
sampler="".join(lines)
sampler=change(sampler,'            "domain_audio_seconds": self.domain_audio_seconds.copy()}',
 '''            "domain_audio_seconds": self.domain_audio_seconds.copy(),
            "source_group_draws":self.group_draws.copy(),
            "source_group_seen":{name:sorted(seen) for name,seen in self.group_seen.items()},
            "source_group_audio_seconds":self.group_audio_seconds.copy()}''')
sampler=change(sampler,'        self.tr_rng.setstate(state["turkish_rng"])',
 '''        expected_groups={name:0 for name in self.group_pool_sizes}
        expected_group_seen={name:set() for name in self.group_pool_sizes}
        for stream in self.turkish_streams.values():
            cycles=max(0,stream.cycles_started-1)
            counts=Counter(self.source_groups[row["source_id"]] for row in stream.rows)
            for name,count in counts.items():expected_groups[name]+=cycles*count
            if stream.cycles_started:
                for row in stream.rows[:stream.position]:expected_groups[self.source_groups[row["source_id"]]]+=1
            for sid in stream.seen:expected_group_seen[self.source_groups[sid]].add(sid)
        if (state.get("source_group_draws")!=expected_groups or state.get("source_group_seen")!=
                {name:sorted(seen) for name,seen in expected_group_seen.items()}):
            raise ValueError("Actual CV/TSC/FLEURS source coverage counters differ")
        group_audio=state["source_group_audio_seconds"]
        if (set(group_audio)!=set(self.group_pool_sizes) or any(not math.isfinite(n) or n<0 for n in group_audio.values())
                or not math.isclose(math.fsum(group_audio.values()),math.fsum(audio.values()),rel_tol=1e-10,abs_tol=1e-7)):
            raise ValueError("CV/TSC/FLEURS audio counters differ")
        self.group_draws=expected_groups.copy();self.group_seen=expected_group_seen
        self.group_audio_seconds=group_audio.copy()
        self.tr_rng.setstate(state["turkish_rng"])''')
sampler=sampler.replace('"""Checkpointable TRAIN-domain sampler with replay independent of TR draws.',
 '"""TSC domain sampler:30%FLEURS,70%uniform CV+TSC; independent replay.')
save("tsc_domain_sampler_v3.py",sampler)

inputs=(HERE/"tsc_training_inputs_v2.py").read_text(encoding="utf-8")
tree=ast.parse(inputs);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="closed_source_identity")
closed="\n".join(inputs.splitlines()[node.lineno-1:node.end_lineno])+"\n"
closed=closed.replace("root_owned_TSC_closed_source_v2","root_owned_TSC_domain30_closed_source_v3")
closed=closed.replace("common_source_recipe(source, closure)","approved_source_recipe(source, closure)")
closed=change(closed,'    require(args.epochs == 1. and args.steps == math.ceil(combined["clips"] / 8),\n            "Declare one full uniform combined-pool pass; horizon must be ceil(actualCombinedN/8), not the old28082 cap")',
 '''    require(args.epochs == 1. and args.steps == domain_horizon(2037,combined["clips"]-2037),
            "One minimum-domain pass requires exact FLEURS30/CV+TSC70 quota horizon, not the old uniform cap")''')
closed=change(closed,'"source_scientific_arguments": source_science, "source_sampling_transition": source_sampling_transition(source, closure),',
 '''"source_scientific_arguments": source_science,
            "source_sampling_transition":{"source_mode":"domain30","target_mode":"domain30",
                "FLEURS_fraction_preserved":.3,"source_replay_RNG_independent":True,"target_replay_RNG_independent":True,
                "target_other_domain":"uniform originalCV plus approved TSC", "same_replay_seed_preserved":True,
                "initial_fresh_Adam_sampler_RNG":True,"source_replay_frontier_continuation_claimed":False},''')
closed=closed.replace('"Set one uniform combined-pool pass and ceil(actualCombinedN/8) update horizon"',
 '"One full CV+TSC coverage pass with30% FLEURS; exact minimum-domain quota horizon"')
closed=closed.replace('"Explicit source sampler transition where applicable; initial new Adam/sampler/RNG, unchanged replay data with no replay-order equality claim"',
 '"Preserve30% FLEURS and independent replay; expand70%other domain CV->uniformCV+TSC; initial new Adam/sampler/RNG"')
source_identity='''"""NEW TSC domain30 source admission; immutable v2 corpus/approval functions reused."""
from pathlib import Path
from tsc_training_inputs_v2 import (policy,require,file_record,read_record,COMMON_TRAINING_KEYS,
    DOMAIN_TRAINING_KEYS,DOMAIN_SOURCES)
from tsc_domain_sampler_v3 import domain_horizon
HERE=policy.OUTPUT


def approved_source_recipe(source,closure):
    recipe=source["config_identity"]["training_arguments"]
    require(set(recipe)==set(COMMON_TRAINING_KEYS|DOMAIN_TRAINING_KEYS),"Only the reviewed original domain30 source recipe is eligible")
    identity=source["config_identity"].get("domain_sampling_identity",{})
    require(recipe["sampling_mode"]=="domain30" and recipe["fleurs_fraction"]==.3 and type(recipe["replay_seed"]) is int
        and identity.get("sampling_mode")=="domain30" and identity.get("replay_seed")==recipe["replay_seed"]
        and identity.get("turkish_seed")==recipe["seed"] and identity.get("replay_rng_independent_of_turkish") is True
        and identity.get("fleurs_fraction_numerator")==3 and identity.get("fleurs_fraction_denominator")==10
        and identity.get("batch_size")==8 and identity.get("accumulation")==2 and identity.get("replay_per_update")==8,
        "Preserve actual30% FLEURS and independent replay source identity")
    require(closure.get("allow_source_domain30_expansion") is True and closure.get("source_sampling_metadata")==identity,
        "ROOT must explicitly approve expansion of other-domain CV to CV+TSC while preserving domain30/replay")
    groups=source.get("sampling",{}).get("initial_pool",{}).get("domain",{})
    require(groups.get("fleurs",{}).get("pool_clips")==2037 and groups.get("cv",{}).get("pool_clips")==72846,
        "Actual original source domain pool membership/counts differ")
    for name,digest in DOMAIN_SOURCES.items():
        require(source["code_sha256"].get(name)==digest and policy.sha256(HERE/name)==digest,
            "Closed original domain source changed: "+name)
    return recipe.copy()


'''+closed
save("tsc_domain_training_identity_v3.py",source_identity)

worker=(HERE/"train_lora_batched_tsc_v2.py").read_text(encoding="utf-8")
worker=change(worker,'from tsc_training_inputs_v2 import annotate_tr_frame_metadata, confirm_tr_metadata_acceptance, confirm_replay_after_cache, audit_tsc_training_inputs, source_isolation_tsc, closed_source_identity',
 '''from tsc_training_inputs_v2 import annotate_tr_frame_metadata, confirm_tr_metadata_acceptance, confirm_replay_after_cache, audit_tsc_training_inputs, source_isolation_tsc
from tsc_domain_training_identity_v3 import closed_source_identity
from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler''')
worker=worker.replace("train_lora_batched_tsc_v2.py","train_lora_batched_tsc_v3.py").replace("launch_native_training_tsc_v2.py","launch_native_training_tsc_v3.py")
worker=change(worker,'    sampler = BalancedBucketSampler(tr, replay, seed=args.seed, batch_size=args.batch_size,\n                                    accumulation=args.accumulation, replay_fraction=args.replay_fraction,\n                                    bucket_size=args.bucket_size)\n    epoch_steps = math.ceil(args.epochs * len(tr) / sampler.tr_per_update)',
 '''    sampler = TSCDomainBalancedBucketSampler(tr,replay,seed=args.seed,replay_seed=args.replay_seed,
        sampling_mode=args.sampling_mode,batch_size=args.batch_size,accumulation=args.accumulation,
        replay_fraction=args.replay_fraction,fleurs_fraction=args.fleurs_fraction,bucket_size=args.bucket_size,
        cycler_factory=BucketCycler,cycler_source_sha256=file_sha(OUTPUT/"train_lora_batched_data_v11.py"))
    epoch_steps = sampler.updates_for_epochs(args.epochs)''')
worker=change(worker,'"train_lora_batched_tsc_v3.py", "tsc_training_inputs_v2.py",',
 '"train_lora_batched_tsc_v3.py", "tsc_domain_sampler_v3.py", "tsc_domain_training_identity_v3.py", "turkish_domain_sampler_v1.py", "train_lora_batched_data_v11.py", "tsc_training_inputs_v2.py",')
worker=change(worker,'                       "training_arguments":{key:arguments[key] for key in ("epochs",',
 '                       "domain_sampling_identity":sampler.identity, "closed_source":tsc_source,\n                       "training_arguments":{key:arguments[key] for key in ("sampling_mode","fleurs_fraction","replay_seed","epochs",')
worker=worker.replace("sampler.tr.draws","sampler.tr_draws").replace("len(sampler.tr.seen)","sampler.tr_unique")
worker=change(worker,'              "sampling":f"without-replacement per-stream shuffled cycles, reservoir{args.bucket_size} two-dimensional T/U length buckets; replay language uniform per clip",',
 '''              "sampling":{"mode":args.sampling_mode,"fleurs_fraction_of_TR_clips":args.fleurs_fraction,
                  "sampler_identity":sampler.identity,"initial_pool":sampler.coverage(),
                  "other_domain":"uniform CV+TSC", "replay_independent_of_Turkish_rng":True,
                  "epochs_semantics":"minimum FLEURS/CV+TSC domain exposure", "all_pool_pass_requires_each_source_seen":True},''')
worker=change(worker,'        full_epoch = abs(epoch-round(epoch)) <= sampler.tr_per_update/len(tr) + 1e-9',
 '''        previous_epochs=[row["turkish_epoch"] for row in history[:-1] if "validation" in row]
        previous_epoch=previous_epochs[-1] if previous_epochs else 0.
        full_epoch=math.floor(epoch+1e-10)>math.floor(previous_epoch+1e-10)''')
worker=change(worker,'                  "turkish_clips_seen":sampler.tr_draws,"turkish_unique_clips_seen":sampler.tr_unique,',
 '''                  "turkish_clips_seen":sampler.tr_draws,"turkish_unique_clips_seen":sampler.tr_unique,
                  "Turkish_domain_coverage":sampler.coverage(),''')
worker=change(worker,'                 "latest_weights":str(latest_path),"turkish_epochs_completed":sampler.turkish_epoch,',
 '''                 "latest_weights":str(latest_path),"turkish_epochs_completed":sampler.turkish_epoch,
                 "turkish_epoch_semantics":"minimum_domain_exposure", "Turkish_domain_coverage":sampler.coverage(),
                 "sampling_identity":sampler.identity,''')
needle='    return p\n\n\ndef validate_arguments(args):'
worker=change(worker,needle,'''    p.add_argument("--sampling-mode",choices=("domain30",),default="domain30")
    p.add_argument("--fleurs-fraction",type=float,default=.3)
    p.add_argument("--replay-seed",type=int,required=True)
    return p


def validate_arguments(args):''')
save("train_lora_batched_tsc_v3.py",worker)
print("NEW TSC domain30v3 sampler/identity/worker created; all frozen sources unchanged")
