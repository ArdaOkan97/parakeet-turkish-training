import ast,ctypes,hashlib,json,os,pathlib,shutil,time
import psutil
P=pathlib.Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
D=pathlib.Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
class MS(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
def sample():
    m=MS();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free)
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def sha(b):return hashlib.sha256(b).hexdigest()
def pin_bytes(p,b):return dict(path=str(p),bytes=len(b),sha256=sha(b))
def pin(p):return pin_bytes(p,p.read_bytes())
def astsha(b):return sha(ast.dump(ast.parse(b.decode('utf-8-sig')),include_attributes=False).encode())
def write_source(p,b):
    ast.parse(b.decode('utf-8-sig'))
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return pin_bytes(p,b)
def write_json(p,obj):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)
t0=time.monotonic();s=sample()
initial=write_json(D/'projection2_reader4_APIproof7_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

import difflib
def raw_diff(old,new,source_path,target_path,labels):
    work=old;records=[];shift=0
    for tag,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if tag=='equal':continue
        at=a+shift;before=work
        assert work[at:at+(b-a)]==old[a:b]
        work=work[:at]+new[c:d]+work[at+(b-a):]
        records.append(dict(offset_before=at,old_utf8=old[a:b].decode(),new_utf8=new[c:d].decode(),before_sha256=sha(before),after_sha256=sha(work)))
        shift+=(d-c)-(b-a)
    assert work==new
    for e in reversed(records):
        at=e['offset_before'];x=e['new_utf8'].encode();y=e['old_utf8'].encode()
        assert work[at:at+len(x)]==x
        work=work[:at]+y+work[at+len(x):]
    assert work==old
    return dict(base=pin_bytes(source_path,old),candidate=pin_bytes(target_path,new),ordered_edits=records,whole_byte_inverse_exact=True,semantic_changes=labels,base_AST_sha256=astsha(old),candidate_AST_sha256=astsha(new))

PROJECTOR_BLOCK="    def _stage_key(self, keys, key, previous_max):\n        # Count exactly the same ingestion-order \"insert below current maximum\"\n        # events as the original sorted insertion, without shifting old pointers.\n        try:\n            if previous_max is None or key >= previous_max:\n                keys.append(key)\n                return key\n            keys.append(key)\n            self.stats[\"out_of_order_inserts\"] += 1\n            return previous_max\n        finally:\n            # A raised comparison/allocation must not retain the staging list in\n            # this traceback frame. Never clear a list that may become cache-owned.\n            keys = key = previous_max = None\n\n    def _merge_keys(self, existing, new_keys):\n        merged = None\n        try:\n            self.guard()\n            new_keys.sort()  # Stable sort of NEW keys only, once per frontier.\n            if not existing:\n                return new_keys  # Ownership transfers to the cache; do not clear.\n            if not new_keys:\n                return existing\n            if new_keys[0] >= existing[-1]:\n                existing.extend(new_keys)\n                return existing\n            # Stable two-way merge of two sorted runs, exactly once. Equal keys\n            # from the old run precede new keys, matching insort_right semantics.\n            merged = []\n            i = j = 0\n            while i < len(existing) and j < len(new_keys):\n                if existing[i] <= new_keys[j]:\n                    merged.append(existing[i]); i += 1\n                else:\n                    merged.append(new_keys[j]); j += 1\n                if (i + j) % 16384 == 0:\n                    self.guard()\n            while i < len(existing):\n                merged.append(existing[i]); i += 1\n                if (i + j) % 16384 == 0:\n                    self.guard()\n            while j < len(new_keys):\n                merged.append(new_keys[j]); j += 1\n                if (i + j) % 16384 == 0:\n                    self.guard()\n            return merged\n        finally:\n            # Drop local ownership only. Cold/append paths can alias cache lists.\n            existing = new_keys = merged = None\n\n    def _refresh(self, cursor):\n        self.guard()\n        stat = self.db_path.stat()\n        identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)\n        # Windows ctime is creation time. POSIX ctime changes on writes, so use\n        # device+inode there; this application is deployed only on Windows.\n        identity = identity if __import__(\"os\").name == \"nt\" else identity[:2]\n        if self.file_identity is not None and identity != self.file_identity:\n            self._reset()\n        self.file_identity = identity\n        upper = cursor.execute(\"SELECT max(id) FROM metrics\").fetchone()[0] or 0\n        if upper < self.watermark:\n            self._reset()\n        # Bounded staging spans this ONE fixed upper-ID frontier. SQL stays in\n        # original chunks256, but historical lists are never merged per chunk.\n        staged_all = []\n        staged_groups = {}\n        staged_max = {}\n        all_max = self.all_keys[-1] if self.all_keys else None\n        new_signatures = 0\n        merged_all = merged_groups = rows = row = metrics = signature = key = None\n        try:\n            while self.watermark < upper:\n                self.guard()\n                rows = cursor.execute(\n                    \"SELECT id, run_id, timestamp, metrics FROM metrics \"\n                    \"WHERE id > ? AND id <= ? ORDER BY id LIMIT 256\",\n                    (self.watermark, upper),\n                ).fetchall()\n                if not rows:\n                    raise RuntimeError(\"Telemetry metrics changed during append-only projection\")\n                for row in rows:\n                    row_id, run_id, timestamp, raw = row\n                    if run_id == self.run_id:\n                        if len(self.all_keys) + len(staged_all) >= self.max_rows:\n                            raise RuntimeError(\"Scalar projection row bound reached\")\n                        metrics = self.decode(raw)\n                        signature = self.storage._metric_row_signature(metrics, scalar_only=True)\n                        if signature and signature not in staged_groups:\n                            if signature not in self.groups:\n                                if len(self.groups) + new_signatures >= self.max_groups:\n                                    raise RuntimeError(\"Scalar projection signature bound reached\")\n                                new_signatures += 1\n                                previous = None\n                            else:\n                                previous = self.groups[signature][-1] if self.groups[signature] else None\n                            staged_groups[signature] = []\n                            staged_max[signature] = previous\n                        key = (timestamp, row_id)\n                        all_max = self._stage_key(staged_all, key, all_max)\n                        if signature:\n                            staged_max[signature] = self._stage_key(\n                                staged_groups[signature], key, staged_max[signature])\n                        self.stats[\"source_rows_decoded\"] += 1\n                    self.watermark = row_id\n            # One merge for all_keys and at most one per touched signature,\n            # after EVERY original SQL256 chunk in this frontier has completed.\n            merged_all = self._merge_keys(self.all_keys, staged_all)\n            merged_groups = dict(self.groups)\n            for signature, keys in staged_groups.items():\n                merged_groups[signature] = self._merge_keys(\n                    self.groups.get(signature, []), keys)\n            self.all_keys = merged_all\n            self.groups = merged_groups\n            self.guard()\n        finally:\n            # fetch() retains its original reset-on-error. Remove all local\n            # staging refs on success/error without clearing transferred aliases.\n            staged_all = staged_groups = staged_max = all_max = None\n            merged_all = merged_groups = rows = row = metrics = signature = key = None\n            keys = previous = raw = None\n"
MATCHING_GATE="    # BOTH original shapes must pass for THIS exact new candidate. An old\n    # reader3 first-shape receipt is historical and cannot authorize reader4.\n    required_shape=scope[\"required_original_shape\"]\n    assert required_shape in (\"metadata_live_order\",\"adverse_reverse_interleaved\")\n    projection_pin=pin(PROJECTION)\n    assert scope[\"projection_source\"]==projection_pin\n    assert projection_pin in scope[\"input_pins\"]\n    prior_matching_pin=scope[\"prior_matching_candidate_first_shape_ROOT\"]\n    if required_shape==\"metadata_live_order\":\n        assert prior_matching_pin is None\n    else:\n        assert type(prior_matching_pin)is dict and prior_matching_pin in scope[\"input_pins\"]\n        assert pin(prior_matching_pin[\"path\"])==prior_matching_pin\n        matching=read(prior_matching_pin[\"path\"])\n        assert matching[\"kind\"]==\"ROOT_Trackio7_projection2_reader4_first_original_shape_actual_acceptance_v1\"\n        assert matching[\"accepted_first_shape\"]is True and matching[\"reviewer\"]==\"/root\"\n        assert matching[\"candidate_reader\"]==scope[\"candidate\"]\n        assert matching[\"projection_source\"]==projection_pin\n        assert matching[\"completed_first_shape\"][\"shape\"]==\"metadata_live_order\"\n        assert matching[\"completed_first_shape\"][\"original_history_rows\"]==500000\n"

scopepath=D/'ROOT_projection2_sorted_batch_merge_reader4_APIproof7_INACTIVE_SOURCE_scope_v1.json'
rawscope=scopepath.read_bytes()
assert len(rawscope)==7796 and sha(rawscope)=='4219ba331adf98f4d9573ece7b7eae584bfb0415efd131d2f704a8802e61d035'
scope=json.loads(rawscope)
oldp=pathlib.Path(scope['base_projection']['path']);oldr=pathlib.Path(scope['base_reader']['path']);oldf=pathlib.Path(scope['base_proof']['path'])
pb=oldp.read_bytes();rb=oldr.read_bytes();fb=oldf.read_bytes()
assert len(pb)==6935 and sha(pb)=='cb071def6f5c22b2dc98e88f6ca0036a641c8fbe516232ec5356772f0bf6c64c'
assert len(rb)==25922 and sha(rb)=='2bf6dbf3518667cf7f4f943c87c6f966101c6f55cab1d5679ced25c3f401db9c'
assert len(fb)==44519 and sha(fb)=='4fbd16bdcd3a714f9707fda5556652d029270bf8aa7cfc0d3328906cc4c29b85'
def rep(text,a,b,count=1):
    assert text.count(a)==count,(a[:100],text.count(a),count)
    return text.replace(a,b)
projector=pb.decode()
projector=rep(projector,'from bisect import insort\n','')
start=projector.index('    def _append_key(self, keys, key):\n')
end=projector.index('    def fetch(self, cursor, run_identity, max_points, *, scalar_only=False):\n',start)
projector=projector[:start]+PROJECTOR_BLOCK+'\n'+projector[end:]
projection_path=D/'incremental_scalar_dashboard_projection_v2.py'
projection_pin=write_source(projection_path,projector.encode())
reader=rb.decode()
reader=rep(reader,'    API_BOOKKEEPING_RESERVE = 16_777_216 # Additional; never freed training credit.\n','    API_BOOKKEEPING_RESERVE = 16_777_216 # Additional; never freed training credit.\n    SORT_MERGE_RESERVE = 67_108_864 # Additional temporary metadata above cache/API/original floors.\n')
reader=rep(reader,'QUERY_CACHE_RESERVE + API_BOOKKEEPING_RESERVE','QUERY_CACHE_RESERVE + API_BOOKKEEPING_RESERVE + SORT_MERGE_RESERVE',2)
reader=rep(reader,'    helper = Path(r"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/incremental_scalar_dashboard_projection_v1.py")','    helper = Path(r"'+projection_path.as_posix()+'")')
reader=rep(reader,'    assert len(helper_raw) == 6935','    assert len(helper_raw) == '+str(projection_pin['bytes']))
reader=rep(reader,'cb071def6f5c22b2dc98e88f6ca0036a641c8fbe516232ec5356772f0bf6c64c',projection_pin['sha256'])
reader=rep(reader,'"asr18_scalar_projection_v1"','"asr18_scalar_projection_v2"')
reader=rep(reader,'"cache_reserve_bytes_above_original_floors": QUERY_CACHE_RESERVE','"cache_reserve_bytes_above_original_floors": QUERY_CACHE_RESERVE,\n                              "sort_merge_reserve_bytes_above_original_cache_and_API": SORT_MERGE_RESERVE')
reader_path=D/'serve_BPE9223_Trackio_live_v7_INACTIVE_v4.py'
reader_pin=write_source(reader_path,reader.encode())
proof=fb.decode()
proof=rep(proof,'CANDIDATE=HERE/"serve_BPE9223_Trackio_live_v7_INACTIVE_v3.py"','CANDIDATE=HERE/"serve_BPE9223_Trackio_live_v7_INACTIVE_v4.py"\nPROJECTION=HERE/"incremental_scalar_dashboard_projection_v2.py"')
proof=rep(proof,'    assert limits["prospective_commit"]>=7452887424','    assert limits["prospective_commit"]>=8056867200')
proof=rep(proof,'    assert limits["prospective_physical"]>=4702887424','    assert limits["prospective_physical"]>=5306867200\n    assert limits["runtime_commit"]>=5538431744 and limits["runtime_physical"]>=5038431744\n    assert limits["family_private"]==limits["family_private_bytes"]==2573741824')
start=proof.index('    # Compose the CLOSED accepted original first shape with this still-unrun\n')
end=proof.index('    start=time.monotonic();children=[];identities=[];known_family={};records=[]\n',start)
proof=proof[:start]+MATCHING_GATE+proof[end:]
proof=rep(proof,'        for shape in ("adverse_reverse_interleaved",):','        for shape in (required_shape,):')
proof=rep(proof,'accepted_original_first_shape_ROOT=accepted_first_shape_pin,this_run_only_remaining_original_adverse_shape=True,completed_first_shape_rerun=False','prior_matching_candidate_first_shape_ROOT=prior_matching_pin,required_original_shape=required_shape,candidate_reader=scope["candidate"],projection_source=projection_pin,this_run_one_explicit_original_shape=True,historical_reader3_first_shape_NOT_activation_authority=True')
assert 'ad1079d8e9247f824f9f27c9b51a0f5afce826fb1e013cfdff335252ac788368'not in proof
proof_path=D/'Trackio7_new_projection_real_API_shape_CPU_proof_v7.py'
proof_pin=write_source(proof_path,proof.encode())
def top(raw):
    return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def class_methods(raw):
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)and n.name=='IncrementalScalarProjection')
    return {n.name:ast.dump(n,include_attributes=False)for n in cls.body if isinstance(n,ast.FunctionDef)}
old_methods=class_methods(pb.decode());new_methods=class_methods(projector)
assert set(new_methods)-set(old_methods)=={'_stage_key','_merge_keys'}
assert set(old_methods)-set(new_methods)=={'_append_key'}
for name in ('__init__','_reset','_source_is_target','fetch','invalidate'):
    assert old_methods[name]==new_methods[name]
assert top(pb.decode())['install']==top(projector)['install']
old_rt=top(rb.decode());new_rt=top(reader)
assert set(old_rt)==set(new_rt)and[k for k in old_rt if old_rt[k]!=new_rt[k]]==['main']
old_ft=top(fb.decode());new_ft=top(proof)
assert set(old_ft)==set(new_ft)and[k for k in old_ft if old_ft[k]!=new_ft[k]]==['fixture_body']
for name in ('server_child','source_bound_lifetime_negatives','request','controller'):assert old_ft[name]==new_ft[name]
# The exact original SQL256 frontier/chunk query and selected ID400 code remain.
for literal in ['ORDER BY id LIMIT 256','range(0, len(ids), 400)','zip(groups, budgets, strict=True)']:
    assert literal in projector
assert '.clear()'not in PROJECTOR_BLOCK
p_labels=['Remove per-row bisect.insort; bounded whole-frontier staging of new all/group key references with same SQL256/max1m/max256 bounds',
'Preserve original out_of_order_inserts meaning using per-list ingestion maximum; target decode counter still per new target row',
'Stable sort only NEW keys; transfer cold list, append ordered warm keys, or stable two-way merge each historical all/touched-group list once AFTER full fixed upper-ID scan',
'Drop local staging/merge refs in finally without clear() on aliases; original fetch reset-on-error/replacement/shrink/target/fallback/sampler/selected400/RLock semantics exact',
'Additional guard checks at merge boundaries and every16384 pointer moves; original SQL256 guards preserved, no bound reduction']
r_labels=['Only new projector2 path/length/SHA/module name dependency binding plus proposed64MiB SORT_MERGE_RESERVE above original cache256/API16 and both original2GB/1.5GB headroom floors',
'Private1.5GB, APIhelper3/officialdispatcher/mirror/logging/STATE/pending/readiness/lifetime/fixture-isolation semantics unchanged',
'Expose additional reserve as scalar status field; no running reader/source modification']
f_labels=['Exact newreader4/projector2 bindings and explicitly charged64MiB stricter prospective/runtime thresholds; family/private/HTTP/wall limits unchanged',
'One ROOT-required original shape per fresh process; metadata_live_order requires prior_matching=None; adverse requires future acceptedfirstROOT with EXACT newreader4/projector2 pins and500k metadata shape',
'Oldreader3/ad107 historical proof no longer activation authority; both fresh newcandidate shapes still required',
'Original shape generator, first cold8+bulk64+version/reference/SQLite timeout/idempotence192/STATE+loss fences/warm/reader0closure/port/alllifetime/caller/Gradio tests exact']
ledger=dict(kind='projection2_reader4_APIproof7_whole_ordered_RAW_AST_v1',projection=raw_diff(pb,projector.encode(),oldp,projection_path,p_labels),reader=raw_diff(rb,reader.encode(),oldr,reader_path,r_labels),proof=raw_diff(fb,proof.encode(),oldf,proof_path,f_labels),AST_checks=dict(projection_added_methods=['_stage_key','_merge_keys'],projection_removed_methods=['_append_key'],projection_changed_method=['_refresh'],projection_unchanged_methods=['__init__','_reset','_source_is_target','fetch','invalidate'],install_unchanged=True,reader_top_delta=['main'],proof_top_delta=['fixture_body'],proof_server_fences_lifetime_caller_unchanged=True),source_scope=pin_bytes(scopepath,rawscope))
raw_pin=write_json(D/'projection2_reader4_APIproof7_full_ordered_RAW_AST_v1.json',ledger)
# Conservative 64-bit pointer bookkeeping estimate, not a timestamp-size/fit proof.
n=1000000;g=256;pointer=8;list_capacity=n+n//8+32
staging_refs=2*list_capacity*pointer
merged_refs=2*list_capacity*pointer
sort_temp=(n//2+1)*pointer
dict_heads=g*3*512+g*2*128
other_bounded=2*1024**2
estimate=staging_refs+merged_refs+sort_temp+dict_heads+other_bounded
assert estimate<67108864
resources=dict(sort_merge_extra_reserve=67108864,original_query_cache_reserve=268435456,original_API_reserve=16777216,
API_prospective_commit=7989758336+67108864,API_prospective_physical=5239758336+67108864,
API_runtime_commit=5471322880+67108864,API_runtime_physical=4971322880+67108864,
reader_original_floor_commit=2000000000,reader_original_floor_physical=1500000000,
reader_new_floor_commit=2000000000+268435456+16777216+67108864,reader_new_floor_physical=1500000000+268435456+16777216+67108864,
family_private_UNCHANGED=2573741824,reader_private_UNCHANGED=1500000000,controller_private_UNCHANGED=536870912,watchdog_private_UNCHANGED=536870912,
HTTP_UNCHANGED=20,body_wall_UNCHANGED=160,external_wall_UNCHANGED=180,
pointer_engineering_arithmetic=dict(pointer_bytes=pointer,row_cap=n,group_cap=g,overallocated_list_capacity_allowance=list_capacity,new_all_and_group_stage_refs=staging_refs,all_and_group_merge_output_refs=merged_refs,max_one_sort_workspace=sort_temp,dict_and_list_heads=dict_heads,other_bounded_allowance=other_bounded,total_extra_estimate=estimate,remaining_to64MiB=67108864-estimate),
limitations=['Engineering reserve proposal; no hard OS allocation or fit guarantee','Existing retained key/timestamp/id objects remain charged to original256MiB cache proposal; max1m rows counts OLD+NEW, not extra1m','Malformed/large timestamp payloads not given a new capacity guarantee; original private/headroom guards remain fatal','No stopped native/reader freed-credit accounting','Both exact future actual shape scopes require independent ROOT acceptance; no benchmark or performance claim yet'])
resource_pin=write_json(D/'projection2_reader4_APIproof7_resource_proposal_v1.json',resources)
prior=[]
for identity in [dict(pid=20556,ctime=1792864246.5160425,ppid=20060),dict(pid=20060,ctime=1792864246.5063748,ppid=20272)]:
    try:q=psutil.Process(identity['pid']);same=q.create_time()==identity['ctime']and q.ppid()==identity['ppid']
    except psutil.NoSuchProcess:same=False
    prior.append(dict(identity=identity,exact_dead=not same))
assert all(x['exact_dead']for x in prior)
handoff=dict(kind='projection2_reader4_APIproof7_INACTIVE_SOURCE_handoff_v1',
projection=projection_pin,reader=reader_pin,proof=proof_pin,RAW_AST=raw_pin,resource_proposal=resource_pin,
source_scope=pin_bytes(scopepath,rawscope),prior_failed_adverse_ROOT=scope['ROOT_API04_failure'],accepted_CPU03_ROOT=scope['ROOT_CPU03_pass'],
reader3_firstshape='HISTORICAL ONLY / cannot authorize reader4',matching_future_first_ROOT='UNKNOWN until newcandidate metadata shape passes',
proposed_sequential_scopes=[
dict(required_original_shape='metadata_live_order',prior_matching_candidate_first_shape_ROOT=None,outputs=dict(body=str(D/'a7_f05'),external=str(D/'a7_e05'))),
dict(required_original_shape='adverse_reverse_interleaved',prior_matching_candidate_first_shape_ROOT='UNKNOWN / REQUIRED matching new reader4+projector2 after new first acceptance',outputs=dict(body=str(D/'a7_f06'),external=str(D/'a7_e06')))],
required_future_first_ROOT_schema=dict(kind='ROOT_Trackio7_projection2_reader4_first_original_shape_actual_acceptance_v1',accepted_first_shape=True,reviewer='/root',candidate_reader=reader_pin,projection_source=projection_pin,completed_first_shape=dict(shape='metadata_live_order',original_history_rows=500000)),
proposed_prefix=[str(P/'work/monitor-env/Scripts/python.exe'),'-X','utf8','-B',str(proof_path),'--ROOT-review','UNKNOWN'],
watchdog1_unchanged=dict(path=str(D/'owned_source_CPU_proof_watchdog_v1.py'),bytes=8988,sha256='04d188dd774c4a9bed0ce896505a1b8563adbc1b5135c207da26d6266daad16a'),
APIhelper3_unchanged=dict(path=str(D/'Trackio_owned_bounded_sync_API_dispatch_v3.py'),bytes=6038,sha256='9f478b8a732d63d710d8163eb9ae179e3fc57a1d8ade40ef3af130c7a0311198'),
actual_execution_admitted=False,candidate_import_execution=False,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_completion_exact_deaths=prior)
handoff_pin=write_json(D/'projection2_reader4_APIproof7_INACTIVE_SOURCE_handoff_v1.json',handoff)
final=sample()
assert final['commit']>=2612368384 and final['physical']>=2112368384 and final['private']<=536870912 and final['E_free']>=100000000000 and time.monotonic()-t0<=180
result=write_json(D/'projection2_reader4_APIproof7_SOURCE_result_v1.json',dict(projection=projection_pin,reader=reader_pin,proof=proof_pin,RAW_AST=raw_pin,resource_proposal=resource_pin,handoff=handoff_pin,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),prior_completion_exact_deaths=prior,final=final,wall=time.monotonic()-t0,candidate_import_execution=False))
print(json.dumps(dict(projection=projection_pin,reader=reader_pin,proof=proof_pin,RAW_AST=raw_pin,resources=resource_pin,handoff=handoff_pin,result=result,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),extra_pointer_estimate=estimate)))

