import ast,difflib,hashlib,json
from pathlib import Path
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
OLD=F.parent/'resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1/verify_owned_packet_credit_CPU_v1.py'
S=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish')
source=OLD.read_bytes().decode('utf-8'); old_newline='CRLF' if '\r\n' in source else 'LF'; text=source.replace('\r\n','\n'); edits=[]
def replace(old,new,count=1):
 global text
 actual=text.count(old)
 assert actual==count,(old[:100],actual,count)
 text=text.replace(old,new)
 edits.append({'old_literal':old,'new_literal':new,'expected_count':count})
replace('Parent harness substitutes constructor/worker only. Actual CPU10 sender, receiver,\nreadiness/get/next_update/release/close/abort and helper3 drain are exercised.',
'''Parent harness substitutes constructor/worker and adds assertion-only observers.
Actual CPU11 sender, receiver, readiness/get/next_update/release/close/abort and
helper3 drain are exercised. Weakrefs observe arrays, base ndarrays and memoryviews;
raw bytearray owners cannot be weak-referenced and are identified honestly.
Two admitted complete packets is NOT a full frontend/process/kernel-pipe bound.''')
replace('import argparse,copy,hashlib,json,os,queue,random,shutil,subprocess,sys,threading,time,traceback,weakref','import argparse,ast,copy,hashlib,json,os,queue,random,shutil,subprocess,sys,threading,time,traceback,weakref')
replace('fastmss_resident_feature_stream_v10','fastmss_resident_feature_stream_v11')
replace('asr18_owned_packet_credit_v1','asr18_owned_packet_credit_v2',4)
replace("BEGIN=time.monotonic();TRACKED={};PEAK=0", "BEGIN=time.monotonic();TRACKED={};PEAK=0\nFEATURE_SHAPE=(1,2048,128);MASK_SHAPE=(1,2048)\nSYNTHETIC_ARRAY_BYTES=2048*128*4+2048")
needle='\ndef worker(a):'
extra='''
def payload_weakrefs(values):
    """Weakref actual ndarray/base/memoryview objects, never retain raw payloads."""
    refs=[];seen=set();types={};unweakrefable=set()
    for root in values:
        current=root
        while current is not None and id(current) not in seen:
            seen.add(id(current));name=type(current).__name__;types[name]=types.get(name,0)+1
            try:refs.append(weakref.ref(current))
            except TypeError:unweakrefable.add(name)
            if isinstance(current,memoryview):current=current.obj
            else:current=getattr(current,'base',None)
    return refs,{'weakref_object_types':types,'unweakrefable_owner_types':sorted(unweakrefable)}

class SendOutputObserver:
    """Observe actual _send chunk views without copying or retaining payloads."""
    def __init__(self,stream):self.stream=stream;self.refs=[];self.view_writes=0
    def write(self,value):
        if isinstance(value,memoryview):
            refs,_=payload_weakrefs([value]);self.refs.extend(refs);self.view_writes+=1
        return self.stream.write(value)
    def flush(self):return self.stream.flush()

def full_packet_cap_source_control():
    tree=ast.parse(Path(transport.__file__).read_text(encoding='utf-8-sig'))
    worker_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker_main')
    calls=[n for n in ast.walk(worker_node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)
        and n.func.id=='require'and len(n.args)>1 and isinstance(n.args[1],ast.Constant)
        and n.args[1].value=='Bounded full update packet with all features/masks']
    require(len(calls)==1,'One original post-prepare full packet cap expression')
    expr=compile(ast.Expression(calls[0].args[0]),str(transport.__file__),'eval')
    def accepted(total):return eval(expr,{'len':len,'arrays':[None,None],'metadata':{'rows':[{}]},
        'total':total,'PACKET_LIMIT':transport.PACKET_LIMIT})
    require(accepted(transport.PACKET_LIMIT)and not accepted(transport.PACKET_LIMIT+1),
        'Original full256MiB geometry guard tested WITHOUT allocating256MiB')
    return {'actual_original_guard_AST':ast.dump(calls[0],include_attributes=False),
        'limit_bytes':transport.PACKET_LIMIT,'at_limit_accepted':True,'over_limit_rejected':True,
        'large_allocation_performed':False,'post_prepare_guard_not_a_frontend_growth_bound':True}

'''
replace(needle,'\n'+extra+needle)
replace("features=np.arange(8,dtype=np.float32).reshape(1,4,2)+index\n                mask=np.ones((1,4),dtype=np.bool_)\n                arrays=[features,mask];refs=[weakref.ref(x)for x in arrays]",
"features=np.arange(2048*128,dtype=np.float32).reshape(FEATURE_SHAPE)+index\n                mask=np.ones(MASK_SHAPE,dtype=np.bool_)\n                arrays=[features,mask];refs,owner_types=payload_weakrefs(arrays)")
replace("                event(folder,{'kind':'synthetic_prepare_started','plan':sha,'index':index})", "                event(folder,{'kind':'synthetic_prepare_started','plan':sha,'index':index,'monotonic':time.monotonic()})")
replace("                if a.fault=='send_eof':transport._send(out,frame);return 0", """                if a.fault=='receiver_full_cap':
                    frame['array_bytes']=transport.PACKET_LIMIT+1
                    frame['array_descriptors'][0]['bytes']=transport.PACKET_LIMIT+1-frame['array_descriptors'][1]['bytes']
                    transport._send(out,frame);return 0
                if a.fault=='send_eof':transport._send(out,frame);return 0""")
replace("                transport._send(out,frame,arrays)\n                del features,mask,arrays,metadata,frame,specs\n                require(all(r()is None for r in refs),'Synthetic producer sent arrays no longer referenced before next acquire')\n                event(folder,{'kind':'synthetic_sent_payload_refs_dead','plan':sha,'index':index})", """                observed_out=SendOutputObserver(out);outer_views=[]
                def send_profile(actual_frame,kind,arg):
                    if kind=='return'and actual_frame.f_code is transport._send.__code__:
                        view=actual_frame.f_locals.get('view')
                        if view is not None:outer_views.append(weakref.ref(view))
                sys.setprofile(send_profile)
                try:transport._send(observed_out,frame,arrays)
                finally:sys.setprofile(None)
                refs.extend(observed_out.refs);refs.extend(outer_views)
                view_writes=observed_out.view_writes
                del features,mask,arrays,metadata,frame,specs,observed_out
                require(view_writes>=2 and outer_views and all(r()is None for r in refs),
                    'Actual send views and synthetic producer arrays/base owners dead BEFORE next real acquire')
                event(folder,{'kind':'synthetic_sent_payload_refs_dead','plan':sha,'index':index,
                    'monotonic':time.monotonic(),'actual_send_view_writes':view_writes,
                    'actual_send_outer_view_weakrefs':len(outer_views),'all_payload_weakrefs_dead':True,
                    'ownership_types':owner_types})""")
replace('    """Only constructor is synthetic; runtime transport methods are production CPU10."""','    """Synthetic constructor plus assertion-only observers; actual CPU11 methods run."""')
replace("        self.credit=PacketCreditOwner();self.required_dead_refs=[]", """        self.credit=PacketCreditOwner();self.required_dead_refs=[]
        self.receiver_trace=[];self.receiver_raw_active=0;self.receiver_raw_peak=0;self.receiver_view_refs=[]
        self.delivered_ownership=[]""")
replace('self._receiver=threading.Thread(target=self._receive,daemon=True);self._sender.start();self._receiver.start()', 'self._receiver=threading.Thread(target=self._observed_receive,daemon=True);self._sender.start();self._receiver.start()')
replace('    def closure(self,normal=None):', '''    def _receive_profile(self,frame,kind,arg):
        if frame.f_code is not transport._read_exact.__code__:return
        caller=frame.f_back
        if caller is None or caller.f_code is not transport.ResidentFeatureProducer._receive.__code__:return
        state=caller.f_locals;header=state.get('frame',{});spec=state.get('spec',{})
        if header.get('kind')!='update':return
        if kind=='call':
            count=frame.f_locals['count']
            require(count==spec['bytes']and self.used_bytes==header['array_bytes']
                and not self.live and not self.packets,
                'Actual receiver FULL byte reservation precedes first partial allocation; LIVE1 including partial')
            self.receiver_raw_active+=1;self.receiver_raw_peak=max(self.receiver_raw_peak,self.receiver_raw_active)
            require(self.receiver_raw_active==1,'No second receiver raw allocation/read overlaps')
            self.receiver_trace.append({'kind':'before_actual_raw_allocation','plan':header['plan_sha256'],
                'index':header['update_index'],'token':header['token'],'read_bytes':count,
                'full_packet_reserved_bytes':self.used_bytes,'live_packets_before_allocation':len(self.live),
                'monotonic':time.monotonic()})
        elif kind=='return':
            view=frame.f_locals.get('view')
            if view is not None:self.receiver_view_refs.append(weakref.ref(view))
            self.receiver_raw_active-=1
            require(self.receiver_raw_active==0,'Actual raw read unwound before subsequent allocation')
    def _observed_receive(self):
        sys.setprofile(self._receive_profile)
        try:return transport.ResidentFeatureProducer._receive(self)
        finally:sys.setprofile(None)
    def get(self,timeout=300):
        packet=transport.ResidentFeatureProducer.get(self,timeout)
        self.required_dead_refs,types=payload_weakrefs([x for row in packet['arrays']for x in row.values()])
        self.delivered_ownership.append({'token':packet['token'],**types})
        return packet
    def closure(self,normal=None):''')
replace("        return {'actual_Popen_returncode':self.process.returncode,'launcher':self.launcher,'producer':who,", """        trace_path=self.folder/'receiver_lifetime_trace.json';write(trace_path,self.receiver_trace)
        return {'actual_Popen_returncode':self.process.returncode,'launcher':self.launcher,'producer':who,
            'receiver_lifetime_trace':pin(trace_path),'receiver_raw_read_peak':self.receiver_raw_peak,
            'receiver_raw_read_active_at_closure':self.receiver_raw_active,
            'actual_read_exact_view_weakrefs_dead':all(r()is None for r in self.receiver_view_refs),
            'delivered_array_and_base_ownership':self.delivered_ownership,
            'raw_bytearray_not_weakrefable':True,""")
replace("        and review['CPU_source']==pin(transport.__file__),'Separate exact ROOT finite CPU fixture scope required')", """        and review['CPU_source']==pin(transport.__file__),'Separate exact ROOT finite CPU fixture scope required')
    for input_pin in review['input_pins']:require(pin(input_pin['path'])==input_pin,'Exact frozen CPU scope input')
    candidate=transport.bound(review['proposed_configuration'])
    require(candidate['resident_RAM_packet_credit_policy']['actual_CPU_control_review']is None,
        'Inactive production candidate has NO fake completed proof')
    require(candidate['resident_RAM_packet_credit_policy']['human_approval']==review['human_approval']
        and transport.bound(review['human_approval'])['answer']=='Approve lookahead and checkpoint restart',
        'Exact direct human logical-lookahead exception')
    transport.checked_resource_profile(candidate)
    require(candidate['resident_RAM_packet_credit_policy']['maximum_logical_packets']==2,
        'Max2 policy selected only by exact explicit approval')""")
replace('        # Genuine max1/overflow/API/collision; no allocation stress or GPU.', '        # Genuine max2/overflow/API/collision and actual blocked third acquisition; no stress/GPU.')
replace("                consumer.acquire(phase='consume_initial',token='synthetic',check=guard)\n                ticks=iter([0.,601.])", """                consumer.acquire(phase='consume_A',token='synthetic_A',check=guard)
                consumer.acquire(phase='consume_B',token='synthetic_B',check=guard)
                started=threading.Event();finished=threading.Event();third=[]
                def real_third_acquire():
                    started.set()
                    try:third.append(consumer.acquire(phase='REAL_BLOCKED_C',token='synthetic_C',check=guard))
                    except BaseException as error:third.append({'error':repr(error)})
                    finally:finished.set()
                waiter=threading.Thread(target=real_third_acquire);waiter.start();started.wait(2);time.sleep(.1)
                require(not finished.is_set(),'Actual third logical acquire blocked while two permits outstanding')
                owner.return_credit();require(owner.last_release_previous_count==0,'Actual release previous0')
                waiter.join(10);require(not waiter.is_alive()and finished.is_set()and third and 'error'not in third[0],
                    'Third acquire wakes only after real post; waiter joined before handle closure')
                owner.return_credit();owner.return_credit()
                require(owner.last_release_previous_count==1,'Actual final release previous1 restores two permits')
                results['genuine_two_credit_third_wait_and_previous01']={'third':third,'owner':owner.receipt(),'waiter_joined':True}
                for field,value in [('logical_packet_maximum',1),('kind','wrong_kind'),('nonce','0'*32),
                    ('helper_source',{'wrong':True}),('owner',{'pid':owner.descriptor['owner']['pid'],
                    'create_time':owner.descriptor['owner']['create_time']+1})]:
                    altered=copy.deepcopy(owner.descriptor);altered[field]=value
                    results['descriptor_reject_'+field]=expect(ValueError,lambda:PacketCreditConsumer(altered))
                consumer.acquire(phase='consume_A_again',token='synthetic_A',check=guard)
                consumer.acquire(phase='consume_B_again',token='synthetic_B',check=guard)
                ticks=iter([0.,601.])""")
replace("        plan1,pin1=synthetic_plan(out,'plan1',0)", "        results['original_full_cap_AST_no_large_allocation']=full_packet_cap_source_control()\n        plan1,pin1=synthetic_plan(out,'plan1',0)")
replace("        time.sleep(.15);guard();require(len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==1,'prepare2 blocked while packet1 queued')", """        until(lambda:len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==2)
        time.sleep(.1);guard()
        require(len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==2,
            'A queued plus B preparation allowed; C logical acquire blocked')
        require(all(e['index']==0 and e['plan']==transport.digest(plan1)for e in h.receiver_trace),
            'No B raw receiver allocation while A remains queued')""")
replace("        p=h.next_update(transport.digest(plan1),0);token=p['token'];refs=[weakref.ref(x)for r in p['arrays']for x in r.values()]\n        h.required_dead_refs=refs", """        p=h.next_update(transport.digest(plan1),0);token=p['token']
        require(any(x.get('memoryview',0)>0 for x in [h.delivered_ownership[-1]['weakref_object_types']]),
            'Actual ndarray base memoryview ownership observed')
        before_posts=h.credit.posts
        results['observer_rejects_premature_payload_release']=expect(ValueError,lambda:h.release(token))
        require(h.credit.posts==before_posts and h.live[token]['delivered'],
            'Assertion-only fixture guard rejects early release WITHOUT kernel post or token mutation')""")
replace("        time.sleep(.15);guard();require(len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==1,'prepare2 blocked delivered-unreleased')", """        time.sleep(.1);guard();require(len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==2,
            'C still blocked with A delivered-unreleased and B allowed')
        require(all(e['index']==0 for e in h.receiver_trace),'No B raw allocation while A delivered-unreleased')""")
replace("            h.required_dead_refs=[weakref.ref(x)for r in p['arrays']for x in r.values()]\n            del p;h.release(token)", "            del p;h.release(token)")
replace("        results['ordered_two_plans_and_periodic_drain']=h.closure(closed);owned.remove(h)", """        events=read_events(h.folder)
        prepared=[e for e in events if e['kind']=='synthetic_prepare_started']
        sent=[e for e in events if e['kind']=='synthetic_sent_payload_refs_dead']
        require(len(prepared)==len(sent)==40,'All40 ordered synthetic packets sent once')
        expected=[(transport.digest(plan1),i)for i in range(20)]+[(transport.digest(plan2),i)for i in range(20)]
        require([(e['plan'],e['index'])for e in prepared]==expected,'Real serial producer current-to-queued plan order')
        require(all(sent[i]['monotonic']<=prepared[i+1]['monotonic']for i in range(39)),
            'Every actual sent array/base/view dead BEFORE following actual acquire succeeds')
        require(h.credit.last_release_previous_count==1 and h.credit.posts==40,
            'Final last-current/queued-plan drain release previous1 restores initial two permits exactly')
        results['ordered_two_plans_and_periodic_drain']=h.closure(closed);owned.remove(h)""")
replace("        for fault in ['prepare_error','send_error','send_eof','receiver_hash_error']:", "        for fault in ['prepare_error','send_error','send_eof','receiver_hash_error','receiver_full_cap']:")
replace("            h.abort();results[fault]={'pre_abort_actual_error':actual,'closure':h.closure()};owned.remove(h)", """            if fault=='receiver_full_cap':require(not h.receiver_trace,
                'Actual receiver rejects over-full-cap header BEFORE payload allocation')
            h.abort();results[fault]={'pre_abort_actual_error':actual,'closure':h.closure()};owned.remove(h)""")
replace("                consumer.acquire(phase='consume_initial',token='owner_death_synthetic',check=guard)", "                consumer.acquire(phase='consume_A',token='owner_death_synthetic_A',check=guard)\n                consumer.acquire(phase='consume_B',token='owner_death_synthetic_B',check=guard)")
replace("                process.stdin.write(b'x');process.stdin.flush();code=process.wait(timeout=10);waiter.join(timeout=10)", "                require(waiter.is_alive()and not observed,'Actual third acquire remains blocked before owner exits')\n                process.stdin.write(b'x');process.stdin.flush();code=process.wait(timeout=10);waiter.join(timeout=10)")
replace("        write(out/'report.json',{'kind':'ASR18_actual_owned_packet_credit_transport_CPU_control_v1','passed':True,", """        for input_pin in review['input_pins']:require(pin(input_pin['path'])==input_pin,'Frozen input unchanged after actual control')
        write(out/'report.json',{'kind':'ASR18_actual_owned_two_logical_packet_one_receiver_transport_CPU_control_v1','passed':True,
            'maximum_logical_packets':2,'maximum_live_receiver_packets':1,
            'payload_lifetime_and_partial_receive_interleavings_passed':True,
            'constructor_and_small_synthetic_preparation_are_explicit_doubles':True,
            'completed_production_CPU_review_remained_None':True,
            'two_full_copy_claim_conditional_on_original_packet_geometry_admission':True,
            'frontend_growth_allocator_workspace_kernel_pipe_are_extra_not_512MiB_bound':True,
            'synthetic_array_geometry':{'features':FEATURE_SHAPE,'mask':MASK_SHAPE,'bytes':SYNTHETIC_ARRAY_BYTES},
            'fake600s_clock_case_not_real600s_wait':True,'real_audio_frontend_or_GPU_fit_not_proved':True,
            'human_approval':review['human_approval'],'input_pins':review['input_pins'],""")
replace("'original_CPU9_numerics_not_executed_here':True", "'original_CPU11_numerics_not_executed_here':True")
# No execution/import of fixture. Freeze CRLF source and exact source-level diff/ledger.
path=F/'verify_two_logical_packet_one_receiver_CPU_v1.py';assert not path.exists();ast.parse(text)
path.write_bytes(text.replace('\n','\r\n').encode('utf-8'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,obj):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(obj,f,indent=2);f.write('\n')
ledger={'kind':'ASR18_synthetic_fixture_source_edit_ledger_v1','old_source':pin(OLD),'new_source':pin(path),
 'old_newline':old_newline,'new_newline':'CRLF','literal_edits_applied_to_LF_normalized_text':edits,
 'actual_control_executed':False,'synthetic_constructor_array_observer_changes_are_not_production_changes':True}
write(F/'two_logical_packet_fixture_source_ledger_v1.json',ledger)
(F/'verify_two_logical_packet_one_receiver_CPU_v1.diff').write_text(''.join(difflib.unified_diff(
 source.replace('\r\n','\n').splitlines(True),text.splitlines(True),fromfile=str(OLD),tofile=str(path))),encoding='utf-8')
schema={'kind':'ASR18_requested_completed_two_logical_packet_transport_ROOT_gate_schema_v1',
 'accepted':True,'reviewer':'/root','scope':'ASR18_owned_two_logical_packet_one_receiver_transport_v1',
 'helper_source':pin(F/'asr18_owned_packet_credit_v2.py'),'CPU_source':pin(S/'fastmss_resident_feature_stream_v11.py'),
 'human_approval':pin(F.parent/'resident_RAM_source27_speed_review_v1/ROOT_direct_human_lookahead_and_checkpoint_restart_approval_v1.json'),
 'maximum_logical_packets':2,'maximum_live_receiver_packets':1,
 'genuine_named_kernel_credit_and_pipe_flow_passed':True,'synthetic_packets_no_audio_model':True,
 'payload_lifetime_and_partial_receive_interleavings_passed':True,
 'NOTE':'Requested schema only. This file is NOT a completed review or production admission.'}
write(F/'requested_completed_two_logical_packet_ROOT_proof_schema_v1.json',schema)
inputs=[path,F/'asr18_owned_packet_credit_v2.py',S/'fastmss_resident_feature_stream_v11.py',
 S/'asr18_resident_periodic_CPU_quiescence_v3.py',F/'proposed_configuration_v1.json',
 F/'helper2_CPU11_exact_raw_inverse_ledger_v1.json',F/'ROOT_independent_helper2_CPU11_inverse_review_v1.json',
 Path(schema['human_approval']['path']),F/'requested_completed_two_logical_packet_ROOT_proof_schema_v1.json',
 Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1')]
handoff={'kind':'ASR18_INACTIVE_two_logical_packet_one_receiver_finite_CPU_fixture_handoff_v1',
 'actual_CPU_execution_authorized':False,'production_or_stop_authorized':False,
 'fixture_source':pin(path),'helper_source':schema['helper_source'],'CPU_source':schema['CPU_source'],
 'proposed_configuration':pin(F/'proposed_configuration_v1.json'),'human_approval':schema['human_approval'],
 'source_ledger':pin(F/'two_logical_packet_fixture_source_ledger_v1.json'),
 'fixture_diff':pin(F/'verify_two_logical_packet_one_receiver_CPU_v1.diff'),
 'requested_completed_proof_schema':pin(F/'requested_completed_two_logical_packet_ROOT_proof_schema_v1.json'),
 'input_pins':[pin(x)for x in inputs],
 'proposed_arguments':['--ROOT-review','<ROOT exact source-scope receipt>','--output',str(F/'finite_transport_actual_CPU_01')],
 'limits':{'internal_seconds':180,'external_seconds':240,'sampled_family_private_bytes':536870912,
 'prospective_family_reservation_bytes':536870912,'full_additional_prefetch_bytes':268435456,
 'original_commit_floor_bytes':2000000000,'original_physical_floor_bytes':1500000000,'E_free_minimum_bytes':100000000000,
 'CUDA_VISIBLE_DEVICES':'-1','all_four_math_thread_caps':1,'priority':'BelowNormal'},
 'honest_scope':['real named semaphore/cancel/owner process handles and actual CPU11 pipe methods',
 'synthetic constructor and bounded1MiB features replace model/audio frontend',
 'actual array/base/memoryview weakrefs, rawbytearray itself is not weakrefable',
 'profile callbacks observe actual _send return view and _read_exact preallocation/return only',
 'original full256MiB cap AST tested at/over cap without fullcap allocation; real overcap receiver negative',
 'two logical permits and receiverLIVE1 INCLUDING partial/delivered, no third acquire and previous0/1 releases',
 'same-plan helper3 full20 drain and last-current to queued-plan transition',
 'production fit/speed/real audio/numericalbody and future production DEV phase NOT proved']}
write(F/'inactive_finite_two_logical_packet_CPU_fixture_handoff_v1.json',handoff)
print(json.dumps({'fixture':pin(path),'handoff':pin(F/'inactive_finite_two_logical_packet_CPU_fixture_handoff_v1.json'),
 'schema':pin(F/'requested_completed_two_logical_packet_ROOT_proof_schema_v1.json'),'ast_parse_only':True,'actual_control_executed':False}))
