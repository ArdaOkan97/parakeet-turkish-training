"""Bounded light observer of new committed labels, retaining all rows unchanged."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from prepare_BPE_partial_geometry_envelope_v1 import MetadataGuard,reader
from ASR18_continuous_data_guard_v4 import ROOT,AREA
from cpu_preparation_guard_v2 import now,write_json


def main():
    guard=MetadataGuard('incoming_label_geometry',heavy=False,maximum_bytes=128_000_000)
    guard.wait();base_path=AREA/'partial_geometry_envelope_v1/report.json'
    assert hashlib.sha256(base_path.read_bytes()).hexdigest()=='3986ce2c542de0ccf7cc539a4c9c1629ca89f3b452bb5f11c140427f14c4a1c1'
    base=json.loads(base_path.read_text());g=dict(base['groups']['all']);frontier=base['pinned_label_frontier']
    labels=reader(ROOT/'indexes/BPE_broad_labels_8711_v1.sqlite');started=time.monotonic();first=[]
    status=AREA/'incoming_label_geometry_status_v1.json'
    def publish(phase):
        write_json(status,{'kind':'ASR18_incoming_committed_label_geometry_v1','at_utc':now(),
            'PID':guard.process.pid,'status':phase,'baseline_frontier':base['pinned_label_frontier'],
            'observed_committed_frontier':frontier,'groups_all_committed':g,'first_U_blocker_ordinals':first,
            'readiness_blockers_not_filters':g['count_U_gt_458']>0 or g['count_U_ge_1024']>0,
            'whole_Alania_baseline_preserved':True,'KIRAAT_and_full_corpus_still_partial':True,
            'rows_excluded':0,'TRAIN_admitted':False,'actual_paced_metadata_bytes':guard.bytes})
    publish('starting_from_pinned_complete_Alania_partial_KIRAAT_baseline')
    while time.monotonic()-started<86400:
        guard.wait()
        target=labels.execute('SELECT frontier FROM progress WHERE id=1').fetchone()[0]
        if target>frontier:
            end=min(target,frontier+8192);guard.pace((end-frontier)*32)
            rows=labels.execute('SELECT ordinal,target_tokens,conservative_estimated_joint_bytes FROM labels WHERE ordinal>? AND ordinal<=? ORDER BY ordinal',(frontier,end)).fetchall()
            assert len(rows)==end-frontier
            for ordinal,U,dense in rows:
                g['rows']+=1;g['sum_U']+=U;g['MAX_U']=max(g['MAX_U'],U)
                g['count_U_gt_458']+=U>458;g['count_U_ge_1024']+=U>=1024
                g['count_U_plus_1_gt_1024']+=U+1>1024
                g['MAX_estimated_T_times_U_plus_1']=max(g['MAX_estimated_T_times_U_plus_1'],dense//(8716*4))
                if U>458 and len(first)<8:first.append({'ordinal':ordinal,'U':U})
            frontier=end;publish('new_commits_checked')
        else:
            publish('caught_up_to_latest_committed_frontier')
            actual=AREA/'BPE_labels_owned_tool_actual_CLOSED0_v3.json'
            if actual.exists() and json.loads(actual.read_text(encoding='utf-8-sig'))['actual_exit_code']==0:
                publish('completed_committed_label_observation_actual_writer_closed');break
            time.sleep(2)
    else:publish('bounded_24hour_observation_finished')
    labels.close()


if __name__=='__main__':main()
