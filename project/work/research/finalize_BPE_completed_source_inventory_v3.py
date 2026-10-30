"""Finite metadata-only final inventory; admit only after actual acquisition/header exits."""
import hashlib
import json
from pathlib import Path
import time

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT, AREA
from cpu_preparation_guard_v2 import now
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/parakeet_turkish"))
from atomic_io_retry_v1 import write_json


class MetadataGuard(DataIOGuard):
    def pace(self, count):
        super().pace(count)
        end = time.monotonic() + count / 262144
        while time.monotonic() < end:
            self.wait(); time.sleep(min(.025, max(0, end-time.monotonic())))


def main():
    guard = MetadataGuard('completed_source_inventory_metadata', heavy=False, maximum_bytes=64_000_000)
    guard.wait()
    target = AREA/'completed_source_inventory_v1.json'
    assert not target.exists(), 'Preserve the first completed source inventory'

    def binding(path):
        path = Path(path)
        assert path.is_absolute() and path.is_relative_to(ROOT)
        size = path.stat().st_size
        assert size < 8_000_000, 'Only bounded metadata receipts may be loaded'
        guard.pace(size)
        raw = path.read_bytes()
        assert len(raw) == size, 'Receipt changed while binding'
        return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

    plan_path = AREA/'preparation_plan_v1.json'
    plan_binding = binding(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    closed = {}
    for role, proof_name, marker_name in (
        ('acquisition','acquisition_owned_tool_actual_CLOSED0_v5.json','acquisition_actual_CLOSED0_v5.json'),
        ('headers','YODAS_header_owned_tool_actual_CLOSED0_v3.json','YODAS_archive_indexes_actual_CLOSED0_v3.json')):
        proof_path = AREA/proof_name
        proof = json.loads(proof_path.read_text(encoding='utf-8-sig'))
        assert proof['kind'] == 'ASR18_source_bound_actual_tool_exit_v1'
        assert proof['owner'] == '/root/cpu_data_preparation'
        assert proof['actual_exit_code'] == 0 and proof['actual_tool_chunk_id']
        marker = binding(AREA/marker_name)
        assert proof['published_marker'] == marker, 'Source-bound actual exit must bind exact final worker marker'
        raw_marker = json.loads((AREA/marker_name).read_text(encoding='utf-8-sig'))
        assert raw_marker['status'] == 'completed' and raw_marker['actual_exit_code'] == 0
        if role == 'acquisition': assert raw_marker['completed_files'] == 773
        else: assert raw_marker['completed_archives'] == len(raw_marker['indexes']) == 51
        closed[role] = {'owned_actual_exit':binding(proof_path),'published_marker':marker}

    jobs = [('new_plan',job) for job in plan['download_jobs']] + [('reused_archive',job) for job in plan['reused_archives']]
    assert len(jobs) == 784 and len({job['destination'] for _,job in jobs}) == 784
    records = []
    for classification, job in jobs:
        guard.wait()
        destination = Path(job['destination'])
        assert destination.is_absolute() and destination.is_relative_to(ROOT)
        assert destination.stat().st_size == job['size']
        receipt_path = destination.with_suffix(destination.suffix+'.receipt.json')
        receipt_binding = binding(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        assert receipt['sha256'] == job['sha256'] and receipt['revision'] == job['revision']
        assert receipt.get('bytes',receipt.get('size')) == job['size']
        records.append({'repository':job['repository'],'revision':job['revision'],
                        'path':job['path'],'destination':str(destination),
                        'source_size_bytes':job['size'],'source_SHA256':job['sha256'],
                        'classification':classification,'source_acquisition_receipt':receipt_binding})
    report = {'schema_version':1,'kind':'ASR18_completed_source_inventory_v1','created_at_utc':now(),
              'owner':'/root/cpu_data_preparation','PID':guard.process.pid,
              'preparation_plan':plan_binding,'finalization_contract':binding(AREA/'production_finalization_contract_v4.json'),
              'new_sources_count':773,'reused_sources_count':11,'sources_count':len(records),
              'total_source_bytes':sum(row['source_size_bytes'] for row in records),
              'actual_acquisition_and_header_closures':closed,'sources':records,
              'source_receipt_hashes_and_current_sizes_verified':True,
              'full_source_bytes_rehashed_by_inventory_builder':False,
              'download_time_full_SHA256_verification_preserved':True,
              'metadata_catalog_and_BPE_finalization_separate':True,
              'runtime_heldout_and_geometry_admission_separate':True,
              'TRAIN_admitted':False,'actual_paced_metadata_bytes':guard.bytes,
              'private_bytes':guard.process.memory_info().private}
    assert report['total_source_bytes'] == 718_862_913_959
    write_json(target,report)
    print(json.dumps({'PID':guard.process.pid,'inventory_path':str(target),
                      'inventory_SHA256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'sources_count':len(records),'total_source_bytes':report['total_source_bytes']}),flush=True)


if __name__=='__main__': main()
