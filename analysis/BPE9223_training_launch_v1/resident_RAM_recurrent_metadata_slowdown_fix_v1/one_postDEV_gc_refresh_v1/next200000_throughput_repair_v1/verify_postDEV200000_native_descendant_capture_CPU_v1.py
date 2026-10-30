"""NEW narrow actual descendant identity helper proof, not the old21 cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import postDEV200000_stop_watcher_v3 as w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scope = w.read(args.ROOT_review)
    w.require(scope['kind'] == 'ROOT_postDEV200000_current_native_descendant_capture_finite_CPU_exact_ONCE_execution_scope_v1'
              and scope['accepted'] is True and scope['reviewer'] == '/root'
              and scope['fixture_source'] == w.pin(__file__)
              and scope['watcher_source'] == w.pin(w.__file__), 'Separate NEW capture-only CPU scope')
    out = Path(args.output)
    w.require(out.drive.lower() == 'e:' and str(out) == scope['fixture_output'] and not out.exists(), 'Fresh E-only capture proof')
    out.mkdir()
    me = w.identity()
    report = {'kind': 'ASR18_actual_owned_native_descendant_capture_CPU_v1',
              'passed': False, 'cases': [], 'fixture_identity': me,
              'fixture_launcher': w.identity(os.getppid()),
              'source': w.pin(__file__), 'watcher_source': w.pin(w.__file__),
              'old21_science_predicate_proof_not_rerun': True,
              'production_process_operation': False}
    child = None
    done = out / 'owned_done.json'
    began = time.monotonic()
    try:
        before = [w.pin(item['path']) for item in scope['input_pins']]
        w.require(before == scope['input_pins'], 'All exact frozen capture-proof pins')
        ready = out / 'owned_native_ready.json'
        code = ("import os,json,time,sys,psutil;from pathlib import Path; p=psutil.Process();"
                "Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}));"
                "start=time.monotonic();"
                "exec(\"while not Path(sys.argv[2]).exists() and time.monotonic()-start<20:time.sleep(.02)\")")
        child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-c', code, str(ready), str(done)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        launcher = w.identity(child.pid)
        report['owned_launcher'] = launcher
        while not ready.exists():
            w.require(child.poll() is None and time.monotonic() - began < 10, 'Owned new CPU target readiness')
            time.sleep(.02)
        target = w.read(ready)
        report['owned_native'] = target
        w.require(target['pid'] != 19320 and me['pid'] != 19320 and w.exact_alive(target), 'Never production target')
        capture = w.capture_native_descendants(me)
        observed = [item['identity'] for item in capture['descendants']]
        w.require(target in observed and launcher in observed
                  and all(item['ancestry_to_native'][-1] == me for item in capture['descendants'])
                  and capture['count'] <= 8, 'Actual launcher/native chain with exact namespaced ancestry')
        report['cases'].append({'name': 'actual_owned_native_descendant_chain', 'passed': True, 'capture': capture})
        wrong = dict(me, create_time=me['create_time'] + 1)
        try:
            w.capture_native_descendants(wrong)
        except ValueError as exc:
            report['cases'].append({'name': 'wrong_creation_time_rejected', 'passed': True,
                                    'type': type(exc).__name__, 'message': str(exc)})
        else:
            raise AssertionError('Changed native ctime must reject before descendants')
        done.write_bytes(b'done')
        report['actual_Popen_wait_code'] = child.wait(timeout=5)
        w.require(report['actual_Popen_wait_code'] == 0 and not w.exact_alive(target)
                  and not w.exact_alive(launcher), 'Actual owned target+launcher exit, never inferred native from launcher')
        empty = w.capture_native_descendants(me)
        w.require(empty['count'] == 0 and empty['descendants'] == [], 'Actual no-child result is truthful empty')
        report['cases'].append({'name': 'actual_exited_chain_then_empty', 'passed': True, 'capture': empty})
        report['input_pins_before'] = before
        report['input_pins_after'] = [w.pin(item['path']) for item in scope['input_pins']]
        w.require(report['input_pins_after'] == before, 'Capture-proof sources unchanged')
        report['passed'] = True
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        if child is not None and child.poll() is None:
            if not done.exists():
                done.write_bytes(b'owned-proof-only-complete')
            try:
                report['cleanup_Popen_wait'] = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                report['cleanup_Popen_wait'] = 'UNKNOWN'
        report['wall_seconds'] = time.monotonic() - began
        result = w.write_once(out / 'report.json', report)
        print(json.dumps({'report': result, 'passed': report['passed'], 'cases': len(report['cases'])}), flush=True)


if __name__ == '__main__':
    main()
