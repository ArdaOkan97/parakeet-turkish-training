"""Only the two explicitly admitted existing metadata archives, scalar extraction."""
import gzip,hashlib,json,math,os,shutil,statistics,sys,time
from pathlib import Path
import psutil
S=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish');sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/checkpoints')
BEGIN=time.monotonic();PEAK=0;OUT=F/'two_existing200_cleanup_boundaries_actual_v1.json'
INPUTS=((158800,5684827,'ebc114bfc90c852fb9e166e192b1dc3384ff325bc8db780bc4cb4153ccd1de71'),
        (162000,5682503,'7a65a88bf7d81f3f63d2831436eba45adac5f44db1b2cca7af4258448a624f44'))
def require(v,m):
    if not v:raise ValueError(m)
def pin(p):
    b=Path(p).read_bytes();return {'path':str(Path(p).resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def guard(admission=False):
    global PEAK
    x=sample_memory(os.getpid(),'two_existing200_cleanup_scalar_archive_read');m=x['system_memory'];PEAK=max(PEAK,x['worker_private_memory']['private_bytes'])
    extra=536870912 if admission else 0
    require(x['paired_sampling_within_one_second']and PEAK<536870912 and time.monotonic()-BEGIN<180
        and m['commit_available_bytes']>=2268435456+extra and m['physical_available_bytes']>=1768435456+extra
        and shutil.disk_usage('E:/').free>=100000000000,'Original CPU512MiB/180s/floors+fullprefetch/prospective/E100GB')
    return x
def summary(values):
    ordered=sorted(values);n=len(ordered)
    return {'count':n,'sum':math.fsum(values),'mean':statistics.mean(values),'median':statistics.median(values),
        'minimum':ordered[0],'maximum':ordered[-1],'p90_nearest_rank':ordered[math.ceil(.90*n)-1],
        'p95_nearest_rank':ordered[math.ceil(.95*n)-1]}
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
    and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Separate CPU-only E environment')
require(not OUT.exists(),'One original2archive extraction')
initial=guard(True);report={'kind':'ASR18_two_existing_original200_cleanup_boundary_scalar_extraction_v1','passed':False,
    'identity':who(me),'launcher':who(me.parent()),'source':pin(__file__),'initial_resource_sample':initial,
    'limits':['Boundary seconds include original GC, CUDA synchronization/emptycache, snapshots, RNG/pointer/invariant checks; NOT pureGC time.',
        'Full original200 updates from each of ONLY the2 admitted metadata archives, no cherry-picking.',
        'No tensor/audio/journal or training operation; no GC setting or admitted process change.'],
    'archives':{}}
for update,size,expected in INPUTS:
    path=B/('update_'+str(update).zfill(9))/'prepared_metadata.json.gz';before=path.stat()
    require(before.st_size==size<=8*1024**2,'Exact admitted compressed archive bound')
    compressed=pin(path);require(compressed['sha256']==expected,'Exact admitted compressed archive')
    chunks=[];length=0;sha=hashlib.sha256()
    with gzip.open(path,'rb')as stream:
        while True:
            guard();chunk=stream.read(min(1024**2,32*1024**2+1-length))
            if not chunk:break
            length+=len(chunk);require(length<=32*1024**2,'Hard32MiB uncompressed metadata bound')
            sha.update(chunk);chunks.append(chunk)
    blob=b''.join(chunks);del chunks;values=json.loads(blob);del blob;guard()
    require(type(values)is list and len(values)==200,'ALL200 original archive update records')
    per_update=[];boundaries=[]
    for index,entry in enumerate(values):
        native=entry['native'];actual=native['update'];require(actual==update-199+index,'Every ordered original update, no missing or repeated record')
        group_records=native['graph_boundary_memory_observations'];require(type(group_records)is list and bool(group_records),'Original recorded cleanup boundaries')
        seconds=[];counts=[];indices=[]
        for gi,v in enumerate(group_records):
            require(v['group']==gi+1 and type(v['collected_objects'])is int and v['collected_objects']>=0
                and isinstance(v['seconds'],(int,float))and math.isfinite(v['seconds'])and v['seconds']>=0,
                'Exact original scalar boundary fields')
            seconds.append(v['seconds']);counts.append(v['collected_objects']);indices+=v['original_row_indices']
            boundaries.append({'update':actual,'group':v['group'],'seconds':v['seconds'],'collected_objects':v['collected_objects'],
                'original_row_indices':v['original_row_indices']})
        require(sorted(indices)==list(range(len(native['source_ids'])))and len(indices)==len(set(indices)),
            'All original native rows covered once by recorded groups')
        per_update.append({'update':actual,'boundary_count':len(group_records),'cleanup_boundary_seconds':math.fsum(seconds),
            'collected_objects':sum(counts),'original_native_row_count':len(native['source_ids'])})
        if index%20==0:guard()
    after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Read-only archive unchanged')
    report['archives'][str(update)]={'compressed':compressed,'uncompressed_bytes':length,'uncompressed_sha256':sha.hexdigest(),
        'gzip_complete_read_and_CRC_verified':True,'exact_full_update_range':[update-199,update],
        'all200_update_scalar_records':per_update,'all_group_scalar_records':boundaries,
        'cleanup_seconds_per_update':summary([v['cleanup_boundary_seconds']for v in per_update]),
        'collected_objects_per_update':summary([v['collected_objects']for v in per_update]),
        'cleanup_seconds_per_group':summary([v['seconds']for v in boundaries]),
        'group_count_distribution':{str(n):sum(v['boundary_count']==n for v in per_update)for n in sorted(set(v['boundary_count']for v in per_update))}}
    del values,per_update,boundaries,group_records,native,entry;guard()
a=report['archives']['158800'];b=report['archives']['162000']
report['comparison']={'late_minus_early_mean_cleanup_boundary_seconds_per_update':b['cleanup_seconds_per_update']['mean']-a['cleanup_seconds_per_update']['mean'],
    'late_over_early_mean_cleanup_boundary_seconds_per_update':b['cleanup_seconds_per_update']['mean']/a['cleanup_seconds_per_update']['mean'],
    'early_mean_collected_objects_per_update':a['collected_objects_per_update']['mean'],
    'late_mean_collected_objects_per_update':b['collected_objects_per_update']['mean']}
report['last_resource_sample']=guard();report['sampled_observer_private_peak_bytes']=PEAK;report['wall_seconds']=time.monotonic()-BEGIN;report['passed']=True
with OUT.open('x',encoding='utf-8')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'report':pin(OUT),'comparison':report['comparison'],'early_seconds':a['cleanup_seconds_per_update'],
    'late_seconds':b['cleanup_seconds_per_update'],'peak':PEAK}),flush=True)
