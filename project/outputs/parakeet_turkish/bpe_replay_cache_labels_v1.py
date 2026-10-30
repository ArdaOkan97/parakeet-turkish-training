"""Read full original replay label storage without a tensor runtime.

The whole genuine cache binding is checked, but only its small label storage
is decoded. Frozen teacher feature storage remains for the native caller.
"""
import json,struct,zipfile
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_checkpoint_scalar_metadata_v1 import ScalarMetadataUnpickler,StorageDescriptor,TensorDescriptor

STORAGES=('ByteStorage','FloatStorage','HalfStorage','BFloat16Storage','LongStorage','DoubleStorage')

class ReplayMetadataUnpickler(ScalarMetadataUnpickler):
    def find_class(self,module,name):
        if module=='torch' and name in STORAGES:return name
        return super().find_class(module,name)

    def persistent_load(self,value):
        require(type(value) is tuple and len(value)==5 and value[0]=='storage' and value[1] in STORAGES
            and type(value[2]) is str and value[2].isdigit() and value[3]=='cpu' and type(value[4]) is int
            and 0<=value[4]<=100_000_000,'Only bounded inert original replay CPU storage metadata')
        self.storages+=1;require(self.storages<=32,'Bounded one replay-cache storage metadata')
        return StorageDescriptor(value[1],value[2],value[4])

def full_replay_labels(row,cache_binding,*,guard=lambda:None):
    path=Path(row['cache_path']).resolve();guard()
    require(path==Path(cache_binding['path']).resolve() and type(cache_binding['bytes']) is int
        and 0<cache_binding['bytes']<=128*1024**2 and record(path)==cache_binding,'Exact bounded original replay cache required')
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    with zipfile.ZipFile(path) as archive:
        entries=[i for i in archive.infolist() if i.filename.endswith('/data.pkl')]
        require(len(entries)==1 and entries[0].file_size<=1024**2,'Bounded original replay metadata')
        prefix=entries[0].filename[:-len('data.pkl')]
        require(archive.read(prefix+'byteorder')==b'little','Explicit original little-endian label storage required')
        with archive.open(entries[0]) as stream:
            reader=ReplayMetadataUnpickler(stream);saved=reader.load();require(stream.read(1)==b'','Single complete replay metadata tree')
        require(type(saved) is dict and 'kind' not in saved and saved['source_id']==row['source_id'],'Genuine original replay cache/source')
        labels=saved['labels'];features=saved['features'];u=row['target_tokens'];t=row['encoder_frames']
        require(type(u) is int and u>0 and type(t) is int and t>0 and type(labels) is TensorDescriptor
            and labels.storage.dtype=='LongStorage' and labels.shape==(u,) and labels.stride==(1,)
            and labels.offset+u<=labels.storage.elements,'Full original contiguous old-vocabulary target storage required')
        require(type(features) is TensorDescriptor and len(features.shape)==2 and features.shape[0]==t
            and features.shape[1]==1024 and features.storage.dtype in ('FloatStorage','HalfStorage','BFloat16Storage'),
            'Original full frozen encoder feature metadata must agree')
        storage=archive.getinfo(prefix+'data/'+labels.storage.key)
        require(storage.file_size==labels.storage.elements*8<=8*1024**2,'Bounded full label storage; never read a prefix as full labels')
        raw=archive.read(storage);start=labels.offset*8;ids=list(struct.unpack('<'+'q'*u,raw[start:start+8*u]))
        require(len(ids)==u and all(type(v) is int and 0<=v<8193 and v!=8192 for v in ids),'Every unchanged original target ID required')
    guard();require(before==(path.stat().st_size,path.stat().st_mtime_ns),'Original replay cache changed during label read')
    return ids,{'cache':cache_binding,'cache_signature':{'bytes':before[0],'mtime_ns':before[1]},
        'source':record(__file__),'full_label_count':u,'teacher_feature_shape':list(features.shape),
        'teacher_feature_storage_decoded':False,'label_storage_bytes_decoded':len(raw),
        'whole_cache_opaque_SHA256_reverified':True,'actual_audio_or_native_update_proved':False}
