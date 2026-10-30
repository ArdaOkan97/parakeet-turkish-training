"""Lossless versioned accepted-reference frontier. No audio or model dependencies.

File: MAGIC + little-endian uint32 JSON-schema length + UTF-8 schema, then
D/S dictionary definitions or R row records. Source IDs and dictionary strings
are UTF-8, durations IEEE754 float64, SHA256 raw32 bytes. Rows retain all eight
original fields exactly; per-row encoder/independent decoder checks are used by
finalization. iter_frontier_rows supports bounded streaming reconstruction.
"""
import json
import struct

MAGIC=b'ARFRONT\x01'
ROW=struct.Struct('<IHHd32sIBI')
DEF=struct.Struct('<HI')
SCHEMA={'format':'accepted_reference_exact_frontier_binary_v1','byte_order':'little',
 'fields':['ordinal','source_id','dataset','duration_s','status','PCM_i16_sha256','frames','spectral_usable'],
 'header':'ARFRONT byte1, uint32 schema length, UTF8 JSON schema',
 'dictionary_definition':'D or S byte, uint16 ID, uint32 UTF8 byte length, UTF8 value',
 'row_record':'R byte then <IHHd32sIBI then source_id UTF8 bytes',
 'row_fixed_fields':['ordinal:u32','dataset_id:u16','status_id:u16','duration_s:float64',
                    'PCM_i16_sha256:32rawbytes','frames:u32','spectral_usable:u8','source_id_UTF8_length:u32'],
 'hash_restore':'raw32.hex(), lowercase, verified against original lowercase SHA256',
 'dictionary_IDs':'zero-based within each of D and S; definitions precede use',
 'null_policy':'All eight fields are non-null for actual_WAV_PCM_verified references; unexpected null fails sealing'}

def header_bytes():
    raw=json.dumps(SCHEMA,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return MAGIC+struct.pack('<I',len(raw))+raw

class FrontierEncoder:
    def __init__(self):self.datasets={};self.statuses={}
    def _intern(self,tag,table,value):
        assert isinstance(value,str)
        if value in table:return b'',table[value]
        index=len(table);assert index<65536
        raw=value.encode('utf-8');table[value]=index
        return tag+DEF.pack(index,len(raw))+raw,index
    def encode(self,row):
        ordinal,sid,dataset,duration,status,digest,frames,spectral=row
        assert isinstance(sid,str) and isinstance(duration,float)
        assert 0<=ordinal<2**32 and 0<=frames<2**32 and spectral in (0,1)
        assert isinstance(digest,str) and len(digest)==64 and digest==digest.lower()
        pcm=bytes.fromhex(digest);assert len(pcm)==32
        d,di=self._intern(b'D',self.datasets,dataset)
        s,si=self._intern(b'S',self.statuses,status)
        raw=sid.encode('utf-8')
        return d+s+b'R'+ROW.pack(ordinal,di,si,duration,pcm,frames,spectral,len(raw))+raw

class FrontierDecoder:
    def __init__(self):self.datasets={};self.statuses={}
    def consume(self,data):
        rows=[];offset=0
        while offset<len(data):
            tag=data[offset:offset+1];offset+=1
            if tag in (b'D',b'S'):
                assert offset+DEF.size<=len(data)
                index,length=DEF.unpack_from(data,offset);offset+=DEF.size
                assert offset+length<=len(data)
                text=data[offset:offset+length].decode('utf-8');offset+=length
                table=self.datasets if tag==b'D' else self.statuses
                assert index==len(table) and text not in table.values()
                table[index]=text
            else:
                assert tag==b'R' and offset+ROW.size<=len(data)
                ordinal,di,si,duration,pcm,frames,spectral,length=ROW.unpack_from(data,offset);offset+=ROW.size
                assert offset+length<=len(data) and spectral in (0,1)
                sid=data[offset:offset+length].decode('utf-8');offset+=length
                rows.append((ordinal,sid,self.datasets[di],duration,self.statuses[si],pcm.hex(),frames,spectral))
        assert offset==len(data)
        return rows

def iter_frontier_rows(path,pace=None):
    """Reconstruct tuples with bounded buffering; optional paced byte callback."""
    pending=0
    with open(path,'rb') as f:
        def rd(n):
            nonlocal pending
            raw=f.read(n);assert len(raw)==n,'Truncated compact frontier'
            pending+=n
            if pending>=65536:
                if pace:pace(pending)
                pending=0
            return raw
        assert rd(len(MAGIC))==MAGIC
        length=struct.unpack('<I',rd(4))[0];assert length<65536
        assert json.loads(rd(length))==SCHEMA
        decoder=FrontierDecoder()
        while True:
            tag=f.read(1)
            if not tag:break
            pending+=1
            if tag in (b'D',b'S'):
                fixed=rd(DEF.size);_,n=DEF.unpack(fixed)
            else:
                assert tag==b'R';fixed=rd(ROW.size);n=ROW.unpack(fixed)[-1]
            assert n<1_000_000
            for row in decoder.consume(tag+fixed+rd(n)):yield row
        if pending and pace:pace(pending)
