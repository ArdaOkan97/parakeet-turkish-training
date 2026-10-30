"""Bounded Counter/sorted-run TRAIN word counts; no random frequency DB writes."""
import collections,heapq,json,os,time
from pathlib import Path

class WordCounts:
    def __init__(self,guard,db):
        self.guard=guard;self.db=db;self.counter=collections.Counter();self.runs=[]
        self.area=Path(db.execute('PRAGMA database_list').fetchone()[2]).parent/'word_count_runs'
        self.area.mkdir();self.total_occurrences=0
    def add(self,batch):
        self.counter.update(batch);self.total_occurrences+=sum(batch.values());self.guard.check()
        if len(self.counter)>=1500000 or self.guard.process.memory_info().private>=600000000:self.spill()
    def spill(self):
        if not self.counter:return
        if len(self.runs)>=64:raise RuntimeError('Finite maximum64 sorted word-count runs')
        path=self.area/('word_counts_%03d.jsonl'%len(self.runs));assert not path.exists()
        with path.open('xb',buffering=1024*1024) as f:
            for seq in sorted(self.counter):
                line=(json.dumps([seq,self.counter[seq]],ensure_ascii=False,separators=(',',':'))+'\n').encode()
                self.guard.pace(len(line));f.write(line)
            f.flush();os.fsync(f.fileno())
        self.runs.append(path);self.counter.clear()
        self.guard.publish(sorted_word_count_runs=len(self.runs),staged_word_occurrences=self.total_occurrences)
    def rows(self,path):
        previous=None
        with path.open('rb',buffering=1024*1024) as f:
            for line in f:
                self.guard.pace(len(line));seq,count=json.loads(line)
                assert previous is None or previous<seq
                assert isinstance(count,int) and count>0
                previous=seq;yield seq,count
    def finish(self):
        self.spill();assert self.runs
        self.guard.phase='sequential_sorted_word_count_merge'
        merged=heapq.merge(*(self.rows(p) for p in self.runs),key=lambda pair:pair[0])
        current=None;count=0;batch=[];total=0;types=0
        for seq,frequency in merged:
            if current is not None and seq!=current:
                batch.append((current,count));types+=1;total+=count;count=0
                if len(batch)>=8192:
                    self.db.executemany('INSERT INTO words(seq,freq) VALUES (?,?)',batch)
                    self.db.commit();batch.clear();self.guard.check()
            current=seq;count+=frequency
        if current is not None:batch.append((current,count));types+=1;total+=count
        if batch:self.db.executemany('INSERT INTO words(seq,freq) VALUES (?,?)',batch);self.db.commit()
        assert total==self.total_occurrences
        self.guard.publish(complete_sorted_word_count_runs=len(self.runs),weighted_word_types=types,weighted_word_occurrences=total)
