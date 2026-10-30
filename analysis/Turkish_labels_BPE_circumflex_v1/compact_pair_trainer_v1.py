"""Exact append-only pair fitting with bounded compact postings and no heap.

All counters are exact occurrence-weighted integers. Candidate selection is the
same frequency-descending/canonical-JSON-pair-ascending order as SQLite v5.
"""
import array, collections, json, sys, unicodedata

def canonical(value):
 return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

class CompactTrainer:
 def __init__(self,candidate,guard,pair_limit,posting_limit,word_limit,cell_limit):
  self.vocab=candidate['model']['vocab'];self.merges=candidate['model']['merges']
  self.pieces=['']*len(self.vocab)
  for p,i in self.vocab.items():self.pieces[i]=p
  assert len(self.pieces)+512<65536
  self.left=[];self.right=[];self.length=[]
  for p in self.pieces:self.properties(p)
  self.g=guard;self.pair_limit=pair_limit;self.posting_limit=posting_limit
  self.word_limit=word_limit;self.cell_limit=cell_limit
  self.sequences=[];self.weights=array.array('Q');self.counts={};self.posts={};self.combos={}
  self.live_postings=0;self.ever_postings=0;self.initial_cells=0;self.word_merges=0
  self.selected_scan_cells=0;self.maximum_pair_keys=0;self.maximum_live_postings=0

 def properties(self,p):
  letters=p[1:] if p.startswith('▁') else p
  good='▁' not in letters and all(c.isalpha() and 'LATIN' in unicodedata.name(c,'') for c in letters)
  self.left.append(good);self.right.append(good and not p.startswith('▁'));self.length.append(len(letters))

 def pair_counts(self,seq):
  answer=collections.Counter()
  for a,b in zip(seq,seq[1:]):
   if self.left[a] and self.right[b] and 1<self.length[a]+self.length[b]<=24:
    key=(a<<16)|b
    piece=self.combos.get(key)
    if piece is None:piece=self.pieces[a]+self.pieces[b]
    if piece not in self.vocab:answer[key]+=1
  return answer

 def add_posting(self,key,wid):
  if key not in self.posts:
   if len(self.posts)>=self.pair_limit:raise RuntimeError('Finite compact pair-key admission exceeded')
   self.posts[key]=array.array('I')
   self.combos[key]=self.pieces[key>>16]+self.pieces[key&65535]
  if self.ever_postings>=self.posting_limit:raise RuntimeError('Finite compact cumulative-posting admission exceeded')
  assert 0<=wid<2**32
  self.posts[key].append(wid);self.live_postings+=1;self.ever_postings+=1
  self.maximum_pair_keys=max(self.maximum_pair_keys,len(self.posts))
  self.maximum_live_postings=max(self.maximum_live_postings,self.live_postings)

 def retire(self,key):
  self.counts.pop(key,None);arr=self.posts.pop(key,None)
  if arr is not None:self.live_postings-=len(arr)
  self.combos.pop(key,None)

 def add_word(self,wid,seq,frequency):
  assert wid==len(self.sequences) and wid<self.word_limit
  assert 0<frequency<2**64 and seq and max(seq)<len(self.pieces)
  assert self.initial_cells+len(seq)<=self.cell_limit
  if wid%4096==0:self.g.check()
  self.sequences.append(array.array('H',seq));self.weights.append(frequency);self.initial_cells+=len(seq)
  for key,n in self.pair_counts(seq).items():
   total=self.counts.get(key,0)+n*frequency;assert 0<total<2**64
   self.counts[key]=total;self.add_posting(key,wid)

 def select(self):
  while self.counts:
   maximum=max(self.counts.values());best=None;tie=None
   self.selected_scan_cells+=len(self.counts)
   for key,value in self.counts.items():
    if value==maximum:
     raw=canonical([self.pieces[key>>16],self.pieces[key&65535]])
     if tie is None or raw<tie:best=key;tie=raw
   if self.combos[best] not in self.vocab:return best,maximum
   self.retire(best)
  raise RuntimeError('Insufficient observed eligible corpus pairs')

 def step(self):
  self.g.check();key,frequency=self.select();left=key>>16;right=key&65535
  piece=self.combos[key];newid=len(self.pieces);assert newid<65536 and piece not in self.vocab
  posting=self.posts[key];affected=0;consumed=0;replacements=0
  for wid in posting:
   seq=self.sequences[wid];old=self.pair_counts(seq)
   if key not in old:continue
   consumed+=old[key]*self.weights[wid]
   merged=[];at=0
   while at<len(seq):
    if at+1<len(seq) and seq[at]==left and seq[at+1]==right:
     merged.append(newid);at+=2;replacements+=1
    else:merged.append(seq[at]);at+=1
   # The new token properties are installed before computing new edges, while
   # vocab insertion stays after every word, matching v5's exact step policy.
   if len(self.pieces)==newid:self.pieces.append(piece);self.properties(piece)
   new=self.pair_counts(merged)
   for pair in old.keys()|new.keys():
    value=self.counts.get(pair,0)+(new.get(pair,0)-old.get(pair,0))*self.weights[wid]
    assert 0<=value<2**64,(pair,value)
    if value==0:self.retire(pair)
    else:
     self.counts[pair]=value
     if pair not in old:self.add_posting(pair,wid)
   self.sequences[wid]=array.array('H',merged);affected+=1
   if affected%4096==0:self.g.check()
  assert consumed==frequency,('Selected frequency/postings mismatch',consumed,frequency)
  assert key not in self.counts
  self.word_merges+=replacements
  assert self.ever_postings<=self.initial_cells-len(self.sequences)+2*self.word_merges
  assert self.word_merges<self.initial_cells
  assert len(self.pieces)==newid+1
  result={'piece':piece,'id':newid,'left':self.pieces[left],'right':self.pieces[right],
   'left_id':left,'right_id':right,'merge_rank':len(self.merges),
   'weighted_pair_frequency':frequency,'affected_word_types':affected}
  self.vocab[piece]=newid;self.merges.append([self.pieces[left],self.pieces[right]])
  self.g.check();return result

 def metrics(self):
  return {'word_types':len(self.sequences),'initial_sequence_cells':self.initial_cells,
   'live_pair_keys':len(self.counts),'maximum_pair_keys':self.maximum_pair_keys,
   'live_posting_cells':self.live_postings,'maximum_live_posting_cells':self.maximum_live_postings,
   'ever_posting_cells':self.ever_postings,'token_replacements':self.word_merges,
   'pair_count_selection_scans':self.selected_scan_cells,'heap_entries':0}

def oracle_fixture():
 class NoGuard:
  def check(self):pass
 # Equal frequencies and repeated/overlapping pairs test exact JSON tie order,
 # greedy nonoverlap replacements, stale postings and existing-piece exclusion.
 vocab={'▁':0,'a':1,'b':2,'c':3,'ç':4,'i':5,'ı':6,'▁a':7,'ab':8}
 words=[(['▁','a','a','a','a'],5),(['▁','a','b','a','b'],6),(['▁a','c','c'],4),
        (['▁','ç','i','ı','a','a'],5),(['▁','b','a','c'],6),(['a','a','a'],2)]
 c={'model':{'vocab':dict(vocab),'merges':[]}};t=CompactTrainer(c,NoGuard(),1000,1000,100,1000)
 for wid,(seq,freq) in enumerate(words):t.add_word(wid,[vocab[x] for x in seq],freq)
 brute=[(list(seq),freq) for seq,freq in words];bv=dict(vocab);out=[]
 for step in range(10):
  pairs=collections.Counter()
  def ok(pair):
   p=''.join(pair);s=p[1:] if p.startswith('▁') else p
   return 1<len(s)<=24 and '▁' not in s and p not in bv and s.isalpha() and all('LATIN' in unicodedata.name(x,'') for x in s)
  for seq,freq in brute:
   for pair in zip(seq,seq[1:]):
    if ok(pair):pairs[pair]+=freq
  if not pairs:break
  pair=min(pairs,key=lambda p:(-pairs[p],canonical(p)));piece=''.join(pair);affected=0
  for j,(seq,freq) in enumerate(brute):
   new=[];at=0
   while at<len(seq):
    if at+1<len(seq) and tuple(seq[at:at+2])==pair:new.append(piece);at+=2
    else:new.append(seq[at]);at+=1
   affected+=new!=seq;brute[j]=(new,freq)
  expected={'piece':piece,'id':len(bv),'left':pair[0],'right':pair[1],
   'left_id':bv[pair[0]],'right_id':bv[pair[1]],'merge_rank':step,
   'weighted_pair_frequency':pairs[pair],'affected_word_types':affected}
  actual=t.step();assert actual==expected,(actual,expected)
  bv[piece]=len(bv);out.append(actual)
 assert len(out)>=8
 return {'passed':True,'independent_bruteforce_exact_merges':len(out),'metrics':t.metrics()}
