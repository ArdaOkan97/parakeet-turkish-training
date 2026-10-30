"""Pinned local torch archive projection: retain tensors/RNG; omit bulky history.

The immutable parent remains the complete historical record. Protocol-2 history
containers and their memo entries are consumed without materialization. Missing
memo references outside history fail closed. No clock, moment or RNG synthesis.
"""
import pickle
import types


class _Discard:
    def __setitem__(self, key, value): pass
    def append(self, value): pass
    def extend(self, values): pass


DISCARD = _Discard()


class HistoryProjectionUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = None
        self.skipping = False
        self.skipped_memos = 0
        self.history_seen = False
        self.history_stack_index = None
        self.preserved_history_key_memos = 0

    def keep_memo(self,index):
        value=self.stack[-1]
        if not self.skipping: self.memo[index]=value
        elif (isinstance(value,str) and len(value)<=80 and value.isascii()
              and value.replace('_','').isalpha()):
            # Metadata key strings can first occur in history and be reused later.
            self.preserved_history_key_memos+=1
            if self.preserved_history_key_memos>50000: raise ValueError('Bounded projected string memo limit exceeded')
            self.memo[index]=value
        else: self.skipped_memos+=1

    def top_key(self, value):
        if len(self.metastack) == 1 and self.metastack[0] and self.metastack[0][0] is self.root:
            if not self.skipping and isinstance(value,str) and value == 'history':
                self.skipping=True;self.history_seen=True;self.history_stack_index=len(self.stack)-1
            elif (self.skipping and len(self.stack)==self.history_stack_index+3
                  and self.stack[self.history_stack_index+1] is DISCARD):
                # Only the next sibling of the completed root history is a key.
                if not isinstance(value,str): raise ValueError('Unknown root-key reference after projected history')
                self.skipping=False

    def load_binunicode(self):
        super().load_binunicode(); self.top_key(self.stack[-1])
    dispatch[pickle.BINUNICODE[0]] = load_binunicode

    def load_empty_dictionary(self):
        self.append(DISCARD if self.skipping else {})
        if self.root is None: self.root = self.stack[-1]
    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary

    def load_empty_list(self): self.append(DISCARD if self.skipping else [])
    dispatch[pickle.EMPTY_LIST[0]] = load_empty_list

    def load_binput(self):
        index = self.read(1)[0]
        self.keep_memo(index)
    dispatch[pickle.BINPUT[0]] = load_binput

    def load_long_binput(self):
        index = int.from_bytes(self.read(4), 'little')
        self.keep_memo(index)
    dispatch[pickle.LONG_BINPUT[0]] = load_long_binput

    def get_memo(self, index):
        if index not in self.memo:
            if not self.skipping: raise ValueError('Skipped history memo referenced outside history: ' + str(index))
            value = DISCARD
        else: value = self.memo[index]
        self.append(value); self.top_key(value)

    def load_binget(self): self.get_memo(self.read(1)[0])
    dispatch[pickle.BINGET[0]] = load_binget
    def load_long_binget(self): self.get_memo(int.from_bytes(self.read(4), 'little'))
    dispatch[pickle.LONG_BINGET[0]] = load_long_binget

    def load_reduce(self):
        if self.skipping:
            self.stack.pop(); self.stack[-1] = DISCARD
        else: super().load_reduce()
    dispatch[pickle.REDUCE[0]] = load_reduce

    def load_binpersid(self):
        if self.skipping: self.stack.pop(); self.append(DISCARD)
        else: super().load_binpersid()
    dispatch[pickle.BINPERSID[0]] = load_binpersid

    def find_class(self, module, name):
        permitted = ((module == 'collections' and name == 'OrderedDict') or
                     (module == 'torch' and name.endswith('Storage')) or
                     (module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor', '_rebuild_parameter')))
        if not permitted: raise ValueError('Unsupported pinned-checkpoint global: ' + module + '.' + name)
        return super().find_class(module, name)


PICKLE = types.ModuleType('asr18_history_projection_pickle')
PICKLE.Unpickler = HistoryProjectionUnpickler
PICKLE.load = pickle.load


def load_without_history(torch, path):
    """Caller must verify the immutable local checkpoint's pinned SHA first."""
    value = torch.load(path, map_location='cpu', mmap=True, weights_only=False, pickle_module=PICKLE)
    if not isinstance(value, dict) or value.get('history') is not DISCARD:
        raise ValueError('Expected top-level projected history was not consumed')
    value.pop('history')
    def check(item):
        if item is DISCARD: raise ValueError('Discarded history reference escaped projection')
        if isinstance(item, dict):
            for k,v in item.items(): check(k); check(v)
        elif isinstance(item, (tuple,list)):
            for child in item: check(child)
    check(value)
    return value
