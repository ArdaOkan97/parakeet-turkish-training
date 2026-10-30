"""One CPU class per official base; no per-call tensor in a class closure.

Only the live official wrapper instance borrows denominator/workspace tensors.
The caller releases those attributes after the synchronous official cost call.
This does not change normalization, DP, kernels or gradient arithmetic.
"""
_CLASSES = {}


def _copy_denominator(self, unused_acts, unused_denom):
    self._ASR18_workspace[:self._ASR18_cells].copy_(self._ASR18_denominator)


def exact_denominator_type(official_base):
    if official_base not in _CLASSES:
        if _copy_denominator.__closure__ is not None:
            raise ValueError('Denominator method must not close over call tensors')
        _CLASSES[official_base] = type('ASR18ExactFullDenominatorInstance',
            (official_base,), {'log_softmax': _copy_denominator, '__module__': __name__})
    return _CLASSES[official_base]


def bind(wrapper, workspace, denominator, cells):
    if hasattr(wrapper, '_ASR18_workspace'):
        raise ValueError('Live wrapper denominator already bound')
    if type(cells) is not int or cells <= 0 or denominator.numel() != cells:
        raise ValueError('Exact full denominator geometry required')
    wrapper._ASR18_workspace = workspace
    wrapper._ASR18_denominator = denominator
    wrapper._ASR18_cells = cells


def release(wrapper):
    for name in ('_ASR18_workspace', '_ASR18_denominator', '_ASR18_cells'):
        delattr(wrapper, name)
