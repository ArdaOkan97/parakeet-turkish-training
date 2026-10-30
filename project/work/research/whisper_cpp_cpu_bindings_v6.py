"""ctypes C ABI for the pinned official whisper.cpp v1.8.2 header only."""
import ctypes as c
import os
from pathlib import Path


class Aheads(c.Structure):
    _fields_=[('n_heads',c.c_size_t),('heads',c.c_void_p)]


class ContextParams(c.Structure):
    _fields_=[('use_gpu',c.c_bool),('flash_attn',c.c_bool),('gpu_device',c.c_int),
              ('dtw_token_timestamps',c.c_bool),('dtw_aheads_preset',c.c_int),
              ('dtw_n_top',c.c_int),('dtw_aheads',Aheads),('dtw_mem_size',c.c_size_t)]


class Vad(c.Structure):
    _fields_=[('threshold',c.c_float),('min_speech_duration_ms',c.c_int),
              ('min_silence_duration_ms',c.c_int),('max_speech_duration_s',c.c_float),
              ('speech_pad_ms',c.c_int),('samples_overlap',c.c_float)]


class Greedy(c.Structure):
    _fields_=[('best_of',c.c_int)]


class Beam(c.Structure):
    _fields_=[('beam_size',c.c_int),('patience',c.c_float)]


class FullParams(c.Structure):
    _fields_=[('strategy',c.c_int),('n_threads',c.c_int),('n_max_text_ctx',c.c_int),
              ('offset_ms',c.c_int),('duration_ms',c.c_int),
              *[(name,c.c_bool) for name in ('translate','no_context','no_timestamps','single_segment',
                 'print_special','print_progress','print_realtime','print_timestamps','token_timestamps')],
              ('thold_pt',c.c_float),('thold_ptsum',c.c_float),('max_len',c.c_int),
              ('split_on_word',c.c_bool),('max_tokens',c.c_int),('debug_mode',c.c_bool),
              ('audio_ctx',c.c_int),('tdrz_enable',c.c_bool),('suppress_regex',c.c_char_p),
              ('initial_prompt',c.c_char_p),('carry_initial_prompt',c.c_bool),
              ('prompt_tokens',c.c_void_p),('prompt_n_tokens',c.c_int),('language',c.c_char_p),
              ('detect_language',c.c_bool),('suppress_blank',c.c_bool),('suppress_nst',c.c_bool),
              *[(name,c.c_float) for name in ('temperature','max_initial_ts','length_penalty',
                 'temperature_inc','entropy_thold','logprob_thold','no_speech_thold')],
              ('greedy',Greedy),('beam_search',Beam),
              *[(name,c.c_void_p) for name in ('new_segment_callback','new_segment_callback_user_data',
                 'progress_callback','progress_callback_user_data','encoder_begin_callback',
                 'encoder_begin_callback_user_data','abort_callback','abort_callback_user_data',
                 'logits_filter_callback','logits_filter_callback_user_data','grammar_rules')],
              ('n_grammar_rules',c.c_size_t),('i_start_rule',c.c_size_t),('grammar_penalty',c.c_float),
              ('vad',c.c_bool),('vad_model_path',c.c_char_p),('vad_params',Vad)]


READ=c.CFUNCTYPE(c.c_size_t,c.c_void_p,c.c_void_p,c.c_size_t)
EOF=c.CFUNCTYPE(c.c_bool,c.c_void_p)
CLOSE=c.CFUNCTYPE(None,c.c_void_p)
LOG=c.CFUNCTYPE(None,c.c_int,c.c_char_p,c.c_void_p)
ABORT=c.CFUNCTYPE(c.c_bool,c.c_void_p)


class Loader(c.Structure):
    _fields_=[('context',c.c_void_p),('read',READ),('eof',EOF),('close',CLOSE)]


class Runtime:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.dll_directory=os.add_dll_directory(str(self.directory))
        self.dll=c.CDLL(str(self.directory/'whisper.dll'))
        def function(name,result,args):
            fn=getattr(self.dll,name);fn.restype=result;fn.argtypes=args;return fn
        self.version=function('whisper_version',c.c_char_p,[])
        self.context_defaults=function('whisper_context_default_params_by_ref',c.POINTER(ContextParams),[])
        self.full_defaults=function('whisper_full_default_params_by_ref',c.POINTER(FullParams),[c.c_int])
        self.init=function('whisper_init_with_params',c.c_void_p,[c.POINTER(Loader),ContextParams])
        self.full=function('whisper_full',c.c_int,[c.c_void_p,FullParams,c.POINTER(c.c_float),c.c_int])
        self.free=function('whisper_free',None,[c.c_void_p])
        self.log_set=function('whisper_log_set',None,[LOG,c.c_void_p])
        self.segments=function('whisper_full_n_segments',c.c_int,[c.c_void_p])
        self.text=function('whisper_full_get_segment_text',c.c_char_p,[c.c_void_p,c.c_int])
        self.t0=function('whisper_full_get_segment_t0',c.c_int64,[c.c_void_p,c.c_int])
        self.t1=function('whisper_full_get_segment_t1',c.c_int64,[c.c_void_p,c.c_int])
        self.language_id=function('whisper_full_lang_id',c.c_int,[c.c_void_p])
        self.language_name=function('whisper_lang_str',c.c_char_p,[c.c_int])
        self.model={name:function('whisper_model_n_'+name,c.c_int,[c.c_void_p])
                    for name in ('audio_layer','text_layer','audio_state','text_state','mels','vocab')}

    def verify_defaults(self):
        context=self.context_defaults().contents
        full=self.full_defaults(0).contents
        assert self.version().decode()=='1.8.2'
        assert context.use_gpu and context.flash_attn and context.dtw_mem_size==134217728
        assert full.strategy==0 and full.no_context and not full.translate and full.language==b'en'
        assert full.greedy.best_of==5 and full.beam_search.beam_size==-1
        assert abs(full.temperature_inc-.2)<1e-6 and not full.vad
        assert abs(full.vad_params.threshold-.5)<1e-6
        return {'version':self.version().decode(),'context_struct_bytes':c.sizeof(ContextParams),
                'full_struct_bytes':c.sizeof(FullParams),'default_sentinel_fields_verified':True,
                'backend':'Official CPU-only release DLLs; no CUDA DLL in bundle'}
