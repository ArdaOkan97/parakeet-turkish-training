"""Create a separately pinned lifetime-only loss revision, not admitted sources."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from bpe_broad_audio_v1 import require,record
from atomic_io_retry_v1 import write_json

def main():
    before=(S/'tdt_tiled_full_native_v5.py').read_text()
    old="""            class ExactFullDenominator(gpu_rnnt.GPUTDT):
                def log_softmax(self,unused_acts,unused_denom):workspace[:cells].copy_(denom)
            wrapper=ExactFullDenominator("""
    new="""            from tdt_exact_denominator_instance_v1 import exact_denominator_type,bind,release
            wrapper=exact_denominator_type(gpu_rnnt.GPUTDT)("""
    require(before.count(old)==1,'Exact old closure-only delta required')
    after=before.replace(old,new)
    needle="""            lengths_t=torch.tensor([t],device=e.device,dtype=torch.long);lengths_u=torch.tensor([u],device=e.device,dtype=torch.long)"""
    after=after.replace(needle,"            bind(wrapper,workspace,denom,cells)\n"+needle)
    after=after.replace("            finally:random.setstate(before)","            finally:\n                random.setstate(before);release(wrapper);del wrapper")
    ast.parse(after)
    with (S/'tdt_tiled_full_native_v6.py').open('x',encoding='utf-8',newline='\n') as f:f.write(after)
    inverse=after.replace(new,old).replace("            bind(wrapper,workspace,denom,cells)\n",'').replace("            finally:\n                random.setstate(before);release(wrapper);del wrapper","            finally:random.setstate(before)")
    require(inverse==before,'Every arithmetic/kernel/source byte unchanged by inverse lifetime delta')
    write_json(A/'candidate_instance_denominator_delta_v1.json',{'kind':'ASR18_loss_instance_lifetime_delta_v1','unadmitted':True,'inverse_reconstructs_v5_exactly':True,
        'bindings':[record(S/n) for n in ('tdt_tiled_full_native_v5.py','tdt_tiled_full_native_v6.py','tdt_exact_denominator_instance_v1.py')],
        'measured_full_training_memory_improvement':False,'numerical_equivalence_proven':False,'changes':['Cache one CPU subclass per official base','Move denominator/workspace/cell references onto short-lived instance','Explicitly release instance bindings after synchronous official cost call'],
        'other_local_autograd_class_cycles_not_claimed_fixed':True,'installed_sources_unchanged':True,'production_ready':False})
    print(json.dumps(record(A/'candidate_instance_denominator_delta_v1.json')))

if __name__=='__main__':main()
