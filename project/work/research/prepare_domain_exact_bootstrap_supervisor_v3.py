"""Minimal sibling supports already-running manual EXACT initial bootstrap."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
source=HERE/"supervise_domain30_same_science_v2.py"
text=source.read_text(encoding="utf-8");before=hashlib.sha256(source.read_bytes()).hexdigest()
def change(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
change("    require(state['kind']==FIRST_KIND,'This supervisor begins with the actual FIRST domain window')",
       "    require(state['kind'] in (FIRST_KIND,EXACT_KIND),'Initial window must be a pinned FIRST or SAME-run EXACT domain controller')")
change("'controller_observation_age_seconds':age,**worker,'scientific_config_identity_sha256':canonical_sha(science),",
       "'controller_observation_age_seconds':age,'initial_controller_kind':state['kind'],\n        'initial_controller_sha256':state['controller_sha256'],**worker,'scientific_config_identity_sha256':canonical_sha(science),")
change("        require(state['kind']==FIRST_KIND and worker['worker_identity']==initial_bootstrap['worker_identity']",
       "        require(state['kind']==initial_bootstrap['initial_controller_kind']\n            and state['controller_sha256']==initial_bootstrap['initial_controller_sha256']\n            and worker['worker_identity']==initial_bootstrap['worker_identity']")
text=text.replace("pinned_live_domain_controller_metadata_bootstrap_v1","pinned_live_domain_controller_metadata_bootstrap_v3")
text=text.replace("pinned_domain_controller_evidence_verified_window_closure_v1","pinned_domain_controller_evidence_verified_window_closure_v3")
text=text.replace("automatic_domain_same_science_window_supervisor_v2","automatic_domain_same_science_window_supervisor_v3")
text=text.replace("'initial_closure_scope':'Pinned first-controller owned-child close/completion publication; no independent cross-job OS exit query'",
    "'initial_closure_scope':'Pinned initially observed FIRST/EXACT controller owned-child close/completion publication; no independent cross-job OS exit query',\n        'initial_controller_kind':initial['initial_controller_kind'],'initial_controller_sha256':initial['initial_controller_sha256']")
text=text.replace("pinned first-domain controller is live","pinned FIRST/EXACT domain controller is live")
text=text.replace("ROOT reviews current independently finite first controller; no successor launched","ROOT reviews current independently finite initial controller; no successor launched")
path=HERE/"supervise_domain30_same_science_v3.py"
compile(text,str(path),"exec")
with path.open("x",encoding="utf-8",newline="\n") as stream:stream.write(text)
assert hashlib.sha256(source.read_bytes()).hexdigest()==before
print(path,hashlib.sha256(path.read_bytes()).hexdigest(),"frozen_v2",before)
