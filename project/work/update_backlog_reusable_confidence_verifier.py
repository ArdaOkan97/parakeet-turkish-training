"""Record reusable verifier without changing active model-quality statuses."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
subprocess.run([str(OUTPUT/'tools/backlog.exe'),'task','edit','ASR-7','--append-notes',
    'The independent CPU perturbation auditor is now reusable as verify_confidence_robustness.py --input PATH with external --reference NAME=RESULT_DIRECTORY arguments for arbitrary model pairs, including v3/selected. The latest real baseline recheck passes 6725 checks with zero metric recalculation difference, adding explicit fingerprint-hash integrity. Config-name/path differences are accepted only after both hashes are reconstructed exactly and every other fingerprint component agrees; exported canonical identities also require finalization_provenance.json with streamed candidate-file SHA verification. CPU CLI regressions pass arbitrary aliases/canonical identity and reject tampered copy/generation metadata. This verifier is handed off to the final selected-checkpoint worker; no selected-pilot inference or quality result is asserted here.',
    '--add-ref','verify_confidence_robustness.py','--add-ref','verify_confidence_robustness_verification.json'],
    cwd=OUTPUT,check=True,capture_output=True)
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(OUTPUT/'refresh_backlog.py')],cwd=OUTPUT,check=True)
