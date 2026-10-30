"""Promote the successful independent auditor to an arbitrary-model CLI."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
original=(ROOT/'work/audit_baseline_robustness.py').read_text(encoding='utf-8')

def replace(old,new):
    global original
    assert original.count(old)==1,(old,original.count(old))
    original=original.replace(old,new)

replace('"""Independent CPU recalculation of the actual baseline perturbation report."""',
'''"""Independently verify a saved Parakeet confidence perturbation experiment.

CPU-only; does not load model weights, run inference, or change existing helpers.
Example:
  python verify_confidence_robustness.py --input runs/RUN/confidence_robustness.json \
      --reference v3=results/base_v3 --reference selected=results/RUN_adapted

Reference directories contain metadata.json and validation_en.jsonl. Paths are
resolved from the caller's current directory; use absolute paths when elsewhere.
Every model requires an external reference (v2/v3 defaults use saved baselines).
Config identity normalization is accepted only when both hashes are exactly
reconstructed and all other fingerprint components match. Exported canonical
config paths require checked finalization_provenance.json with file hashes.
"""''')
replace('import hashlib','import argparse\nimport hashlib')
replace("ROOT=Path(__file__).resolve().parents[1]","ROOT=Path(__file__).resolve().parents[2]")
replace("source=OUTPUT/'baseline_confidence_robustness.json'\nreport=json.loads(source.read_text(encoding='utf-8'))\nmanifest=ROOT/'work/data/validation_en.jsonl'\nexpected=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()][:10]",'''parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path)
parser.add_argument('--manifest',type=Path,default=ROOT/'work/data/validation_en.jsonl')
parser.add_argument('--reference',action='append',default=[],metavar='NAME=RESULT_DIRECTORY')
parser.add_argument('--resamples',type=int,default=2000)
args=parser.parse_args()
if args.resamples<1:
    parser.error('--resamples must be positive')
source=args.input
report=json.loads(source.read_text(encoding='utf-8'))
manifest=args.manifest
count=len(report['samples'])
if not count:
    parser.error('The input experiment has no selected recordings')
expected=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()][:count]
references={'v2':OUTPUT/'results/base_v2','v3':OUTPUT/'results/base_v3'}
for specification in args.reference:
    name,delimiter,path=specification.partition('=')
    if not delimiter or not name or not path:
        parser.error('--reference requires NAME=RESULT_DIRECTORY')
    references[name]=Path(path)
for name in report['models']:
    if name not in references:
        parser.error(f'External benchmark reference missing for {name!r}; supply --reference {name}=RESULT_DIRECTORY')
output=args.output or source.with_name(source.stem+'_independent_audit.json')
seed=int(report['seed'])''')
replace("check('ordered fixed ten recording IDs',[s['source_id'] for s in report['samples']]==ids)",
        "check('ordered fixed development recording IDs',[s['source_id'] for s in report['samples']]==ids and len(ids)==count)")
replace("check('same fixed perturbation seed',report['seed']==20261010)",
        "check('valid perturbation seed',isinstance(report['seed'],int) and not isinstance(report['seed'],bool))")
replace('f"20261010:{row[\'source_id\']}"', 'f"{seed}:{row[\'source_id\']}"')
replace("baseline=OUTPUT/'results'/('base_v2' if 'v2' in name else 'base_v3')", "baseline=references[name]")
replace("        configuration['_name_or_path']=model['model_path'].replace('/','\\\\')\n        check('baseline config differs only path spelling '+name,digest(configuration)==original_identity['model_config_sha256'])\n        fingerprint_notes.append({'model':name,'difference':'Only _name_or_path slash spelling: work/models/parakeet-v2 versus work\\\\models\\\\parakeet-v2.',\n            'all_weights_generation_tokenizer_processor_code_dtypes_match':True,'both_config_hashes_exactly_reconstructed':True})",'''        candidates=[model['model_path'].replace('/','\\\\'),model['model_path'].replace('\\\\','/')]
        provenance_path=Path(model['model_path'])/'finalization_provenance.json'
        provenance_evidence=None
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
            canonical=provenance.get('canonical_model_name_or_path')
            hashes=provenance.get('candidate_files_sha256')
            if not canonical or not isinstance(hashes,dict) or not hashes:
                raise ValueError('Canonical provenance requires explicit path identity and exported file SHA256 evidence.')
            for relative,digest_value in hashes.items():
                target=(provenance_path.parent/relative).resolve()
                if not target.is_relative_to(provenance_path.parent.resolve()):
                    raise ValueError('Exported provenance file path is outside its model directory.')
                check('exact published checkpoint file '+name+relative,
                      target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==digest_value)
            candidates.append(canonical)
            provenance_evidence={'path':str(provenance_path.resolve()),
                'sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                'canonical_model_name_or_path':canonical,'exported_files_verified':len(hashes)}
        matched_path=None
        for alternate in candidates:
            configuration['_name_or_path']=alternate
            if digest(configuration)==original_identity['model_config_sha256']:
                matched_path=alternate
                break
        check('reference config differs only proven name/path identity '+name,matched_path is not None)
        fingerprint_notes.append({'model':name,'difference':'Only serialized _name_or_path differs.',
            'probe_name_or_path':model['model_path'],'reference_name_or_path':matched_path,
            'all_weights_generation_tokenizer_processor_code_dtypes_match':True,
            'both_config_hashes_exactly_reconstructed':True,'exported_provenance':provenance_evidence})''')
replace("check('forty predictions '+name,len(model['records'])==40)",
        "check('complete predictions '+name,len(model['records'])==count*len(report['variants']))")
replace('rng=random.Random(20261010); means=[]','rng=random.Random(seed); means=[]')
replace('for _ in range(2000):','for _ in range(args.resamples):')
replace("'Ten fixed English development recordings; descriptive, not an independent robustness test.'", "f'{count} fixed English development recordings; descriptive, not an independent robustness test.'")
replace("'Bootstrap resamples only ten recordings and does not turn this small probe into strong general evidence.'", "f'Bootstrap resamples only {count} recordings and does not turn this small probe into strong general evidence.'")
replace("(OUTPUT/'baseline_confidence_robustness_independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')",
        "audit['reference_directories']={name:str(references[name].resolve()) for name in report['models']}\naudit['bootstrap_resamples']=args.resamples\noutput.parent.mkdir(parents=True,exist_ok=True)\noutput.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')")
replace("print(json.dumps({key:audit[key] for key in ('passed','check_count','maximum_numerical_recalculation_difference','models','paired_model_comparison')},ensure_ascii=False,indent=2))",
        "print(json.dumps({'passed':audit['passed'],'check_count':audit['check_count'],'maximum_numerical_recalculation_difference':audit['maximum_numerical_recalculation_difference'],'models':list(audit['models']),'model_pairs':list(audit['paired_model_comparison']),'output':str(output.resolve())},ensure_ascii=False,indent=2))")
(ROOT/'outputs/parakeet_turkish/verify_confidence_robustness.py').write_text(original,encoding='utf-8')
print('Reusable CPU perturbation auditor exported.')
