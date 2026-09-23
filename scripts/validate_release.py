"""Recompute release claims from exported score files; requires no GPU or network."""
import argparse,hashlib,json,math
from pathlib import Path
from openjeff.calibration import Calibration,fit_temperature,digest
from openjeff.evaluation import evaluate
from openjeff.scorer_identity import scorer_id
from openjeff.training import adapter_digest


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise AssertionError((a,b))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default='runs/pilot-v1');args=parser.parse_args()
    run=Path(args.run);summary=json.loads((run/'summary.json').read_text());release=json.loads((run/'release.json').read_text());runtime=summary['runtime']
    for entry in runtime['dataset_manifest']['files']:
        if hashlib.sha256((Path('data/curriculum-v1')/entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise AssertionError('Dataset checksum')
    adapter=adapter_digest(run/'adapter')
    if adapter!=release['package_adapter_sha256'] or adapter!=release['evaluation_adapter_sha256']:raise AssertionError('Adapter checksum')
    meta={k:runtime[k] for k in ('model','revision','prompt','max_tokens','torch','transformers','peft','gpu','dtype','attention','chat_template_sha256','evaluation_batch_size')}
    checked=0
    for label,sha in [('base',None),('adapter',adapter)]:
        expected_id=scorer_id(meta,sha)
        fit=read_rows(run/f'{label}-calibration_fit.jsonl')
        cal=Calibration(**json.loads((run/f'{label}-calibration.json').read_text()))
        close(fit_temperature(fit).temperature,cal.temperature)
        if cal.scorer_id!=expected_id:raise AssertionError('Calibration identity')
        for split in ('calibration_check','final','adversarial','production_sim'):
            rows=read_rows(run/f'{label}-{split}.jsonl');source={r['id']:r for r in read_rows(f'data/curriculum-v1/{split}.jsonl')}
            if len(rows)!=len(source):raise AssertionError('Missing score rows')
            cal.validate_evaluation(rows)
            for row in rows:
                src=source[row['id']]
                for key in ('target','input_sha256','group_id','family','split'):
                    if row[key]!=src[key]:raise AssertionError('Score/source mismatch')
                if row['scorer_id']!=expected_id or len(row['scores'])!=len(src['request']['candidates']):raise AssertionError('Scorer mismatch')
            for mode,temp in [('raw',1.),('calibrated',cal.temperature)]:
                actual=evaluate(rows,temp);claimed=summary['models'][label]['splits'][split][mode]
                for key in ('accuracy','nll','brier_sum_over_classes','ece','wrong_high_confidence_fraction'):close(actual[key],claimed[key])
            checked+=len(rows)
    train=read_rows(run/'training.jsonl')
    if len(train)!=runtime['training']['steps'] or [r['step'] for r in train]!=list(range(1,len(train)+1)):raise AssertionError('Incomplete training')
    dev=[]
    for path in run.glob('adapter-development-step*.jsonl'):
        step=int(path.stem.split('step')[-1]);m=evaluate(read_rows(path));dev.append((m['accuracy'],-m['nll'],-step))
    best=max(dev);base=evaluate(read_rows(run/'base-development.jsonl'))
    expected_choice='adapter' if best[:2]>(base['accuracy'],-base['nll']) else 'base'
    if -best[2]!=summary['best_step'] or expected_choice!=release['selected_backend']:raise AssertionError('Development selection mismatch')
    service=json.loads((run/'service-verification.json').read_text())
    if any(not x['scorer_id_match'] or x['max_probability_error']>1e-6 for x in service['saved_adapter_reload_and_http_parity']):raise AssertionError('Service parity')
    result={'status':'passed','heldout_score_rows_verified':checked,'temperature_fits_recomputed':2,'adapter_hash':adapter,
        'training_steps':len(train),'training_examples_seen':train[-1]['examples_seen'],'selected_backend':expected_choice,
        'checks':['dataset hashes','adapter hashes','score/source alignment','calibration separation','metrics recomputation','development-only checkpoint selection','saved weights and HTTP parity']}
    Path('reports/release-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
