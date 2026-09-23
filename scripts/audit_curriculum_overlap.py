"""Report repeated synthetic states after removing decorative case identifiers.

This post-hoc diagnostic never changes weights, calibration, or headline scores.
It does not establish semantic novelty for states that fail this literal match.
"""
import collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
def state_key(row):
    state=dict(row['request']['state']);tag=state.pop('case_ref')
    return json.dumps({'family':row['family'],'state':state},sort_keys=True,separators=(',',':')).replace(tag,'<CASE>')
def main():
    splits=('train','development','calibration_fit','calibration_check','final','adversarial','production_sim')
    rows={s:read(ROOT/f'data/curriculum-v1/{s}.jsonl') for s in splits}
    keys={s:{state_key(r) for r in rs} for s,rs in rows.items()}
    counts={s:{'rows':len(rs),'unique_normalized_states':len(keys[s]),'rows_matching_train':sum(state_key(r) in keys['train'] for r in rs)} for s,rs in rows.items() if s!='train'}
    fit_overlap={s:sum(state_key(r) in keys['calibration_fit'] for r in rows[s]) for s in ('calibration_check','final','adversarial','production_sim')}
    final={r['id']:r for r in rows['final']};breakdown={}
    for label in ('base','adapter'):
        score=read(ROOT/f'runs/pilot-v1/{label}-final.jsonl')
        groups={'matches_training_state':[],'does_not_match_training_state':[]}
        for row in score:groups['matches_training_state' if state_key(final[row['id']]) in keys['train'] else 'does_not_match_training_state'].append(row)
        breakdown[label]={}
        for name,rs in groups.items():
            correct=sum(max(range(len(r['scores'])),key=r['scores'].__getitem__)==r['target'] for r in rs)
            breakdown[label][name]={'rows':len(rs),'correct':correct,'accuracy':correct/len(rs)}
    result={'scope':'post-hoc literal normalized-state overlap diagnostic; not semantic novelty or a new benchmark',
        'normalization':'Drop state.case_ref; replace that case tag everywhere in serialized state with <CASE>; preserve family, all other values and array order. Ignore question wording and candidate order.',
        'limitations':'Finite boolean/rule families repeat naturally. Entity renaming and shared generators mean this check is not a proof of novelty. Adversarial messages make that split different under this literal comparison.',
        'training_rows':len(rows['train']),'training_unique_normalized_states':len(keys['train']),
        'splits':counts,'rows_matching_calibration_fit':fit_overlap,'final_breakdown':breakdown,
        'final_train_matches_by_family':dict(collections.Counter(r['family'] for r in rows['final'] if state_key(r) in keys['train']))}
    (ROOT/'reports/curriculum-overlap.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
