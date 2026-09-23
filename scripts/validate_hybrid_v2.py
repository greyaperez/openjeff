"""Independently verify saved experiment inputs, guidance, calibration and fusion."""
import argparse,json,math
from pathlib import Path
from openjeff.calibration import fit_temperature,validate_rows,mean_nll,probabilities
from openjeff.guidance import evidence_view,make_ir
from openjeff.training import adapter_digest
from scripts.hybrid_common import ROOT,SPLITS,cases,read
from scripts.analyze_hybrid_v2 import METHODS,mix,aligned

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='runs/hybrid-v2');a=p.parse_args();out=Path(a.out)
    rows={m:{s:read(out/f'{m}-{s}.jsonl') for s in SPLITS} for m in METHODS}
    count=0
    for s in SPLITS:
        expected=list(cases(s))
        for m in METHODS:
            actual=rows[m][s];validate_rows(actual)
            assert len(actual)==len(expected),(m,s,'row count')
            for (original,req,_),r in zip(expected,actual):
                assert all(original[k]==r[k] for k in ('id','group_id','input_sha256','target','family')),(m,s,r['id'])
                assert len(r['scores'])==len(req.candidates)
                for field in ('latency_s','first_stage_latency_s','serial_pipeline_latency_s'):
                    if field in r:assert math.isfinite(r[field]) and r[field]>=0
                if m in ('diffusion','ar_guided_ar'):
                    assert make_ir(evidence_view(req),r['guidance_values'])==r['ir']
                if m=='diffusion':
                    assert r['candidate_ids']==[c.id for c in req.candidates]
                    expected_scores=[math.log(max(r['probabilities_as_returned'][c.id],1e-300)) for c in req.candidates]
                    assert r['scores']==expected_scores
                count+=1
        for x,y in zip(rows['diffusion'][s],rows['diffusion_guided_ar'][s]):assert x['ir']==y['ir']
    calibration_checks={}
    for m in METHODS:
        fitted=fit_temperature(rows[m]['calibration_fit'])
        saved=json.loads((out/f'{m}-calibration.json').read_text())
        # Python/libm reduction differences can move a flat optimum slightly.
        # Preserve the cloud calibrator and verify objective/probability parity.
        t=fitted.temperature; saved_t=saved['temperature']
        assert math.isclose(t,saved_t,rel_tol=1e-6,abs_tol=1e-8),(m,'temperature')
        nll_delta=abs(mean_nll(rows[m]['calibration_fit'],t)-mean_nll(rows[m]['calibration_fit'],saved_t))
        probability_delta=max(abs(a-b) for rs in rows[m].values() for r in rs for a,b in zip(probabilities(r['scores'],t),probabilities(r['scores'],saved_t)))
        assert nll_delta<=1e-12 and probability_delta<=5e-7,(m,'calibration parity')
        calibration_checks[m]={'saved_temperature':saved_t,'recomputed_temperature':t,'fit_nll_absolute_difference':nll_delta,'maximum_probability_difference_all_splits':probability_delta}
        assert fitted.scorer_id==saved['scorer_id']
        assert fitted.fit_data_sha256==saved['fit_data_sha256']
        for s in ('calibration_check','final','adversarial','production_sim','public'):
            fitted.validate_evaluation([{**r,'split':'test'} for r in rows[m][s]] if s=='public' else rows[m][s])
    selection=json.loads((out/'fusion-selection.json').read_text())
    candidates=[(mean_nll(mix(rows['ar']['development'],rows['diffusion']['development'],w),1),w) for w in (0,.25,.5,.75,1)]
    w=min(candidates)[1];assert w==selection['selected_weight_diffusion']
    for s in SPLITS:
        expected=mix(rows['ar'][s],rows['diffusion'][s],w)
        assert expected==rows['fusion'][s]
    fingerprint=adapter_digest(ROOT/'runs/pilot-v1/adapter')
    assert fingerprint=='a1811874542eb3b9ca4235d79826dfabb847f0c5ad076bfd25b5e4b2f426ba2b'
    result={'status':'passed','validated_score_rows':count,'methods':list(METHODS),'splits':list(SPLITS),'calibration_fits_recomputed':6,'fusion_selection_recomputed_from_development_only':True,'guidance_reconstructed_from_recorded_outputs':True,'original_adapter_sha256':fingerprint,'scope':'Saved-score validation; does not rerun GPU inference or establish benchmark contamination absence.'}
    result['calibration_checks']=calibration_checks
    result['calibration_tolerances']={'temperature_relative':1e-6,'temperature_absolute':1e-8,'fit_nll_absolute':1e-12,'maximum_probability_absolute':5e-7}
    (ROOT/'reports/hybrid-v2-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
