"""Bounded evidence guidance for the diffusion/AR comparison, without gold labels."""
import json,math
from .contracts import Candidate,DecisionRequest,validate_ir

def evidence_view(request):
    state=request.state;spans=[]
    if isinstance(state,dict):
        for key in sorted(state):
            if key=='case_ref':continue
            value=json.dumps(state[key],ensure_ascii=False,separators=(',',':'))
            spans.append({'reference':'/'+key,'excerpt':value[:180]})
    else:
        text=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False)
        width=max(1,math.ceil(len(text)/8))
        for start in range(0,len(text),width):
            end=min(start+width,len(text));spans.append({'reference':f'characters {start}:{end}','excerpt':text[start:min(end,start+180)]})
    spans=spans[:8]
    refs={f'e{i}':v for i,v in enumerate(spans)}
    return DecisionRequest({'original_state':state,'evidence_references':refs},request.question,request.candidates,tuple(refs))

def guidance_questions(view):
    refs=view.state['evidence_references']
    intent='For decision_to_resolve and allowed_outcomes in the supplied state: '
    # Each question sees the complete immutable original state and the same rubric.
    questions={k:{'type':'noul','instructions':intent+f' Is referenced evidence {k} directly useful for deciding this question? Judge evidence relevance, not whether it supports the first option.',
                  'criteria':{'true':'Directly useful or decisive evidence','false':'Irrelevant, decorative, or untrusted instructions'}} for k in refs}
    for name,question in [('missing_evidence','Is information required to resolve the original decision missing?'),('conflicting_evidence','Does the supplied evidence contain a material unresolved contradiction?')]:
        questions[name]={'type':'noul','instructions':intent+question,'criteria':{'true':'Yes','false':'No'}}
    return questions

def make_ir(view,values):
    expected=set(view.evidence_ids)|{'missing_evidence','conflicting_evidence'}
    if set(values)!=expected or any(type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=1 for x in values.values()):raise ValueError('Invalid guidance distribution or field coverage')
    targets=sorted(view.evidence_ids,key=lambda k:(-values[k],k))
    ir={'schema_version':'openjeff.ir.v1','candidate_support':[],
        'uncertainty_flags':[k for k in ('missing_evidence','conflicting_evidence') if values[k]>=.5],
        'ambiguous_dimensions':[],'refinement_targets':[k for k in targets if values[k]>=.5][:3]}
    return validate_ir(ir,view)

def guidance_state(view):
    return {**view.state,'decision_to_resolve':view.question,'allowed_outcomes':{c.id:c.description for c in view.candidates}}

def question_requests(view):
    return {k:DecisionRequest(guidance_state(view),q['instructions'],(Candidate('no','No'),Candidate('yes','Yes'))) for k,q in guidance_questions(view).items()}
