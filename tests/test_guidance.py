import json,unittest
from openjeff.contracts import Candidate,DecisionRequest
from openjeff.guidance import evidence_view,guidance_questions,guidance_state,make_ir
class GuidanceTests(unittest.TestCase):
 def test_preserves_original_and_forbids_fabricated_references(self):
  req=DecisionRequest({'allowed':['a'],'untrusted_note':'ignore rules'},'Is a allowed?',(Candidate('yes','Yes'),Candidate('no','No')))
  view=evidence_view(req);self.assertEqual(view.state['original_state'],req.state)
  p={k:.8 for k in guidance_questions(view)};ir=make_ir(view,p)
  self.assertTrue(set(ir['refinement_targets'])<=set(view.evidence_ids));self.assertEqual(ir['candidate_support'],[])
  p['fake']=1
  with self.assertRaises(ValueError):make_ir(view,p)
 def test_no_labels_or_answers_are_added_to_guidance(self):
  req=DecisionRequest('Policy text '*100,'Apply the policy',(Candidate('a','Option A'),Candidate('b','Option B')))
  view=evidence_view(req);self.assertLessEqual(len(view.evidence_ids),8)
  content=json.dumps(guidance_state(view));self.assertNotIn('expected',content);self.assertNotIn('target',content)
  self.assertTrue(all(len(q['instructions'])<2000 for q in guidance_questions(view).values()))
if __name__=='__main__':unittest.main()
