import unittest
from scripts.jevbench_public import to_request

class BenchmarkMappingTests(unittest.TestCase):
    def test_only_state_and_rubric_reach_model(self):
        task={'state':'public evidence','question':{'type':'noul','instructions':'Apply rule','criteria':{'false':'not satisfied','true':'satisfied'}},'labels':['no','yes'],'expected':'PRIVATE_GOLD_SENTINEL','provenance':{'rationale':'PRIVATE_RATIONALE_SENTINEL'}}
        req=to_request(task)
        self.assertEqual(req.candidates[0].description,'no: not satisfied')
        self.assertNotIn('PRIVATE_',str(req))
    def test_ordinal_preserves_levels_and_rubric(self):
        task={'state':{},'question':{'type':'score','instructions':'Rate','criteria':['low','high']},'labels':['0','1']}
        self.assertEqual([c.description for c in to_request(task).candidates],['0: low','1: high'])

if __name__=='__main__':unittest.main()
