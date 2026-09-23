from types import SimpleNamespace
import unittest
from openjeff.serve import decide

class ServeTests(unittest.TestCase):
    def test_probabilities_and_abstention(self):
        scorer=SimpleNamespace(scorer_id='frozen',calibration=SimpleNamespace(scorer_id='frozen',temperature=2,artifact_id='cal'),score=lambda req:[2,0])
        req={'state':{},'question':'Choose','candidates':[{'id':'a','description':'A'},{'id':'b','description':'B'}]}
        result=decide(req,scorer)
        self.assertTrue(result['abstained']);self.assertIsNone(result['decision'])
        self.assertAlmostEqual(sum(result['probabilities'].values()),1)
        self.assertEqual(decide(req,scorer,.7)['decision'],'a')
        scorer.calibration.scorer_id='wrong'
        with self.assertRaises(ValueError):decide(req,scorer)
    def test_rejects_tool_and_candidate_injection(self):
        with self.assertRaises(ValueError):decide({'tools':[{'execute':'anything'}]},None)
        with self.assertRaises(ValueError):decide({'state':{},'question':'x','candidates':[{'id':'a','description':'a','execute':'x'}]},None)

if __name__=='__main__':unittest.main()
