import math,unittest
from scripts.diffusion_eval import extract
from openjeff.calibration import probabilities

class DiffusionContractTests(unittest.TestCase):
    def test_complete_distributions_and_zero_probability(self):
        scores,p=extract({'answers':{'decision':{'type':'choice','probabilities':{'b':.75,'a':.25}}}},['a','b'])
        self.assertAlmostEqual(probabilities(scores)[0],.25)
        self.assertAlmostEqual(probabilities(scores)[1],.75)
        scores,p=extract({'answers':{'decision':{'type':'noul','noul':1.0}}},['no','yes'])
        self.assertTrue(all(math.isfinite(x) for x in scores))
        self.assertEqual(p,{'no':0.,'yes':1.})
    def test_never_fills_missing_candidates_or_renormalizes_bad_data(self):
        for p in [{'a':1.},{'a':.2,'b':.2},{'a':float('nan'),'b':1.}]:
            with self.assertRaises(ValueError):extract({'answers':{'decision':{'type':'choice','probabilities':p}}},['a','b'])

if __name__=='__main__':unittest.main()
