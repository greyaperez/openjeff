import unittest
from scripts.analyze_hybrid_v2 import mix,paired
from openjeff.calibration import probabilities

class HybridAnalysisTests(unittest.TestCase):
    def test_probability_mixture_endpoints_and_alignment(self):
        a=[{'id':'a','target':0,'input_sha256':'a'*64,'scorer_id':'ar','scores':[2.,0.],'latency_s':.1,'family':'f'}]
        b=[{**a[0],'scorer_id':'diffusion','scores':[0.,2.],'latency_s':.2}]
        self.assertAlmostEqual(probabilities(mix(a,b,0)[0]['scores'])[0],probabilities(a[0]['scores'])[0])
        self.assertAlmostEqual(probabilities(mix(a,b,1)[0]['scores'])[0],probabilities(b[0]['scores'])[0])
        self.assertAlmostEqual(probabilities(mix(a,b,.5)[0]['scores'])[0],.5)
        self.assertAlmostEqual(mix(a,b,0)[0]['latency_s'],.1)
        self.assertAlmostEqual(mix(a,b,1)[0]['latency_s'],.2)
        self.assertAlmostEqual(mix(a,b,.5)[0]['latency_s'],.3)
        with self.assertRaises(ValueError):mix(a,[{**b[0],'id':'b'}],.5)
        report=paired(a,b);self.assertEqual(report['broken'],1);self.assertEqual(report['repaired'],0)

if __name__=='__main__':unittest.main()
