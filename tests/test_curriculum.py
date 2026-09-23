import tempfile
import unittest
from pathlib import Path
from openjeff.curriculum import FAMILIES, build_curriculum, make_case, reference_answer
from openjeff.probe_cases import to_request

class CurriculumTests(unittest.TestCase):
    def test_generated_answers_and_permutations(self):
        for family in FAMILIES:
            for split in ('train', 'final', 'adversarial', 'production_sim'):
                for i in range(30):
                    row = make_case(family, split, i)
                    req = to_request(row)
                    self.assertEqual(reference_answer(family, req.state), row['answer_id'])
                    self.assertEqual(req.candidates[row['target']].id, row['answer_id'])
                    self.assertEqual(make_case(family, split, i), row)
    def test_boundaries(self):
        self.assertEqual(reference_answer('interval_overlap', dict(a_start=1,a_end=3,b_start=3,b_end=4)), 'no')
        self.assertEqual(reference_answer('schema_validation', {'record': {'id':'x','count':True,'enabled':True}}), 'invalid')
        self.assertEqual(reference_answer('unit_conversion', {'kilograms':'0.001','grams':1}), 'equal')
        self.assertEqual(reference_answer('verified_consent', {}), 'unknown')
        self.assertEqual(reference_answer('policy_precedence', dict(blocked=True,urgent=True,member=True)), 'deny')
    def test_reproducible_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            one = build_curriculum(a, {'train': 3, 'final': 2})
            self.assertEqual(one, build_curriculum(b, {'train': 3, 'final': 2}))
            self.assertEqual(len(one['files']), 2)
            self.assertEqual(one['files'][0]['rows'], 60)

if __name__ == '__main__': unittest.main()
