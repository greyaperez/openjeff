import json
from http.server import HTTPServer
import threading
from urllib.request import Request, urlopen
from pathlib import Path
from types import SimpleNamespace
import unittest
from openjeff.benchmark_serve import MODEL, answer, to_request, benchmark_handler
from scripts.jevbench_public import to_request as original_mapping


class BenchmarkServeTests(unittest.TestCase):
    def test_all_public_requests_preserve_evidence_and_candidate_meanings(self):
        count = 0
        for path in Path('research/snapshots/jevbench/datasets/public').glob('*.jsonl'):
            if path.stem not in ('easy', 'original', 'hard'):
                continue
            for line in path.read_text().splitlines():
                task = json.loads(line)
                q = {k: task['question'][k] for k in ('type', 'instructions', 'criteria')}
                _, req = to_request({'state': task['state'], 'model': MODEL, 'questions': {'decision': q}})
                original = original_mapping(task)
                self.assertEqual(req.state, original.state)
                self.assertEqual(req.question, original.question)
                self.assertEqual({c.id: c.description for c in req.candidates},
                                 {c.id: c.description for c in original.candidates})
                count += 1
        self.assertEqual(count, 231)

    def test_types_and_exact_probabilities(self):
        scorer = SimpleNamespace(temperature=1, score=lambda req: [0, 0])
        for kind, criteria in [('noul', {'false':'No', 'true':'Yes'}),
                               ('choice', {'reject':'Reject', 'accept':'Accept'}),
                               ('score', ['Low', 'High'])]:
            payload = {'state': 'Evidence', 'model': MODEL, 'questions': {'decision':
                       {'type': kind, 'instructions':'Decide', 'criteria':criteria}}}
            result = answer(payload, scorer)['answers']['decision']
            self.assertEqual(result['type'], kind)
            self.assertEqual(list(result['probabilities'].values()), [.5, .5])
            if kind == 'noul': self.assertEqual(result['noul'], .5)
            if kind == 'score': self.assertEqual(result['score'], .5)
            payload['expected'] = 'must not reach model'
            with self.assertRaises(ValueError): to_request(payload)

    def test_unknown_model_and_malformed_criteria(self):
        payload = {'state': {}, 'model': 'unknown', 'questions': {}}
        with self.assertRaises(ValueError): to_request(payload)
        payload = {'state': {}, 'model': MODEL, 'questions': {'decision':
                   {'type':'choice','instructions':'Choose','criteria':{'a':1,'b':2}}}}
        with self.assertRaises(ValueError): to_request(payload)

    def test_http_wire_roundtrip(self):
        scorer = SimpleNamespace(temperature=1, score=lambda req: [0, 0])
        server = HTTPServer(('127.0.0.1', 0), benchmark_handler(scorer))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(base + '/health') as response:
                self.assertEqual(json.load(response)['model'], MODEL)
            for kind, criteria in [('noul', {'false':'No', 'true':'Yes'}),
                                   ('choice', {'reject':'Reject','accept':'Accept'}),
                                   ('score', ['Low','High'])]:
                payload = {'state': 'Do not log me', 'model': MODEL, 'questions': {'decision':
                           {'type':kind,'instructions':'Decide','criteria':criteria}}}
                request = Request(base + '/v1/systemone', data=json.dumps(payload).encode(),
                                  headers={'Content-Type':'application/json'})
                with urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(result['answers']['decision']['type'], kind)
                self.assertNotIn('Do not log me', json.dumps(result))
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
