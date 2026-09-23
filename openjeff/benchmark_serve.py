"""Loopback-only JevBench endpoint for the frozen A100 pilot (8,192-token diagnostic)."""
import argparse
from http.server import HTTPServer
import json
from .contracts import Candidate, DecisionRequest
from .calibration import probabilities
from .serve import handler_for

MODEL = 'openjeff-pilot-v1'


def to_request(payload):
    if not isinstance(payload, dict) or set(payload) != {'state', 'model', 'questions'}:
        raise ValueError('Expected state, model, questions')
    if payload['model'] != MODEL:
        raise ValueError('Unknown model')
    questions = payload['questions']
    if not isinstance(questions, dict) or set(questions) != {'decision'}:
        raise ValueError('Exactly one question named decision required')
    q = questions['decision']
    if not isinstance(q, dict) or set(q) != {'type', 'instructions', 'criteria'}:
        raise ValueError('Expected type, instructions, criteria')
    kind, criteria = q['type'], q['criteria']
    if kind == 'noul':
        if not isinstance(criteria, dict) or set(criteria) != {'false', 'true'}:
            raise ValueError('noul requires false/true criteria')
        descriptions = [('no', criteria['false']), ('yes', criteria['true'])]
    elif kind == 'choice':
        if not isinstance(criteria, dict):
            raise ValueError('choice requires an ordered criteria object')
        descriptions = list(criteria.items())
    elif kind == 'score':
        if not isinstance(criteria, list):
            raise ValueError('score requires a criteria list')
        descriptions = [(str(i), value) for i, value in enumerate(criteria)]
    else:
        raise ValueError('Unsupported question type')
    if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in descriptions):
        raise ValueError('Labels and criteria must be nonempty strings')
    request = DecisionRequest(payload['state'], q['instructions'],
                              tuple(Candidate(k, f'{k}: {v}') for k, v in descriptions))
    return kind, request


def answer(payload, scorer):
    kind, request = to_request(payload)
    scores = scorer.score(request)
    if len(scores) != len(request.candidates):
        raise ValueError('Candidate count mismatch')
    probs = dict(zip((c.id for c in request.candidates), probabilities(scores, scorer.temperature)))
    result = {'type': kind, 'probabilities': probs}
    if kind == 'noul':
        result['noul'] = probs['yes']
    elif kind == 'choice':
        result['choice'] = max(probs, key=probs.get)
    else:
        result['score'] = sum(int(k) * p for k, p in probs.items())
    return {'model': MODEL, 'answers': {'decision': result},
            'calibration_scope': 'synthetic temperature transferred to 8192-token diagnostic; not refitted'}


class BenchmarkScorer:
    def __init__(self, run):
        from .adapters.pilot import PilotScorer
        # This verifies the original weights, packages, prompt, and A100 runtime.
        self.pilot = PilotScorer(run)
        self.temperature = self.pilot.calibration.temperature

    def score(self, request):
        from .prompting import compile_request
        from .training import collate, candidate_logits
        p = self.pilot
        compiled = compile_request(p.tokenizer, request, max_tokens=8192)
        batch = collate([compiled], p.tokenizer.pad_token_id, p.device)
        with p.torch.inference_mode():
            return candidate_logits(p.model, batch)[0].float().cpu().tolist()


def benchmark_handler(scorer):
    class Handler(handler_for(None, 0)):
        def do_GET(self):
            if self.path == '/health':
                return self.send_json(200, {'status': 'ready', 'model': MODEL, 'max_tokens': 8192})
            self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/v1/systemone':
                return self.send_json(404, {'error': 'not found'})
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('Chunked requests unsupported')
                size = int(self.headers.get('Content-Length', '0'))
                if not 1 <= size <= 192_000:
                    raise ValueError('Request size outside limit')
                self.connection.settimeout(10)
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError('Incomplete request')
                result = answer(json.loads(data), scorer)
            except (ValueError, TypeError, KeyError, OverflowError, TimeoutError):
                return self.send_json(400, {'error': 'Invalid request or input exceeds model limits'})
            except Exception:
                return self.send_json(500, {'error': 'Inference failed'})
            self.send_json(200, result)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/pilot-v1')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be 1024..65535')
    scorer = BenchmarkScorer(args.run)
    server = HTTPServer(('127.0.0.1', args.port), benchmark_handler(scorer))
    print(f'{MODEL}: http://127.0.0.1:{args.port}/v1/systemone', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
