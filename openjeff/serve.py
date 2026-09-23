"""Local typed decision HTTP interface. Bind loopback only; no tool execution."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from .calibration import probabilities
from .contracts import Candidate, DecisionRequest


def decide(payload, scorer, min_confidence=.9):
    if not isinstance(payload, dict) or set(payload) - {'state','question','candidates','evidence_ids'}:
        raise ValueError('Expected state, question, candidates, and optional evidence_ids')
    if not isinstance(payload.get('candidates'), list):raise ValueError('candidates must be a list')
    candidates=[]
    for candidate in payload['candidates']:
        if not isinstance(candidate,dict) or set(candidate)!={'id','description'}:
            raise ValueError('Each candidate requires id and description')
        candidates.append(Candidate(**candidate))
    req=DecisionRequest(payload['state'],payload['question'],tuple(candidates),tuple(payload.get('evidence_ids',())))
    if scorer.calibration.scorer_id!=scorer.scorer_id:raise ValueError('Calibrator does not match scorer')
    scores=scorer.score(req)
    if len(scores)!=len(candidates):raise ValueError('Scorer omitted a candidate')
    p=probabilities(scores,scorer.calibration.temperature)
    winner=max(range(len(p)),key=p.__getitem__)
    accepted=p[winner]>=min_confidence
    return {'schema_version':'openjeff.decision.v1','decision':candidates[winner].id if accepted else None,
        'suggested_candidate':candidates[winner].id,'abstained':not accepted,
        'probabilities':{c.id:v for c,v in zip(candidates,p)},'confidence':p[winner],
        'min_confidence':min_confidence,'scorer_id':scorer.scorer_id,
        'calibration_id':scorer.calibration.artifact_id,
        'scope':'experimental; confidence calibrated on synthetic rule tasks'}


def handler_for(scorer, threshold):
    class Handler(BaseHTTPRequestHandler):
        server_version='OpenJeff/0.1'
        def log_message(self,format,*args):pass # Never log request bodies or evidence.
        def send_json(self,code,value):
            data=json.dumps(value,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':return self.send_json(200,{'status':'ready','scorer_id':scorer.scorer_id})
            self.send_json(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/v1/decide':return self.send_json(404,{'error':'not found'})
            try:
                if self.headers.get('Transfer-Encoding'):raise ValueError('Chunked requests unsupported')
                n=int(self.headers.get('Content-Length','0'))
                if not 1<=n<=192_000:raise ValueError('Request size outside limit')
                self.connection.settimeout(10)
                data=self.rfile.read(n)
                if len(data)!=n:raise ValueError('Incomplete request body')
                result=decide(json.loads(data),scorer,threshold)
            except (ValueError,TypeError,KeyError,OverflowError,TimeoutError) as exc:
                return self.send_json(400,{'error':str(exc)})
            except Exception:
                return self.send_json(500,{'error':'Inference failed'})
            self.send_json(200,result)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--min-confidence',type=float,default=.9)
    args=parser.parse_args()
    if not math.isfinite(args.min_confidence) or not 0<=args.min_confidence<=1:parser.error('threshold must be in [0,1]')
    if not 1024<=args.port<=65535:parser.error('port must be 1024..65535')
    from .adapters.pilot import PilotScorer
    scorer=PilotScorer(args.run)
    server=HTTPServer(('127.0.0.1',args.port),handler_for(scorer,args.min_confidence))
    print(f'OpenJeff ready at http://127.0.0.1:{args.port}/v1/decide',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
