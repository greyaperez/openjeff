"""Exercise saved weights, calibration, and the loopback HTTP interface on GPU."""
import json
from pathlib import Path
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer
from openjeff.adapters.pilot import PilotScorer
from openjeff.serve import handler_for
from openjeff.calibration import probabilities


def main():
    run=Path('runs/pilot-v1')
    scorer=PilotScorer(run)
    server=HTTPServer(('127.0.0.1',0),handler_for(scorer,.9))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    origin=f'http://127.0.0.1:{server.server_port}'
    health=json.load(urllib.request.urlopen(origin+'/health',timeout=20))
    selected=json.loads((run/'release.json').read_text())['selected_backend']
    expected={r['id']:r for r in map(json.loads,(run/f'{selected}-calibration_check.jsonl').read_text().splitlines())}
    source=[json.loads(s) for s in Path('data/curriculum-v1/calibration_check.jsonl').read_text().splitlines()][:3]
    checks=[]
    try:
        for row in source:
            req=urllib.request.Request(origin+'/v1/decide',data=json.dumps(row['request']).encode(),headers={'Content-Type':'application/json'})
            actual=json.load(urllib.request.urlopen(req,timeout=60))
            golden=probabilities(expected[row['id']]['scores'],scorer.calibration.temperature)
            error=max(abs(actual['probabilities'][c['id']]-p) for c,p in zip(row['request']['candidates'],golden))
            if error>1e-6:raise AssertionError(f'Saved adapter parity failed: {error}')
            checks.append({'id':row['id'],'max_probability_error':error,'scorer_id_match':actual['scorer_id']==expected[row['id']]['scorer_id']})
        bad=urllib.request.Request(origin+'/v1/decide',data=b'{"tools": [{"execute":"invalid"}]}',headers={'Content-Type':'application/json'})
        try:urllib.request.urlopen(bad,timeout=20);raise AssertionError('Invalid request accepted')
        except urllib.error.HTTPError as e:
            if e.code!=400:raise
        result={'health':health,'saved_adapter_reload_and_http_parity':checks,'invalid_request_status':400,'bound_address':'127.0.0.1','publicly_exposed':False}
        (run/'service-verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:server.shutdown();server.server_close();thread.join(timeout=10)

if __name__=='__main__':main()
