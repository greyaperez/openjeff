"""Own-pod bounded shutdown using the official Runpod GraphQL stop primitive.

Run detached after normal image initialization. No CLI configuration, SSH key
creation, credential output, or cross-pod control. The provider API can still
fail; watch logs and verify console cleanup. Not a guaranteed billing cap.
"""
import argparse,json,os,re,time,urllib.parse,urllib.request

def settings(deadline,environ,now):
    pod=environ.get('RUNPOD_POD_ID','');key=environ.get('RUNPOD_API_KEY','')
    if not re.fullmatch(r'[a-z0-9]{8,32}',pod) or not key:raise ValueError('Existing own-pod credentials required')
    if not 0<deadline-now<=10800:raise ValueError('Deadline must be within three hours')
    return pod,key

def stop(pod,key):
    # Matches runpod/runpodctl v1.14.3 api/query.go and api/pod.go.
    query='mutation stopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }'
    url='https://api.runpod.io/graphql?'+urllib.parse.urlencode({'api_key':key})
    req=urllib.request.Request(url,data=json.dumps({'query':query,'variables':{'podId':pod}}).encode(),headers={'Content-Type':'application/json','User-Agent':'OpenJeff-own-pod-shutdown/2'})
    with urllib.request.urlopen(req,timeout=20) as response:body=json.load(response)
    result=body.get('data',{}).get('podStop') or {}
    if body.get('errors') or result.get('id')!=pod or result.get('desiredStatus')!='EXITED':raise RuntimeError('Own-pod stop was not confirmed')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deadline-unix',type=int,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    pod,key=settings(a.deadline_unix,os.environ,time.time())
    print(json.dumps({'event':'configuration_validated' if a.check_only else 'watchdog_armed','pod_id':pod,'deadline_unix':a.deadline_unix,'provider_permission_verified':False}),flush=True)
    if a.check_only:return
    while time.time()<a.deadline_unix:time.sleep(min(5,max(0,a.deadline_unix-time.time())))
    for n in range(40):
        try:stop(pod,key);print('{"event":"stop_confirmed"}',flush=True);return
        except Exception as e:print(json.dumps({'event':'stop_retry','attempt':n+1,'error_type':type(e).__name__}),flush=True)
        time.sleep(15)
    raise SystemExit('Stop unconfirmed; explicit console cleanup required')
if __name__=='__main__':main()
