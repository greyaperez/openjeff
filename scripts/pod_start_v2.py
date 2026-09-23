"""Runpod CMD wrapper: arm a detached own-pod stop, then start standard services.

Only existing injected credentials are used. No CLI config, key creation, or
credential logging. The provider can still reject/fail a shutdown request.
"""
import json,os,re,time,urllib.parse,urllib.request

def arm(deadline, *, environ=None, clock=time.time, fork=os.fork, sleep=time.sleep, opener=urllib.request.urlopen, execv=os.execv):
    env=os.environ if environ is None else environ
    pod,key=env.get('RUNPOD_POD_ID',''),env.get('RUNPOD_API_KEY','')
    if not re.fullmatch(r'[a-z0-9]{8,32}',pod) or not key:raise ValueError('Existing own-pod credentials missing')
    remaining=deadline-clock()
    if not 0<remaining<=10800:raise ValueError('Absolute deadline must be within three hours')
    child=fork()
    if child:
        print(json.dumps({'event':'watchdog_started','pod_id':pod,'deadline_unix':deadline,'watchdog_pid':child}),flush=True)
        execv('/start.sh',['/start.sh'])
        return
    # The child retains its inherited credential and its fixed own-pod ID.
    os.setsid()
    while clock()<deadline:sleep(min(5,max(0,deadline-clock())))
    query='mutation stopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }'
    url='https://api.runpod.io/graphql?'+urllib.parse.urlencode({'api_key':key})
    payload=json.dumps({'query':query,'variables':{'podId':pod}}).encode()
    for attempt in range(40):
        try:
            req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
            with opener(req,timeout=20) as response:body=json.load(response)
            result=(body.get('data') or {}).get('podStop') or {}
            if body.get('errors') or result.get('id')!=pod or result.get('desiredStatus')!='EXITED':raise RuntimeError('Stop not confirmed')
            print('{"event":"own_pod_stop_confirmed"}',flush=True)
            return
        except Exception as error:
            print(json.dumps({'event':'shutdown_retry','attempt':attempt+1,'error_type':type(error).__name__}),flush=True)
        sleep(15)
    print('{"event":"shutdown_unconfirmed_manual_cleanup_required"}',flush=True)

if __name__=='__main__':
    import sys
    arm(int(sys.argv[1]))
