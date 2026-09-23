"""Tiny paid CPU-pod diagnostic: print safe status, then stop only this pod."""
import json,os,time,urllib.parse,urllib.request
pod,key=os.getenv('RUNPOD_POD_ID',''),os.getenv('RUNPOD_API_KEY','')
print(json.dumps({'event':'diagnostic_started','pod_id':pod,'key_present':bool(key)}),flush=True)
for name in ('RUNPOD_API_URL','RUNPOD_GRAPHQL_URL'):
    value=urllib.parse.urlsplit(os.getenv(name,''))
    print(json.dumps({'setting':name,'scheme':value.scheme,'host':value.hostname,'path':value.path}),flush=True)
time.sleep(30)
query='mutation stopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }'
base=os.getenv('RUNPOD_GRAPHQL_URL') or os.getenv('RUNPOD_API_URL') or 'https://api.runpod.io/graphql'
if '/graphql' not in base:base='https://api.runpod.io/graphql'
for auth in ('query','header'):
    try:
        url=base+('?' +urllib.parse.urlencode({'api_key':key}) if auth=='query' else '')
        headers={'Content-Type':'application/json','User-Agent':'RunPod-CLI/1.14.3 (linux/amd64)'}
        if auth=='header':headers['Authorization']='Bearer '+key
        req=urllib.request.Request(url,data=json.dumps({'query':query,'variables':{'podId':pod}}).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=20) as response:body=json.load(response)
        result=(body.get('data') or {}).get('podStop') or {}
        errors=[str(e.get('message',''))[:300].replace(key,'<redacted>') for e in body.get('errors',[])]
        print(json.dumps({'auth_mode':auth,'pod_id':result.get('id'),'status':result.get('desiredStatus'),'errors':errors}),flush=True)
        if result.get('id')==pod and result.get('desiredStatus')=='EXITED':break
    except Exception as error:
        print(json.dumps({'auth_mode':auth,'error_type':type(error).__name__,'http_code':getattr(error,'code',None)}),flush=True)
    time.sleep(5)
# Keep diagnostics available briefly if stop is rejected; provider console cleanup required.
time.sleep(120)
