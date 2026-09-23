"""Foreground, deadline-bound pod bootstrap; payload contains only public research.

The startup command supplies an immutable bundle SHA256. Results and status live
on the ordinary pod volume; dependencies/model caches live on ephemeral storage.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request


def log(event, **fields):
    print(json.dumps({'event': event, 'epoch': int(time.time()), **fields}), flush=True)


def stop_pod():
    pod, key = os.environ['RUNPOD_POD_ID'], os.environ['RUNPOD_API_KEY']
    query = 'mutation stopPod($podId: String!) { podStop(input: {podId: $podId}) { id desiredStatus } }'
    url = 'https://api.runpod.io/graphql?' + urllib.parse.urlencode({'api_key': key})
    for attempt in range(12):
        try:
            request = urllib.request.Request(url, data=json.dumps({'query': query, 'variables': {'podId': pod}}).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'RunPod-CLI/1.14.3 (linux/amd64)'})
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.load(response)
            result = (body.get('data') or {}).get('podStop') or {}
            log('stop_response', pod_id=result.get('id'), status=result.get('desiredStatus'), error_count=len(body.get('errors', [])))
            if result.get('id') == pod and result.get('desiredStatus') == 'EXITED':
                time.sleep(120)
                return
        except Exception as error:
            log('stop_error', error_type=type(error).__name__, http_code=getattr(error, 'code', None))
        time.sleep(5)
    raise RuntimeError('Provider stop failed; manual cleanup required')


def unpack(bundle, destination, expected):
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected:
        return False
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or path.parts[0] != 'openjeff-diffusion':
                raise ValueError('Unsafe archive member')
        archive.extractall(destination, members=members)
    return True


def main(expected):
    workspace = Path('/workspace')
    workspace.mkdir(exist_ok=True)
    state = workspace / 'openjeff-supervisor-state.json'
    project = workspace / 'openjeff-diffusion'
    recovering = (project / 'hybrid-exit-status.txt').exists()
    deadline = time.time() + (600 if recovering else 10800)
    state.write_text(json.dumps({'deadline': deadline, 'recovering': recovering, 'bundle_sha256': expected}))
    log('supervisor_armed', deadline=int(deadline), recovering=recovering, pod_id=os.getenv('RUNPOD_POD_ID'))
    if Path('/start.sh').exists():
        subprocess.Popen(['bash', '/start.sh'], start_new_session=True)
    if recovering:
        log('recovery_window', seconds=600)
        time.sleep(600)
        return
    bundle = workspace / 'openjeff-hybrid-v2.tar.gz'
    upload_deadline = time.time() + 300
    while time.time() < upload_deadline:
        if bundle.exists():
            try:
                if unpack(bundle, workspace, expected):
                    break
            except (OSError, tarfile.TarError):
                pass
        time.sleep(5)
    else:
        log('upload_timeout')
        return
    log('verified_bundle_starting')
    env = dict(os.environ, HF_HOME='/tmp/openjeff-model-cache', UV_CACHE_DIR='/tmp/openjeff-uv-cache', PIP_CACHE_DIR='/tmp/openjeff-pip-cache', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1')
    with (project / 'supervisor-work.log').open('w') as output:
        child = subprocess.Popen(['bash', 'scripts/run_hybrid_v2.sh'], cwd=project, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None and time.time() < deadline - 90:
            log('work_running', remaining_seconds=int(deadline-time.time()))
            time.sleep(30)
        if child.poll() is None:
            log('work_deadline')
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        log('work_finished', returncode=child.poll())
    # Ordinary volume survives provider stop. Export remains downloadable on restart.
    time.sleep(min(60, max(0, deadline-time.time())))


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1])
    except Exception as error:
        log('supervisor_failure', error_type=type(error).__name__)
    finally:
        stop_pod()
