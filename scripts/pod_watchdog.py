"""Stop billing for this disposable Runpod probe at an explicit UTC deadline.

Run detached on the Pod. Uses only that Pod's existing environment credentials.
No credentials or API response bodies are printed. Provider credit exhaustion is
the separate fallback if this process or its host fails.
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request


def settings(deadline, environ, now):
    pod_id = environ.get("RUNPOD_POD_ID", "")
    key = environ.get("RUNPOD_API_KEY", "")
    if not re.fullmatch(r"[a-z0-9]{8,32}", pod_id) or not key:
        raise ValueError("Existing Pod-scoped credentials are required")
    if not 0 < deadline - now <= 5400:
        raise ValueError("Deadline must be in the next 90 minutes")
    return "https://rest.runpod.io/v1/pods/" + pod_id, key, pod_id


def call(url, key, method):
    request = urllib.request.Request(url, method=method,
                                     headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-unix", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    url, key, pod_id = settings(args.deadline_unix, os.environ, time.time())
    record = json.loads(call(url, key, "GET"))
    if record.get("id") != pod_id:
        raise RuntimeError("Provider did not confirm this Pod's identity")
    print(json.dumps({"event": "watchdog_checked" if args.check_only else "watchdog_armed",
                      "pod_id": pod_id, "deadline_unix": args.deadline_unix}), flush=True)
    if args.check_only:
        return
    while time.time() < args.deadline_unix:
        time.sleep(max(0, min(5, args.deadline_unix - time.time())))
    for attempt in range(40):
        try:
            call(url, key, "DELETE")
            print('{"event":"termination_request_accepted"}', flush=True)
            return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                print('{"event":"pod_already_absent"}', flush=True)
                return
            print(json.dumps({"event": "termination_retry", "http_status": error.code,
                              "attempt": attempt + 1}), flush=True)
        except (urllib.error.URLError, TimeoutError, OSError):
            print(json.dumps({"event": "termination_retry", "attempt": attempt + 1}), flush=True)
        time.sleep(15)
    raise RuntimeError("Termination not confirmed; explicit cleanup required")


if __name__ == "__main__":
    main()
