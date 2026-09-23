import contextlib,io,json,unittest
from unittest.mock import patch
from scripts.pod_start_v2 import arm
class PodStartV2Tests(unittest.TestCase):
    def test_startup_does_not_call_cli_and_preserves_watchdog(self):
        calls=[]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            arm(120,environ={'RUNPOD_POD_ID':'abcdefgh1234','RUNPOD_API_KEY':'synthetic-test-only'},clock=lambda:100,fork=lambda:123,execv=lambda *a:calls.append(a))
        self.assertEqual(calls,[('/start.sh',['/start.sh'])]);self.assertNotIn('synthetic-test-only',output.getvalue())
    def test_deadline_calls_only_own_pod_and_does_not_leak_key(self):
        now=[100];requests=[]
        def sleep(n):now[0]+=n
        def opener(req,timeout):
            requests.append(json.loads(req.data));return io.BytesIO(b'{"data":{"podStop":{"id":"abcdefgh1234","desiredStatus":"EXITED"}}}')
        with patch('scripts.pod_start_v2.os.setsid'),contextlib.redirect_stdout(io.StringIO()) as output:
            arm(110,environ={'RUNPOD_POD_ID':'abcdefgh1234','RUNPOD_API_KEY':'synthetic-test-only'},clock=lambda:now[0],fork=lambda:0,sleep=sleep,opener=opener)
        self.assertEqual(now[0],110);self.assertEqual(requests[0]['variables'],{'podId':'abcdefgh1234'});self.assertIn('own_pod_stop_confirmed',output.getvalue());self.assertNotIn('synthetic-test-only',output.getvalue())
    def test_rejects_unbounded_deadline_and_missing_credentials(self):
        for env,deadline in [({},110),({'RUNPOD_POD_ID':'abcdefgh1234','RUNPOD_API_KEY':'test'},12000)]:
            with self.assertRaises(ValueError):arm(deadline,environ=env,clock=lambda:100,fork=lambda:self.fail('must fail before fork'))
if __name__=='__main__':unittest.main()
