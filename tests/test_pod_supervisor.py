import contextlib,hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.pod_supervisor_v2 import stop_pod,unpack

class SupervisorTests(unittest.TestCase):
    def test_confirmed_stop_scopes_to_own_pod_and_redacts_credentials(self):
        calls=[]
        def opener(req,timeout):
            calls.append(req)
            return io.BytesIO(b'{"data":{"podStop":{"id":"ownpod123","desiredStatus":"EXITED"}}}')
        with patch.dict('os.environ',{'RUNPOD_POD_ID':'ownpod123','RUNPOD_API_KEY':'synthetic-secret'}),patch('urllib.request.urlopen',opener),patch('time.sleep'),contextlib.redirect_stdout(io.StringIO()) as output:
            stop_pod()
        self.assertEqual(json.loads(calls[0].data)['variables'],{'podId':'ownpod123'})
        self.assertIn('EXITED',output.getvalue());self.assertNotIn('synthetic-secret',output.getvalue())
    def test_hash_and_archive_path_checks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);bundle=root/'input.tgz'
            with tarfile.open(bundle,'w:gz') as archive:
                item=tarfile.TarInfo('openjeff-diffusion/../../escape');item.size=1;archive.addfile(item,io.BytesIO(b'x'))
            self.assertFalse(unpack(bundle,root,'0'*64))
            with self.assertRaises(ValueError):unpack(bundle,root,hashlib.sha256(bundle.read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
