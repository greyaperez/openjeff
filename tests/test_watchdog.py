import unittest
from scripts.pod_watchdog import settings


class WatchdogTests(unittest.TestCase):
    def test_uses_only_current_pod_and_fixed_provider(self):
        url, _, pod_id = settings(110, {"RUNPOD_POD_ID": "abcdefgh1234",
                                       "RUNPOD_API_KEY": "test-only-key"}, 100)
        self.assertEqual(url, "https://rest.runpod.io/v1/pods/abcdefgh1234")
        self.assertEqual(pod_id, "abcdefgh1234")

    def test_rejects_missing_auth_unsafe_id_and_unbounded_deadline(self):
        valid = {"RUNPOD_POD_ID": "abcdefgh1234", "RUNPOD_API_KEY": "test-only-key"}
        for env in ({}, {**valid, "RUNPOD_POD_ID": "../other"}, {**valid, "RUNPOD_API_KEY": ""}):
            with self.assertRaises(ValueError):
                settings(110, env, 100)
        for deadline in (99, 100, 5501):
            with self.assertRaises(ValueError):
                settings(deadline, valid, 100)
