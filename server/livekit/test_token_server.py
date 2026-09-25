"""Policy and HTTP checks for the call token server. No video, no real LiveKit project."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request

from token_server import TokenHandler, normalize_room
from username_policy import validate_username

from http.server import ThreadingHTTPServer


class UsernamePolicyTests(unittest.TestCase):
    def test_accepts_display_name(self):
        self.assertEqual(validate_username("Alex"), ("ok", "Alex"))

    def test_rejects_short_and_empty(self):
        self.assertEqual(validate_username("Al")[0], "rejected")
        self.assertEqual(validate_username("   ")[0], "rejected")

    def test_rejects_phone_like_and_reserved(self):
        self.assertIn("phone", validate_username("abc1234567")[1])
        self.assertEqual(validate_username("admin")[1], "That name is reserved.")

    def test_rejects_leet_and_underscored_blocklist(self):
        self.assertEqual(validate_username("n1gg3r")[0], "rejected")
        self.assertEqual(validate_username("f_u_c_k")[0], "rejected")

    def test_does_not_treat_a_normal_name_as_a_phone(self):
        status, cleaned = validate_username("alex_1")
        self.assertEqual(status, "ok")
        self.assertEqual(cleaned, "alex_1")


class RoomCodeTests(unittest.TestCase):
    def test_room_code(self):
        self.assertEqual(normalize_room("AB12cd34"), "ab12cd34")
        self.assertIsNone(normalize_room("ab"))
        self.assertIsNone(normalize_room("room-1"))
        self.assertIsNone(normalize_room(None))


class TokenHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), TokenHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=3)

    def post(self, payload: dict) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/token",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_rejects_invalid_username_without_minting(self):
        status, body = self.post({"room": "ab12cd34", "username": "1234567890"})
        self.assertEqual(status, 400)
        self.assertIn("letter", body["error"])
        self.assertNotIn("token", body)

    def test_rejects_bad_room(self):
        status, body = self.post({"room": "no", "username": "Alex"})
        self.assertEqual(status, 400)
        self.assertNotIn("token", body)

    def test_mints_room_scoped_token_when_configured(self):
        try:
            import livekit.api  # noqa: F401
        except ImportError:
            self.skipTest("livekit-api is not installed")

        keys = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
        previous = {key: os.environ.get(key) for key in keys}
        os.environ["LIVEKIT_URL"] = "wss://example.livekit.cloud"
        os.environ["LIVEKIT_API_KEY"] = "devkey"
        os.environ["LIVEKIT_API_SECRET"] = "secretsecretsecretsecretsecret12"
        try:
            status, body = self.post({"room": "AB12CD34", "username": "Alex"})
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(status, 200, body)
        self.assertEqual(body["room"], "ab12cd34")
        self.assertEqual(body["identity"], "Alex")
        self.assertEqual(body["url"], "wss://example.livekit.cloud")
        self.assertEqual(len(body["token"].split(".")), 3)


if __name__ == "__main__":
    unittest.main()
