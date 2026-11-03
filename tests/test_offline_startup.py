"""Regression coverage for keyless offline startup, without online model imports."""
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['FORCE_OFFLINE_MODE'] = 'true'
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
for name in ['ZHIPU_API_KEY', 'MIMO_API_KEY', 'TAVILY_API_KEY']:
    os.environ.pop(name, None)

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class OfflineStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_connect = socket.socket.connect

        def connect_local_only(sock, address):
            # Windows asyncio creates a loopback socket pair for its event loop.
            if isinstance(address, tuple) and address[0] not in {'127.0.0.1', '::1', 'localhost'}:
                raise AssertionError('Unexpected non-local connection in offline tests')
            return original_connect(sock, address)

        cls.network_guard = patch.object(
            socket.socket, 'connect', new=connect_local_only
        )
        cls.network_guard.start()
        cls.application = importlib.import_module('app')
        cls.client = TestClient(cls.application.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.network_guard.stop()

    def test_health_reports_offline_mode(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['mode'], 'offline')

    def test_home_identifies_the_demo(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('offline-demo-banner', response.text)

    def test_chat_uses_local_responder_even_when_an_online_agent_is_selected(self):
        response = self.client.post('/chat', json={
            'query': 'Hello, this is a project demonstration.', 'language': 'en', 'preferred_agent': 'RAG_AGENT'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['agent'], 'OFFLINE_FALLBACK_AGENT')
        self.assertTrue(response.json()['response'])

    def test_upload_rejects_unsupported_extension(self):
        response = self.client.post('/upload', files={'image': ('sample.exe', b'not an image')})
        self.assertEqual(response.status_code, 400)

    def test_upload_stays_on_the_local_responder(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.application, 'UPLOAD_FOLDER', directory):
                response = self.client.post('/upload', files={'image': ('sample.png', b'demo input')},
                                            data={'text': 'Project demonstration', 'language': 'en'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['agent'], 'OFFLINE_FALLBACK_AGENT')

    def test_online_only_features_are_disabled(self):
        requests = [
            ('/api/brain_stroke/segment', {'files': {'image': ('sample.png', b'demo')}}),
            ('/validate', {'data': {'validation_result': 'yes'}}),
            ('/transcribe', {'files': {'audio': ('sample.wav', b'demo')}}),
            ('/generate-speech', {'json': {'text': 'demo', 'language': 'en'}}),
        ]
        for endpoint, kwargs in requests:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.client.post(endpoint, **kwargs).status_code, 503)

    def test_online_graph_and_clients_were_not_imported(self):
        self.assertNotIn('agents.agent_decision', sys.modules)
        self.assertNotIn('langchain_openai', sys.modules)
        self.assertFalse(any(name.startswith('langgraph') for name in sys.modules))

    def test_online_mode_still_requires_credentials(self):
        environment = dict(os.environ, FORCE_OFFLINE_MODE='false', PYTHON_DOTENV_DISABLED='1')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', 'import config'], cwd=ROOT,
                                env=environment, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing required environment variable: ZHIPU_API_KEY', result.stderr)


if __name__ == '__main__':
    unittest.main()
