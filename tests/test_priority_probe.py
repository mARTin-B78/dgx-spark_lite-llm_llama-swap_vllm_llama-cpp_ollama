"""Local deterministic probe checks; no model or credentials required."""
import importlib.util
from pathlib import Path
import threading
import time
import os
import subprocess
import json
from unittest.mock import patch
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'scripts/test_priority.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Handler(BaseHTTPRequestHandler):
    attempts = 0
    posts = 0
    mode = "valid"
    def do_GET(self):
        key = self.headers.get("Authorization", "").removeprefix("Bearer ")
        tier = int(key[-1]) if key[-1:].isdigit() else 0
        if Handler.mode == "stall":
            self.wfile.write(b"HTTP/1.0 200 OK\r\nX-Slow: ")
            self.wfile.flush()
            for _ in range(150):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(.02)
            return
        if Handler.mode == "denied":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(key.encode())
            return
        self.send_response(200)
        self.end_headers()
        if self.path == "/model/info":
            data = {"data": [{"model_name": "mock", "litellm_params": {"api_base": "http://llama-swap:8080"}}]}
        else:
            data = {"key": key, "info": {"metadata": {"priority": 1 if Handler.mode == "mismatch" else 100-20*tier}, "blocked": True if Handler.mode == "blocked" else False, "max_parallel_requests": 1 if Handler.mode == "cap" else 2}}
        self.wfile.write(json.dumps(data).encode())
    def log_message(self, *args):
        pass
    def do_POST(self):
        Handler.posts += 1
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/ok')
            self.end_headers()
            return
        if self.path == '/retry':
            Handler.attempts += 1
            if Handler.attempts == 1:
                self.send_response(429)
                self.end_headers()
                return
        if self.headers.get('Authorization') == 'Bearer stall-headers':
            self.wfile.write(b'HTTP/1.0 200 OK\r\nX-Slow: ')
            self.wfile.flush()
            for _ in range(150):
                try:
                    self.wfile.write(b'x')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(.02)
            return
        self.send_response(200)
        if Handler.mode != 'unknown_backend':
            self.send_header('x-litellm-model-api-base', 'http://llama-swap:8080')
        self.end_headers()
        if self.path == '/slow':
            for _ in range(20):
                try:
                    self.wfile.write(b':')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(.05)
        elif self.path == '/bad':
            self.wfile.write(b'data: invalid\n\n')
        else:
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n')
            self.wfile.flush()
            if self.path == '/v1/chat/completions' and body.get('max_tokens') == 256 and Handler.mode != 'no_overlap':
                time.sleep(.15)
            if self.path != '/truncated':
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')


def check_preflight_diagnostics():
    secret = 'secret-marker-key-alias-http://private.example/token'
    class Response:
        status = 200
        def __init__(self, payload):
            self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, limit):
            return self.raw[:limit]
    def call(payload, tier='P0', route=False, failure=None):
        with patch.object(probe.urllib.request.OpenerDirector, 'open', return_value=Response(payload), side_effect=failure):
            result = (probe.model_route('http://localhost', secret, 'mock', 'llama-swap', 1)
                      if route else probe.key_info('http://localhost', secret, tier, 1))
        assert secret not in json.dumps(result), result
        return result
    valid = {'info': {'metadata': {'priority': 100}, 'blocked': None, 'max_parallel_requests': None}}
    assert call(valid)['valid']
    with patch.object(probe.urllib.request.OpenerDirector, 'open', return_value=Response({'key': [secret], 'info': [valid['info']]})) as opened:
        admin = probe.key_info('http://localhost', secret, 'P0', 1, admin_key='admin-' + secret)
    assert admin['valid'] and admin['status'] == 200, admin
    sent = opened.call_args[0][0]
    assert sent.full_url.endswith('/v2/key/info') and secret not in sent.full_url
    assert sent.headers['Authorization'] == 'Bearer admin-' + secret
    with patch.object(probe.urllib.request.OpenerDirector, 'open', return_value=Response({'key': [secret], 'info': []})):
        assert probe.key_info('http://localhost', secret, 'P0', 1, admin_key='x')['error'] == 'key_not_found'
    for code in (401, 403, 404, 429, 500):
        for route in (False, True):
            result = call({}, route=route, failure=urllib.error.HTTPError(secret, code, secret, {}, None))
            expected = 'admin_key_required' if code == 403 and not route else 'http_error'
            assert result['status'] == code and result['error'] == expected, result
    for route in (False, True):
        result = call({}, route=route, failure=urllib.error.URLError(secret))
        assert result['status'] == 0 and result['error'] == 'transport_error', result
        assert call(secret.encode(), route=route)['error'] == 'invalid_json'
        assert call(None, route=route)['error'] == 'invalid_schema'
    cases = [
        ({}, 'missing_info'),
        ({'info': {}}, 'missing_metadata'),
        ({'info': {'metadata': secret}}, 'invalid_metadata'),
        ({'info': {'metadata': {}}}, 'missing_priority'),
    ]
    for field, value, error in (
        ('priority', secret, 'invalid_priority'), ('priority', True, 'invalid_priority'),
        ('priority', 20, 'priority_mismatch'), ('blocked', True, 'key_blocked'),
        ('blocked', secret, 'invalid_blocked'), ('blocked', 0, 'invalid_blocked'),
        ('max_parallel_requests', secret, 'invalid_max_parallel_requests'),
        ('max_parallel_requests', 0, 'invalid_max_parallel_requests'),
        ('max_parallel_requests', True, 'invalid_max_parallel_requests'),
    ):
        payload = json.loads(json.dumps(valid))
        target = payload['info']['metadata'] if field == 'priority' else payload['info']
        target[field] = value
        cases.append((payload, error))
    for field in ('blocked', 'max_parallel_requests'):
        payload = json.loads(json.dumps(valid))
        del payload['info'][field]
        cases.append((payload, 'missing_' + field))
    for payload, error in cases:
        result = call(payload)
        assert result['status'] == 200 and result['error'] == error and not result['valid'], result
    valid['info']['metadata']['priority'] = 20
    valid['info']['max_parallel_requests'] = 1
    assert call(valid, 'P4')['error'] == 'insufficient_max_parallel_requests'
    valid['info']['max_parallel_requests'] = 2
    assert call(valid, 'P4')['valid']
    deployment = {'model_name': 'mock', 'litellm_params': {'api_base': 'http://llama-swap:8080'}}
    assert call({'data': [deployment]}, route=True)['valid']
    hidden = call({'data': [{'model_name': 'mock', 'litellm_params': {}}]}, route=True)
    assert hidden['valid'] and hidden['host_checked'] is False, hidden
    for payload, error in (
        ({}, 'missing_data'), ({'data': {}}, 'invalid_schema'),
        ({'data': []}, 'model_not_found'),
        ({'data': [{'model_name': 'mock', 'litellm_params': {'api_base': 'invalid'}}]}, 'invalid_api_base'), ({'data': [None]}, 'invalid_schema'),
        ({'data': [{'model_name': 'mock', 'litellm_params': {'api_base': secret}}]}, 'backend_mismatch'),
        ({'data': [{'model_name': 'mock', 'litellm_params': {'api_base': 'http://other'}}]}, 'backend_mismatch'),
    ):
        result = call(payload, route=True)
        assert result['status'] == 200 and result['error'] == error and not result['valid'], result


def main():
    check_preflight_diagnostics()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def run(path, retries=0):
            return probe.request(f'http://127.0.0.1:{server.server_port}{path}', 'dummy', 'mock', 'P0', 'baseline', 2, 2, retries)
        ok = run('/ok')
        assert ok['status'] == 200 and ok['done'] and ok['finish'] and ok['ttft_s'] is not None and ok['error'] is None, ok
        assert run('/redirect')['status'] == 302
        assert run('/bad')['error'] == 'invalid_stream'
        assert run('/truncated')['error'] == 'incomplete_stream'
        denied = run('/retry')
        assert denied['status'] == 429 and denied['attempts'] == 1
        Handler.attempts = 0
        started = time.monotonic()
        slow = probe.request(f'http://127.0.0.1:{server.server_port}/slow', 'dummy', 'mock', 'P0', 'baseline', 2, .15, 0)
        assert slow['error'] and time.monotonic() - started < .7, slow
        retried = run('/retry', 1)
        assert retried['status'] == 200 and retried['attempts'] == 2 and retried['error'] is None, retried
        env = {**os.environ, 'MODEL': 'mock', 'LITELLM_URL': f'http://127.0.0.1:{server.server_port}'}
        env.update({f'P{i}_KEY': f'dummy-secret-marker{i}' for i in range(5)})
        command = [str(Path(__file__).parents[1] / 'scripts/test-priority.sh'), '--rounds', '2', '--load-delay', '0', '--timeout', '2']
        def subprocess_probe(extra=()):
            result = subprocess.run(command + list(extra), env=env, capture_output=True, text=True, timeout=15)
            assert all(value not in result.stdout + result.stderr for name, value in env.items() if name.endswith('_KEY'))
            return result
        completed = subprocess_probe()
        assert completed.returncode == 0, completed.stdout + completed.stderr
        records = [json.loads(line) for line in completed.stdout.splitlines()]
        assert len([r for r in records if r.get('phase') == 'load']) == 2
        assert all(r['done'] for r in records if 'done' in r)
        assert all(r['load_overlap'] for r in records if 'load_overlap' in r)
        assert len([r for r in records if r.get('phase') == 'under_load' and 'tier' in r]) == 10
        assert subprocess_probe(['--p0-max-ttft', '.000001']).returncode == 1
        for mode in ('unknown_backend', 'no_overlap'):
            Handler.mode = mode
            assert subprocess_probe().returncode == 2
        for mode in ('mismatch', 'denied', 'cap', 'blocked'):
            Handler.mode = mode
            posts = Handler.posts
            failed = subprocess_probe()
            assert failed.returncode == 1 and Handler.posts == posts
        Handler.mode = 'valid'
        posts = Handler.posts
        assert subprocess_probe(['--expected-backend-host', 'unmatched']).returncode == 1
        assert Handler.posts == posts
        Handler.mode = 'stall'
        stalled = subprocess_probe(['--rounds', '1', '--timeout', '.05'])
        assert stalled.returncode == 1 and 'run_deadline_exceeded' in stalled.stdout
        Handler.mode = 'valid'
        env['P1_KEY'] = env['P0_KEY']
        assert subprocess_probe().returncode == 1 and Handler.posts == posts
    finally:
        server.shutdown()
        server.server_close()
    print('priority probe: local stream, preflight, repetition, overlap and SLA checks passed')


if __name__ == '__main__':
    main()
