#!/usr/bin/env python3
"""Measure five key tiers without changing keys or supplying body priority."""
import argparse
import concurrent.futures
import json
import os
import socket
import time
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import math
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, key, model, tier, phase, tokens, timeout, retries, ready=None):
    started = time.monotonic()
    deadline = started + timeout
    result = dict(phase=phase, tier=tier, status=0, attempts=0, ttft_s=None,
                  elapsed_s=None, backend_host=None, started_at=started, ended_at=None, first_token_at=None, max_gap_s=0, chunks=0, done=False, finish=False, error=None)
    body = json.dumps(dict(model=model, stream=True, max_tokens=tokens,
                           messages=[dict(role='user', content='Write a story about a mountain. Request ' + uuid.uuid4().hex)])).encode()
    watchdog = None
    previous_token = None
    for attempt in range(retries + 1):
        result['attempts'] += 1
        try:
            req = urllib.request.Request(url, data=body, headers={
                'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urllib.request.build_opener(NoRedirect).open(req, timeout=max(.001, deadline - time.monotonic())) as response:
                result['status'] = response.status
                backend = response.headers.get('x-litellm-model-api-base')
                result['backend_host'] = urlsplit(backend).hostname if backend else None
                sock = response.fp.raw._sock
                def expire():
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                watchdog = threading.Timer(max(.001, deadline - time.monotonic()), expire)
                watchdog.daemon = True
                watchdog.start()
                event = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    # Bound each read by the remaining overall request budget.
                    response.fp.raw._sock.settimeout(remaining)
                    line = response.readline(1024 * 1024)
                    if len(line) >= 1024 * 1024:
                        raise ValueError('oversized_event')
                    if not line:
                        break
                    line = line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):
                        event.append(line[5:].lstrip())
                    elif not line and event:
                        data = '\n'.join(event)
                        event = []
                        if data == '[DONE]':
                            result['done'] = True
                            break
                        chunk = json.loads(data)
                        if 'error' in chunk:
                            raise ValueError('stream_error')
                        choices = chunk.get('choices')
                        if not isinstance(choices, list):
                            raise ValueError('invalid_choices')
                        for choice in choices:
                            delta = choice.get('delta', {})
                            if delta.get('content') or delta.get('reasoning_content'):
                                now = time.monotonic()
                                if previous_token is not None:
                                    result['max_gap_s'] = max(result['max_gap_s'], round(now - previous_token, 4))
                                previous_token = now
                                result['chunks'] += 1
                                if result['ttft_s'] is None:
                                    result['first_token_at'] = time.monotonic()
                                    result['ttft_s'] = round(result['first_token_at'] - started, 4)
                                    if ready is not None:
                                        ready.set()
                            if choice.get('finish_reason') is not None:
                                result['finish'] = True
                if not (result['done'] and result['finish'] and result['chunks']):
                    result['error'] = 'incomplete_stream'
            break
        except urllib.error.HTTPError as exc:
            result['status'] = exc.code
            exc.close()
            result['error'] = 'http_error'
            delay = .25 * (2 ** attempt)
            if exc.code != 429 or attempt == retries or time.monotonic() + delay >= deadline:
                break
            time.sleep(delay)
            result['error'] = None
        except (ValueError, UnicodeError, KeyError, TypeError):
            result['error'] = 'invalid_stream'
            break
        except Exception:
            # Never echo server bodies, headers, URLs or exceptions: they may contain secrets.
            result['error'] = 'transport_error'
            break
        finally:
            if watchdog is not None:
                watchdog.cancel()
    result['ended_at'] = time.monotonic()
    result['elapsed_s'] = round(result['ended_at'] - started, 4)
    return result


def key_info(base, key, tier, timeout):
    expected = 100 - 20 * int(tier[1])
    result = dict(preflight=tier, expected_priority=expected, actual_priority=None,
                  max_parallel_requests=None, valid=False)
    try:
        req = urllib.request.Request(base + '/key/info', headers={'Authorization': 'Bearer ' + key})
        with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError()
            info = json.loads(raw)['info']
        priority = info['metadata']['priority']
        cap = info['max_parallel_requests']
        blocked = info['blocked']
        result['actual_priority'] = priority if type(priority) is int else None
        result['max_parallel_requests'] = cap if type(cap) is int else None
        result['valid'] = (type(priority) is int and priority == expected
                           and (blocked is False or blocked is None)
                           and (cap is None or type(cap) is int and cap > 0)
                           and (tier != 'P4' or cap is None or cap >= 2))
    except Exception:
        pass  # Server text and exception details must never reach output.
    return result


def model_route(base, key, model, expected_host, timeout):
    result = dict(route_preflight=True, expected_backend_host=expected_host, valid=False)
    try:
        req = urllib.request.Request(base + '/model/info', headers={'Authorization': 'Bearer ' + key})
        with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError()
            deployments = json.loads(raw)['data']
        matches = [item for item in deployments if item['model_name'] == model]
        hosts = [urlsplit(item['litellm_params']['api_base']).hostname for item in matches]
        result['valid'] = bool(hosts) and all(host == expected_host for host in hosts)
    except Exception:
        pass
    return result


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__ + ' Required env: MODEL, P0_KEY through P4_KEY; optional LITELLM_URL. JSON lines, no credentials or response text.')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=0, help='429 retries only (0..3)')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--load-tokens', type=int, default=256)
    parser.add_argument('--load-delay', type=float, default=.5)
    parser.add_argument('--expected-backend-host', default='llama-swap', help='required host of every matching deployment')
    parser.add_argument('--rounds', type=int, default=3, help='measurement rounds (1..10)')
    parser.add_argument('--p0-max-ttft', type=float, help='optional loaded P0 TTFT SLA in seconds')
    args = parser.parse_args()
    if not (0 < args.timeout <= 600 and 0 <= args.retries <= 3 and 1 <= args.tokens <= 4096 and 1 <= args.load_tokens <= 4096 and 0 <= args.load_delay <= 10 and 1 <= args.rounds <= 10 and (args.p0_max_ttft is None or 0 < args.p0_max_ttft <= 600)):
        parser.error('settings outside bounded limits')
    tiers = [f'P{i}' for i in range(5)]
    required = ['MODEL'] + [tier + '_KEY' for tier in tiers]
    if any(not os.environ.get(name) for name in required):
        parser.error('required environment: ' + ', '.join(required))
    keys = [os.environ[tier + '_KEY'] for tier in tiers]
    if len(set(keys)) != 5:
        print(json.dumps(dict(error='duplicate_tier_keys')))
        return 1
    base = os.environ.get('LITELLM_URL', 'http://localhost:4000').rstrip('/')
    try:
        parsed = urlsplit(base)
        valid_url = parsed.scheme in ('http', 'https') and parsed.hostname and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        parsed.port
    except ValueError:
        valid_url = False
    if not valid_url:
        parser.error('LITELLM_URL must be an HTTP(S) URL without credentials, query or fragment')
    run_budget = (7 + 7 * args.rounds) * args.timeout + args.rounds * args.load_delay + 1
    def expire_run():
        print(json.dumps(dict(error='run_deadline_exceeded', budget_s=run_budget)), flush=True)
        os._exit(1)
    watchdog = threading.Timer(run_budget, expire_run)
    watchdog.daemon = True
    watchdog.start()
    def emit(record):
        print(json.dumps(record), flush=True)
    preflight = [key_info(base, key, tier, args.timeout) for key, tier in zip(keys, tiers)]
    for record in preflight:
        emit(record)
    if not all(record['valid'] for record in preflight):
        watchdog.cancel()
        emit(dict(error='key_preflight_failed'))
        return 1
    route = model_route(base, keys[0], os.environ['MODEL'], args.expected_backend_host, args.timeout)
    emit(route)
    if not route['valid']:
        watchdog.cancel()
        emit(dict(error='model_route_preflight_failed'))
        return 1
    def run(tier, phase, tokens, ready=None):
        result = request(base + '/v1/chat/completions', os.environ[tier + '_KEY'], os.environ['MODEL'], tier, phase, tokens, args.timeout, args.retries, ready)
        result['backend_verified'] = result.pop('backend_host') == args.expected_backend_host
        return result
    warmup = run('P0', 'warmup', args.tokens)
    emit(warmup)
    if warmup['error'] or warmup['status'] != 200:
        watchdog.cancel()
        return 1
    if not warmup['backend_verified']:
        watchdog.cancel()
        emit(dict(error='warmup_backend_unverified', measurement='INCONCLUSIVE'))
        return 2
    results = []
    overlaps = []
    for number in range(1, args.rounds + 1):
        current = []
        for tier in tiers:
            current.append(run(tier, 'baseline', args.tokens))
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            ready = threading.Event()
            blocker = pool.submit(run, 'P4', 'load', args.load_tokens, ready)
            ready.wait(args.timeout)
            time.sleep(args.load_delay)
            probes = [pool.submit(run, tier, 'under_load', args.tokens) for tier in reversed(tiers)]
            current.extend(future.result() for future in [blocker] + probes)
        load = next(r for r in current if r['phase'] == 'load')
        loaded = [r for r in current if r['phase'] == 'under_load']
        overlap = all(load['first_token_at'] is not None and load['first_token_at'] <= r['started_at'] < load['ended_at'] for r in loaded)
        overlaps.append(overlap)
        for result in current:
            result['round'] = number
            emit(result)
        emit(dict(round=number, load_overlap=overlap,
                  observed_first_token_order=[r['tier'] for r in sorted(loaded, key=lambda r: r['first_token_at'] or float('inf'))]))
        results.extend(current)
    route_verified = all(r['backend_verified'] for r in results + [warmup])
    failures = any(r['error'] or r['status'] != 200 for r in results)
    retried = any(r['attempts'] > 1 for r in results + [warmup])
    inconclusive = failures or retried or not all(overlaps) or not route_verified
    for tier in tiers:
        for phase in ('baseline', 'under_load'):
            samples = [r for r in results if r['tier'] == tier and r['phase'] == phase and not r['error'] and r['status'] == 200]
            summary = dict(summary=tier, phase=phase, samples=len(samples))
            for field in ('ttft_s', 'elapsed_s', 'max_gap_s'):
                values = [r[field] for r in samples]
                summary[field + '_p50'] = percentile(values, .5) if values else None
                summary[field + '_p95'] = percentile(values, .95) if values else None
            emit(summary)
    sla = 'NOT_CONFIGURED'
    if args.p0_max_ttft is not None:
        sla = 'INCONCLUSIVE' if inconclusive else ('PASS' if all(r['ttft_s'] <= args.p0_max_ttft for r in results if r['tier'] == 'P0' and r['phase'] == 'under_load') else 'FAIL')
    emit(dict(actual_backend_verified=route_verified, measurement='INCONCLUSIVE' if inconclusive else 'COMPLETE', p0_sla=sla,
              p0_max_ttft_s=args.p0_max_ttft, retries_change_admission=retried,
              note='Timing observations do not prove queue order, preemption, or actual GPU execution. Route metadata identifies configured deployment only; timings are end-to-end client measurements.'))
    watchdog.cancel()
    return 1 if failures or sla == 'FAIL' else (2 if inconclusive else 0)


if __name__ == '__main__':
    raise SystemExit(main())
