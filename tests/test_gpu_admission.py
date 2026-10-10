"""Real disposable Redis self-check; GPU_ADMISSION_TEST_REDIS_URL must be explicit."""
import asyncio
import os
import sys
import uuid
from pathlib import Path

from redis.asyncio import Redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'LiteLLM'))
from gpu_admission import GPUAdmission, AdmissionError, AdmissionQueueFull, AdmissionTimeout


async def main():
    client = Redis.from_url(os.environ['GPU_ADMISSION_TEST_REDIS_URL'])
    client2 = Redis.from_url(os.environ['GPU_ADMISSION_TEST_REDIS_URL'])
    prefix = 'test:gpu-admission:' + uuid.uuid4().hex
    gate = GPUAdmission(client, key_prefix=prefix, lease_ttl=.6, wait_timeout=2)
    other = GPUAdmission(client2, key_prefix=prefix, lease_ttl=.6, wait_timeout=2)
    try:
        active = await gate.acquire(50)
        order = []
        async def work(g, priority, label):
            lease = await g.acquire(priority)
            order.append(label)
            await asyncio.sleep(.02)
            await lease.release()
        tasks = []
        for g, p, label in [(gate,80,'low'),(other,0,'first'),(gate,0,'second')]:
            tasks.append(asyncio.create_task(work(g,p,label)))
            await asyncio.sleep(.03)
        await active.release()
        await asyncio.gather(*tasks)
        assert order == ['first','second','low'], order
        await active.release()  # idempotent

        active = await gate.acquire(0)
        waiting = asyncio.create_task(other.acquire(10))
        await asyncio.sleep(.05)
        waiting.cancel()
        try: await waiting
        except asyncio.CancelledError: pass
        assert await client.zcard(gate.keys[0]) == 0
        full = GPUAdmission(client, key_prefix=prefix, max_waiters=1, wait_timeout=.15, lease_ttl=.6)
        waiting = asyncio.create_task(full.acquire(10))
        await asyncio.sleep(.04)
        try: await full.acquire(20)
        except AdmissionQueueFull: pass
        else: raise AssertionError('queue cap not enforced')
        try: await waiting
        except AdmissionTimeout: pass
        else: raise AssertionError('timeout not enforced')
        assert await client.zcard(gate.keys[0]) == 0
        await active.release()

        # A crashed waiter's expired priority ticket cannot block the next caller.
        await gate._call('enqueue', 'crashed-waiter', 0, 30, 128)
        await asyncio.sleep(.05)
        lease = await other.acquire(50)
        assert await client.hlen(gate.keys[2]) == 0
        # A crashed owner stops renewing; its lease eventually expires.
        lease._heartbeat.cancel()
        try: await lease._heartbeat
        except asyncio.CancelledError: pass
        successor = await gate.acquire(0)
        await lease.release()  # stale owner must not delete successor
        assert await client.get(gate.keys[4]) == successor.ticket.encode()
        await successor.release()

        lease = await gate.acquire(0)
        await client.set(gate.keys[4], 'different-owner', px=600)
        await asyncio.wait_for(lease.lost.wait(), .5)
        await lease.release()
        assert await client.get(gate.keys[4]) == b'different-owner'
        await client.delete(gate.keys[4])

        # Renewal errors fail closed, and storage errors reveal no endpoint details.
        class BrokenRedis:
            async def eval(self, *args):
                raise RuntimeError('credential and endpoint must not leak')
        broken = GPUAdmission(BrokenRedis(), lease_ttl=.6)
        try: await broken.acquire(0)
        except AdmissionError as exc:
            assert str(exc) == 'GPU admission storage unavailable'
            assert exc.__cause__ is None
        else: raise AssertionError('storage failure accepted')
        lease = await gate.acquire(0)
        saved = gate.redis
        gate.redis = BrokenRedis()
        await asyncio.wait_for(lease.lost.wait(), .5)
        gate.redis = saved
        await lease.release()

        for invalid in (-1,101,True,'0'):
            try: await gate.acquire(invalid)
            except ValueError: pass
            else: raise AssertionError('invalid priority accepted')
        print('GPU admission real-Redis checks passed')
    finally:
        await client.delete(*gate.keys)
        await client.aclose()
        await client2.aclose()


asyncio.run(main())
