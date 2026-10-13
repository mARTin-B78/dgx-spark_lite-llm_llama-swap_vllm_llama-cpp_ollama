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
    tasks = []
    leases = []

    async def queued(priorities):
        deadline = asyncio.get_running_loop().time() + 1
        while True:
            scores = [int(score) for _, score in
                      await asyncio.wait_for(
                          client.zrange(gate.keys[0], 0, -1, withscores=True),
                          timeout=max(.001, deadline - asyncio.get_running_loop().time()))]
            if scores == sorted(priorities):
                return
            assert asyncio.get_running_loop().time() < deadline, (priorities, scores)
            await asyncio.sleep(.005)

    async def acquire(g, priority):
        lease = await g.acquire(priority)
        leases.append(lease)
        return lease

    try:
        # Reverse submission proves all five tiers across independent Redis clients.
        active = await acquire(gate, 50)
        order = []
        async def tier_work(g, priority):
            lease = await acquire(g, priority)
            try:
                order.append(priority)
            finally:
                await lease.release()
        submitted = []
        for index, priority in enumerate((80, 60, 40, 20, 0)):
            tasks.append(asyncio.create_task(tier_work((gate, other)[index % 2], priority)))
            submitted.append(priority)
            await queued(submitted)
        assert await client.get(gate.keys[4]) == active.ticket.encode()
        await active.release()
        await asyncio.gather(*tasks)
        assert order == [0, 20, 40, 60, 80], order

        active = await acquire(gate, 50)
        order = []
        async def work(g, priority, label):
            lease = await acquire(g, priority)
            order.append(label)
            await asyncio.sleep(.02)
            await lease.release()
        fifo_tasks = []
        for g, p, label in [(gate,80,'low'),(other,0,'first'),(gate,0,'second')]:
            task = asyncio.create_task(work(g,p,label))
            tasks.append(task)
            fifo_tasks.append(task)
            await queued([80] + [0] * (len(fifo_tasks) - 1))
        await active.release()
        await asyncio.gather(*fifo_tasks)
        assert order == ['first','second','low'], order
        await active.release()  # idempotent

        active = await acquire(gate, 0)
        waiting = asyncio.create_task(acquire(other, 10))
        tasks.append(waiting)
        await queued([10])
        waiting.cancel()
        try: await waiting
        except asyncio.CancelledError: pass
        assert await client.zcard(gate.keys[0]) == 0
        full = GPUAdmission(client, key_prefix=prefix, max_waiters=1, reserved_p0_waiters=0,
                            wait_timeout=.15, lease_ttl=.6)
        waiting = asyncio.create_task(acquire(full, 10))
        tasks.append(waiting)
        await queued([10])
        try: await full.acquire(20)
        except AdmissionQueueFull: pass
        else: raise AssertionError('queue cap not enforced')
        try: await waiting
        except AdmissionTimeout: pass
        else: raise AssertionError('timeout not enforced')
        assert await client.zcard(gate.keys[0]) == 0
        await active.release()

        # Lower priorities across workers cannot consume the reserved P0 position.
        reserved = GPUAdmission(client, key_prefix=prefix, max_waiters=3,
                                wait_timeout=2, lease_ttl=.6)
        active = await acquire(gate, 40)
        other_reserved = GPUAdmission(client2, key_prefix=prefix, max_waiters=3,
                                wait_timeout=2, lease_ttl=.6)
        low_tasks = []
        for g, priority in ((reserved, 40), (other_reserved, 80)):
            task = asyncio.create_task(acquire(g, priority))
            tasks.append(task)
            low_tasks.append(task)
            await queued([40] if len(low_tasks) == 1 else [40, 80])
        try: await reserved.acquire(20)
        except AdmissionQueueFull: pass
        else: raise AssertionError('P0 reservation consumed by lower priority')
        p0 = asyncio.create_task(acquire(other_reserved, 0))
        tasks.append(p0)
        await queued([0, 40, 80])
        try: await reserved.acquire(0)
        except AdmissionQueueFull: pass
        else: raise AssertionError('P0 total queue cap not enforced')
        await active.release()
        lease = await p0
        await queued([40, 80])
        for task in low_tasks:
            task.cancel()
        await asyncio.gather(*low_tasks, return_exceptions=True)
        await lease.release()

        # A crashed waiter's expired priority ticket cannot block the next caller.
        await gate._call('enqueue', 'crashed-waiter', 0, 30, 128, 1)
        await asyncio.sleep(.05)
        lease = await acquire(other, 50)
        assert await client.hlen(gate.keys[2]) == 0
        # A crashed owner stops renewing; its lease eventually expires.
        lease._heartbeat.cancel()
        try: await lease._heartbeat
        except asyncio.CancelledError: pass
        successor = await acquire(gate, 0)
        await lease.release()  # stale owner must not delete successor
        assert await client.get(gate.keys[4]) == successor.ticket.encode()
        await successor.release()

        lease = await acquire(gate, 0)
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
        lease = await acquire(gate, 0)
        saved = gate.redis
        gate.redis = BrokenRedis()
        try:
            await asyncio.wait_for(lease.lost.wait(), .5)
        finally:
            gate.redis = saved
        await lease.release()

        # Reserved P0 execution slot: P0 starts at once beside a running P4, others still wait.
        slot_prefix = prefix + ':p0slot'
        slots = GPUAdmission(client, key_prefix=slot_prefix, lease_ttl=.6, wait_timeout=2,
                             p0_execution_slots=1)
        p4 = await acquire(slots, 80)
        p0 = await asyncio.wait_for(acquire(slots, 0), .5)      # second slot, no wait
        p1_task = asyncio.create_task(acquire(slots, 20))        # P1 must keep waiting
        p0_third = asyncio.create_task(acquire(slots, 0))        # both slots busy: waits
        await asyncio.sleep(.3)
        assert not p1_task.done() and not p0_third.done()
        await p0.release()                                       # frees the P0 slot -> next P0
        third = await asyncio.wait_for(p0_third, .5)
        assert not p1_task.done()
        await third.release()
        await asyncio.sleep(.1)
        assert not p1_task.done()                                # P1 only gets the main slot
        await p4.release()
        await asyncio.wait_for(p1_task, .5)
        await p1_task.result().release()
        # Disabled (default): P0 queues behind a running P4 like everyone else.
        single = GPUAdmission(client, key_prefix=slot_prefix + ':off', lease_ttl=.6, wait_timeout=2)
        holder = await acquire(single, 80)
        waiting = asyncio.create_task(acquire(single, 0))
        await asyncio.sleep(.2)
        assert not waiting.done()
        await holder.release()
        await (await asyncio.wait_for(waiting, .5)).release()
        try: GPUAdmission(client, p0_execution_slots=2)
        except ValueError: pass
        else: raise AssertionError('invalid slot count accepted')

        for invalid in (-1,101,True,'0'):
            try: await gate.acquire(invalid)
            except ValueError: pass
            else: raise AssertionError('invalid priority accepted')
        for invalid in (-1, 129, True, '1'):
            try: GPUAdmission(client, reserved_p0_waiters=invalid)
            except ValueError: pass
            else: raise AssertionError('invalid P0 reservation accepted')
        print('GPU admission real-Redis checks passed: P0-P4 order, cross-client P0 reservation, '
              'queue caps, zero reservation, FIFO, cancellation, expiry and storage failures')
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.gather(*(lease.release() for lease in leases), return_exceptions=True)
            await client.delete(*gate.keys)
        finally:
            await asyncio.gather(client.aclose(), client2.aclose())


asyncio.run(main())
