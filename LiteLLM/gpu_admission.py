"""One shared GPU slot (plus an optional P0-only slot) with Redis priority/FIFO admission and expiring leases."""
import asyncio
import math
import uuid


class AdmissionError(RuntimeError):
    pass


class AdmissionQueueFull(AdmissionError):
    pass


class AdmissionTimeout(AdmissionError):
    pass


# All queue mutations and ownership comparisons run atomically on Redis.
_SCRIPT = """
local q, expiry, tickets, seq, active, active_p0 = unpack(KEYS)
local op, id = ARGV[1], ARGV[2]
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
for _, dead in ipairs(redis.call('ZRANGEBYSCORE', expiry, '-inf', now)) do
  local member = redis.call('HGET', tickets, dead)
  if member then redis.call('ZREM', q, member) end
  redis.call('ZREM', expiry, dead)
  redis.call('HDEL', tickets, dead)
end
if op == 'enqueue' then
  local limit = tonumber(ARGV[5])
  if tonumber(ARGV[3]) ~= 0 then limit = limit-tonumber(ARGV[6]) end
  if redis.call('ZCARD', q) >= limit then return -1 end
  local n = redis.call('INCR', seq)
  local member = string.format('%020.0f', n)..':'..id
  redis.call('ZADD', q, tonumber(ARGV[3]), member)
  redis.call('ZADD', expiry, now+tonumber(ARGV[4]), id)
  redis.call('HSET', tickets, id, member)
  return 1
elseif op == 'claim' then
  local member = redis.call('HGET', tickets, id)
  if not member then return -1 end
  local first = redis.call('ZRANGE', q, 0, 0)[1]
  if member ~= first then return 0 end
  local got = redis.call('SET', active, id, 'NX', 'PX', ARGV[3])
  -- P0 sorts first (score 0), so only P0 can be head while waiting for its reserved slot.
  if not got and ARGV[4] == '1' and redis.call('ZSCORE', q, member) == '0' then
    got = redis.call('SET', active_p0, id, 'NX', 'PX', ARGV[3])
  end
  if got then
    redis.call('ZREM', q, member)
    redis.call('ZREM', expiry, id)
    redis.call('HDEL', tickets, id)
    return 1
  end
  return 0
elseif op == 'renew' then
  local slot = active
  if redis.call('GET', slot) ~= id then slot = active_p0 end
  if redis.call('GET', slot) ~= id then return 0 end
  redis.call('PEXPIRE', slot, ARGV[3])
  return 1
elseif op == 'remove' then
  local member = redis.call('HGET', tickets, id)
  if member then redis.call('ZREM', q, member) end
  redis.call('ZREM', expiry, id)
  redis.call('HDEL', tickets, id)
  if redis.call('GET', active) == id then redis.call('DEL', active) end
  if redis.call('GET', active_p0) == id then redis.call('DEL', active_p0) end
  return 1
end
return redis.error_reply('invalid operation')
"""


class GPUAdmission:
    def __init__(self, redis_client, *, key_prefix="litellm:gpu-admission",
                 max_waiters=128, wait_timeout=120.0, lease_ttl=30.0,
                 poll_interval=0.05, reserved_p0_waiters=1, p0_execution_slots=0):
        if (type(max_waiters) is not int or not 1 <= max_waiters <= 128
                or type(reserved_p0_waiters) is not int
                or not 0 <= reserved_p0_waiters <= max_waiters
                or p0_execution_slots not in (0, 1)
                or any(not math.isfinite(v) or v <= 0
                       for v in (wait_timeout, lease_ttl, poll_interval))):
            raise ValueError("invalid GPU admission limits")
        self.redis = redis_client
        self.keys = [key_prefix + ':' + part for part in
                     ('queue', 'expiry', 'tickets', 'sequence', 'active', 'active-p0')]
        self.max_waiters = max_waiters
        self.reserved_p0_waiters = reserved_p0_waiters
        self.p0_execution_slots = p0_execution_slots
        self.wait_timeout = wait_timeout
        self.lease_ms = max(1, math.ceil(lease_ttl * 1000))
        self.poll_interval = poll_interval

    async def _call(self, operation, ticket, *args):
        try:
            return await asyncio.wait_for(
                self.redis.eval(_SCRIPT, len(self.keys), *self.keys,
                                operation, ticket, *args),
                timeout=min(5.0, self.lease_ms / 3000))
        except asyncio.CancelledError:
            raise
        except Exception:
            raise AdmissionError("GPU admission storage unavailable") from None

    async def acquire(self, priority):
        if type(priority) is not int or not 0 <= priority <= 100:
            raise ValueError("priority must be an integer from 0 to 100")
        ticket = uuid.uuid4().hex
        deadline = asyncio.get_running_loop().time() + self.wait_timeout
        try:
            result = await self._call('enqueue', ticket, priority,
                                      math.ceil(self.wait_timeout * 1000), self.max_waiters,
                                      self.reserved_p0_waiters)
            if result == -1:
                raise AdmissionQueueFull("GPU admission queue full")
            while True:
                result = await self._call('claim', ticket, self.lease_ms,
                                          str(self.p0_execution_slots))
                if result == 1:
                    return GPULease(self, ticket)
                remaining = deadline - asyncio.get_running_loop().time()
                if result == -1 or remaining <= 0:
                    raise AdmissionTimeout("GPU admission wait timed out")
                await asyncio.sleep(min(self.poll_interval, remaining))
        except BaseException:
            # Also removes a slot claimed just before cancellation; UUID protects successors.
            cleanup = asyncio.create_task(self._call('remove', ticket))
            try:
                await asyncio.shield(cleanup)
            except (AdmissionError, asyncio.CancelledError):
                # Expiry is the fallback when storage or the worker disappears.
                cleanup.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            raise


class GPULease:
    def __init__(self, gate, ticket):
        self.gate = gate
        self.ticket = ticket
        self.lost = asyncio.Event()
        self._released = False
        self._heartbeat = asyncio.create_task(self._keep_alive())

    async def _keep_alive(self):
        try:
            while True:
                await asyncio.sleep(self.gate.lease_ms / 3000)
                if await self.gate._call('renew', self.ticket, self.gate.lease_ms) != 1:
                    self.lost.set()
                    return
        except AdmissionError:
            self.lost.set()

    async def release(self):
        if self._released:
            return
        self._heartbeat.cancel()
        try:
            await self._heartbeat
        except asyncio.CancelledError:
            pass
        await self.gate._call('remove', self.ticket)
        self._released = True
