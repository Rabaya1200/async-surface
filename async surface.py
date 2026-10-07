"""
AI Production Async Reference
=============================
Covers:
  1. Observability (structured logs + metrics)
  2. Token-bucket rate limiter
  3. Resilient LLM client: timeouts, retries, jitter, Retry-After,
     bounded concurrency, streaming
  4. Blocking boundary (asyncio.to_thread) for CPU/GPU work
  5. Backpressure worker pool with asyncio.Queue
  6. Graceful shutdown with TaskGroup
  7. FastAPI: lifespan, streaming endpoint, health
  8. Async DB / Redis / vector DB patterns
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import aiohttp

# ---------------------------------------------------------------------------
# 1. Config + Observability
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("ai.prod")


@dataclass
class Metrics:
    """In production: use prometheus_client. Kept minimal here."""
    requests_total: int = 0
    requests_failed: int = 0
    retries_total: int = 0
    in_flight: int = 0
    tokens_streamed: int = 0

    def snapshot(self) -> dict:
        return self.__dict__.copy()


METRICS = Metrics()


# ---------------------------------------------------------------------------
# 2. Token-bucket rate limiter (per provider / API key)
# ---------------------------------------------------------------------------

class TokenBucket:
    """Provider-level rate limit. Semaphore alone is NOT enough."""

    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate = rate_per_sec
        self.capacity = capacity
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1.0) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.capacity,
                self._tokens + (now - self._updated) * self.rate,
            )
            self._updated = now
            if self._tokens < n:
                wait = (n - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._updated = time.monotonic()
            else:
                self._tokens -= n


# ---------------------------------------------------------------------------
# 3. Resilient LLM client
# ---------------------------------------------------------------------------

class RetryableError(Exception):
    """429, 5xx, timeouts, connection errors."""


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_concurrency: int = 10,
        rate_per_sec: float = 20.0,
        burst: int = 20,
        max_retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.sem = asyncio.Semaphore(max_concurrency)
        self.bucket = TokenBucket(rate_per_sec, burst)
        self.max_retries = max_retries
        self.session: aiohttp.ClientSession | None = None
        # total=global deadline, sock_read=per-chunk read (good for streaming)
        self.timeout = aiohttp.ClientTimeout(total=60, sock_read=30, connect=5)

    async def start(self) -> None:
        # ONE session for the whole app — connection pooling + keep-alive
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=50),
        )

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    # ---- retry wrapper: semaphore is released during backoff ----
    async def _with_retry(self, coro_factory):
        for attempt in range(self.max_retries):
            try:
                async with self.sem:
                    await self.bucket.acquire()
                    METRICS.in_flight += 1
                    try:
                        return await coro_factory()
                    finally:
                        METRICS.in_flight -= 1
            except RetryableError as e:
                METRICS.retries_total += 1
                if attempt == self.max_retries - 1:
                    METRICS.requests_failed += 1
                    raise
                # full jitter exponential backoff
                backoff = min(8.0, 0.5 * 2 ** attempt) + random.uniform(0, 0.5)
                if isinstance(e, RetryAfter):
                    backoff = max(backoff, e.seconds)
                log.warning("retry attempt=%d backoff=%.2f err=%s",
                            attempt + 1, backoff, e)
                await asyncio.sleep(backoff)  # NOT holding the semaphore
        raise RuntimeError("unreachable")

    @staticmethod
    def _classify(status: int, headers) -> None:
        if status == 429:
            ra = headers.get("Retry-After")
            raise RetryAfter(float(ra) if ra else 1.0)
        if 500 <= status < 600:
            raise RetryableError(f"server error {status}")

    # ---- non-streaming completion ----
    async def complete(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        async def call() -> str:
            assert self.session
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                ) as resp:
                    self._classify(resp.status, resp.headers)
                    resp.raise_for_status()
                    data = await resp.json()
                    METRICS.requests_total += 1
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                raise RetryableError(str(e)) from e

        return await self._with_retry(call)

    # ---- streaming completion (tokens -> caller) ----
    async def stream(self, prompt: str, model: str = "gpt-4o-mini") -> AsyncIterator[str]:
        async def open_stream():
            assert self.session
            try:
                cm = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": model, "stream": True,
                          "messages": [{"role": "user", "content": prompt}]},
                )
                resp = await cm.__aenter__()
                self._classify(resp.status, resp.headers)
                resp.raise_for_status()
                return cm, resp
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                raise RetryableError(str(e)) from e

        # Only retry the *connection*; mid-stream failures are not retried
        cm, resp = await self._with_retry(open_stream)
        try:
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                METRICS.tokens_streamed += 1
                yield payload  # in real code: json.loads + extract delta
        finally:
            await cm.__aexit__(None, None, None)


class RetryAfter(RetryableError):
    def __init__(self, seconds: float):
        self.seconds = seconds
        super().__init__(f"429 retry-after={seconds}")


# ---------------------------------------------------------------------------
# 4. Blocking boundary — CPU/GPU work must NOT block the loop
# ---------------------------------------------------------------------------

async def embed_texts(encoder, texts: list[str]) -> list[list[float]]:
    """encoder.encode is blocking (numpy/torch). Offload to a thread."""
    return await asyncio.to_thread(encoder.encode, texts)


async def rerank(reranker, query: str, docs: list[str]) -> list[float]:
    return await asyncio.to_thread(reranker.score, query, docs)


# ---------------------------------------------------------------------------
# 5. Backpressure worker pool (asyncio.Queue)
# ---------------------------------------------------------------------------

async def worker(
    name: str,
    queue: asyncio.Queue,
    client: LLMClient,
    results: list,
    shutdown: asyncio.Event,
) -> None:
    while not shutdown.is_set():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if item is None:  # poison pill
            queue.task_done()
            break
        try:
            out = await client.complete(item)
            results.append((item, out))
        except Exception as e:
            log.error("worker=%s failed prompt=%r err=%s", name, item, e)
            results.append((item, None))
        finally:
            queue.task_done()


async def run_pipeline(prompts: list[str], concurrency: int = 8):
    client = LLMClient("https://api.example.com/v1", "sk-...", max_concurrency=8)
    await client.start()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # backpressure
    results: list = []
    shutdown = asyncio.Event()

    async with asyncio.TaskGroup() as tg:  # cancels children on any failure
        workers = [
            tg.create_task(worker(f"w{i}", queue, client, results, shutdown))
            for i in range(concurrency)
        ]
        producer = tg.create_task(_produce(queue, prompts))
        await producer
        for _ in workers:
            await queue.put(None)  # poison pills

    await client.close()
    return results


async def _produce(queue: asyncio.Queue, prompts: list[str]) -> None:
    for p in prompts:
        await queue.put(p)  # blocks when queue is full -> backpressure


# ---------------------------------------------------------------------------
# 6. Graceful shutdown with signal handling
# ---------------------------------------------------------------------------

async def serve_forever(worker_coro) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in ("SIGINT", "SIGTERM"):
        with contextlib.suppress(NotImplementedError):
            import signal
            loop.add_signal_handler(getattr(signal, sig), stop.set)

    task = asyncio.create_task(worker_coro)
    await stop.wait()
    log.info("shutdown requested — cancelling work")
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# 7. FastAPI integration — lifespan, streaming, health
# ---------------------------------------------------------------------------

def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = LLMClient("https://api.example.com/v1", "sk-...", max_concurrency=20)
        await client.start()
        app.state.llm = client
        log.info("startup complete")
        try:
            yield
        finally:
            await client.close()
            log.info("shutdown complete")

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        ok = app.state.llm.session is not None
        return JSONResponse({"ok": ok, **METRICS.snapshot()}, status_code=200 if ok else 503)

    @app.post("/chat")
    async def chat(req: Request):
        body = await req.json()
        prompt = body["prompt"]

        async def gen():
            async for chunk in app.state.llm.stream(prompt):
                yield f"data: {chunk}\n\n"  # SSE
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/embed")
    async def embed(req: Request):
        body = await req.json()
        vecs = await embed_texts(app.state.encoder, body["texts"])
        return {"vectors": vecs}

    return app


# ---------------------------------------------------------------------------
# 8. Async DB / Redis / vector DB patterns (illustrative)
# ---------------------------------------------------------------------------

async def data_layer_patterns():
    """
    Postgres:   asyncpg / SQLAlchemy[asyncio]  -> 'await conn.fetch(...)'
    Redis:      redis.asyncio                   -> 'await r.get(...)'
    Vector DB:  qdrant-client AsyncQdrantClient / weaviate async / pgvector
    MongoDB:    motor

    Rules:
      - One pool per process, created in lifespan.
      - Never wrap a sync driver in async; use the async driver.
      - Cache LLM outputs in Redis keyed by hash(prompt+model).
      - Batch vector upserts; don't await per-row.
    """
    return None


if __name__ == "__main__":
    prompts = [f"Summarize paper {i}" for i in range(20)]
    results = asyncio.run(run_pipeline(prompts))
    for p, r in results:
        print("OK " if r else "FAIL", p)