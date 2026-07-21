"""In-memory per-session event bus for streaming board-session progress over WebSocket.

Single-process only (no Redis/pubsub) - appropriate for this project's deployment scope
(a portfolio demo, not a horizontally-scaled service). If this ever needs multiple API
processes, this is the file that would grow a Redis-backed backend instead.
"""
import asyncio
import queue
import time
import uuid


class SessionBus:
    def __init__(self):
        self._queues: dict[str, "queue.Queue[dict | None]"] = {}

    def create(self) -> str:
        session_id = str(uuid.uuid4())
        self._queues[session_id] = queue.Queue()
        return session_id

    def make_emitter(self, session_id: str):
        """Returns a plain (non-async) callback safe to call from a worker thread - this is
        what gets threaded through BossAgent.run(on_event=...), which executes off the asyncio
        event loop entirely (see routes/sessions.py's run_in_executor call)."""
        q = self._queues[session_id]

        def emit(event_type: str, data: dict) -> None:
            # Groq's structured-output call returns the full synthesis text in one shot (see
            # backend/agents/boss/graph.py's docstring on this tradeoff) - drip-feeding it word
            # by word here is what makes the frontend's "streamed token-by-token" UX genuine
            # network-level streaming rather than a client-side-only typing animation.
            if event_type == "synthesis_ready":
                words = data["synthesis"].split(" ")
                for i, word in enumerate(words):
                    chunk = word if i == 0 else " " + word
                    q.put({"type": "synthesis_chunk", "data": {"chunk": chunk}})
                    time.sleep(0.02)
            q.put({"type": event_type, "data": data})

        return emit

    def close(self, session_id: str) -> None:
        q = self._queues.get(session_id)
        if q is not None:
            q.put(None)

    async def stream(self, session_id: str):
        """Async generator yielding events for a session until it's closed. Bridges the
        thread-safe stdlib Queue to the asyncio world with a thread-pool get() per item -
        cheap, and avoids needing an asyncio.Queue plus call_soon_threadsafe wiring."""
        q = self._queues[session_id]
        loop = asyncio.get_running_loop()
        try:
            while True:
                event = await loop.run_in_executor(None, q.get)
                if event is None:
                    break
                yield event
        finally:
            self._queues.pop(session_id, None)


session_bus = SessionBus()
