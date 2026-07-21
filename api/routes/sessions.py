import asyncio
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.deps import get_governance_logger
from api.events import session_bus
from api.schemas import DecisionRequest, SessionCreateRequest, SessionCreateResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_session_sync(session_id: str, query: str) -> None:
    """Runs entirely on a worker thread (see create_session's run_in_executor call) - this is
    where the actual Databricks/Groq network calls happen, kept off the asyncio event loop."""
    emit = session_bus.make_emitter(session_id)
    governance_logger = get_governance_logger()
    try:
        log = governance_logger.run_with_logging(query, on_event=emit, session_id=session_id)
        emit("session_complete", log.model_dump(mode="json"))
    except Exception as e:
        logger.exception("Board session %s failed", session_id)
        emit("session_error", {"error": str(e)})
    finally:
        session_bus.close(session_id)


@router.post("/sessions", response_model=SessionCreateResponse, status_code=202)
async def create_session(req: SessionCreateRequest):
    session_id = session_bus.create()
    loop = asyncio.get_running_loop()
    # Fire-and-forget: the executor call isn't awaited, so this handler returns immediately
    # with the session_id and the caller connects to the WebSocket to watch it run.
    loop.run_in_executor(None, _run_session_sync, session_id, req.query)
    return SessionCreateResponse(session_id=session_id)


@router.get("/sessions")
async def list_sessions(limit: int = 10):
    logs = get_governance_logger().list_recent_sessions(limit)
    return [log.model_dump(mode="json") for log in logs]


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    log = get_governance_logger().get_session(session_id)
    if log is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return log.model_dump(mode="json")


@router.post("/sessions/{session_id}/decision")
async def record_decision(session_id: str, req: DecisionRequest):
    updated = get_governance_logger().record_human_decision(session_id, req.decision, req.notes)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.websocket("/ws/sessions/{session_id}")
async def stream_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        async for event in session_bus.stream(session_id):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except KeyError:
        # session_id was never created (bad ID) or its bus entry already drained/closed
        await websocket.close(code=4404, reason="Unknown or already-completed session_id")
