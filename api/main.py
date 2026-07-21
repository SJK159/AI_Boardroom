from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.sessions import router as sessions_router

app = FastAPI(title="AI for the Boardroom - Agent API")

# The Express gateway is the only intended caller in production; permissive CORS here is fine
# for local dev where the React dev server may also hit this port directly during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
