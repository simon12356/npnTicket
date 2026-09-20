from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.extra_route import router as routing_router
from app.supabase_route import router as db_router
from app.stats_route import router as stats_router   # ✅ new

app = FastAPI(
    title="Ticket Auto-Router API",
    description="Send raw ticket text — the server routes it and stores it.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(routing_router)   # /route, /route/both
app.include_router(db_router)        # /tickets, ...
app.include_router(stats_router)     # ✅ new — /stats/*


@app.get("/", tags=["Health"])
def root():
    return {"message": "Ticket Auto-Router API is running 🚀"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}