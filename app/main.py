# app/main.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np

from app.model_loader import load_models

# ---------- Load models once at startup ----------
category_model, urgency_vectorizer, urgency_model = load_models()

# ---------- Server-side routing policy ----------
CONFIDENCE_THRESHOLD = 0.70

# ---------- App ----------
app = FastAPI(
    title="Ticket Auto-Router API",
    description="Send raw ticket text — the server routes it.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class RoutingResult(BaseModel):
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    routing_status: str                # AUTO_ROUTED | MANUAL_REVIEW
    assigned_queue: Optional[str]
    priority: Optional[str]
    requires_human: bool
    reason: str


# ---------- Core router ----------
def route_ticket(text: str) -> RoutingResult:
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Ticket text is empty.")

    # --- Category (pipeline: raw text in) ---
    cat_pred = str(category_model.predict([text])[0])
    cat_conf = float(np.max(category_model.predict_proba([text])[0]))

    # --- Urgency (vectorizer → model) ---
    urg_features = urgency_vectorizer.transform([text])
    urg_pred = str(urgency_model.predict(urg_features)[0])
    urg_conf = float(np.max(urgency_model.predict_proba(urg_features)[0]))

    # --- Routing policy ---
    if cat_conf < CONFIDENCE_THRESHOLD or urg_conf < CONFIDENCE_THRESHOLD:
        return RoutingResult(
            category=cat_pred,
            category_confidence=round(cat_conf, 4),
            urgency=urg_pred,
            urgency_confidence=round(urg_conf, 4),
            routing_status="MANUAL_REVIEW",
            assigned_queue=None,
            priority=None,
            requires_human=True,
            reason=(
                f"Low confidence (cat={cat_conf:.2f}, "
                f"urg={urg_conf:.2f}, threshold={CONFIDENCE_THRESHOLD})."
            ),
        )

    return RoutingResult(
        category=cat_pred,
        category_confidence=round(cat_conf, 4),
        urgency=urg_pred,
        urgency_confidence=round(urg_conf, 4),
        routing_status="AUTO_ROUTED",
        assigned_queue=cat_pred,
        priority=urg_pred,
        requires_human=False,
        reason=f"Auto-routed to '{cat_pred}' with priority '{urg_pred}'.",
    )


# ---------- Endpoints ----------
@app.get("/", tags=["Health"])
def root():
    return {"message": "Ticket Auto-Router API is running 🚀"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "threshold": CONFIDENCE_THRESHOLD}


@app.post("/route", response_model=RoutingResult, tags=["Routing"])
def route(
    body: str = Body(
        ...,
        media_type="text/plain",
        description="Just the ticket body text.",
        examples=["I tried resetting my password multiple times but it keeps failing."],
    ),
):
    """
    Send **raw ticket body text**. The server routes it.
    """
    return route_ticket(body)