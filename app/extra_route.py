from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.predictor import (
    predict_category,
    predict_category_rf,
    predict_urgency_rf,
)

router = APIRouter(prefix="/route", tags=["Routing"])

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
#  Response schemas
# ============================================================
class ModelPrediction(BaseModel):
    label: str
    confidence: float


class ThreeModelsResult(BaseModel):
    category_final: ModelPrediction
    category_rf:    ModelPrediction
    urgency_rf:     ModelPrediction


class RoutingResult(BaseModel):
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    routing_status: str
    assigned_queue: Optional[str] = None
    priority: Optional[str] = None
    requires_human: bool
    reason: str


# ============================================================
#  Helpers
# ============================================================
def _to_model_prediction(result: dict) -> ModelPrediction:
    label = result.get("category") or result.get("urgency") or ""
    return ModelPrediction(label=str(label), confidence=float(result["confidence"]))


def route_ticket(text: str) -> RoutingResult:
    text = str(text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Ticket text is empty.")

    cat = _to_model_prediction(predict_category(text))
    urg = _to_model_prediction(predict_urgency_rf(text))

    if cat.confidence < CONFIDENCE_THRESHOLD or urg.confidence < CONFIDENCE_THRESHOLD:
        return RoutingResult(
            category=cat.label,
            category_confidence=cat.confidence,
            urgency=urg.label,
            urgency_confidence=urg.confidence,
            routing_status="MANUAL_REVIEW",
            assigned_queue=None,
            priority=None,
            requires_human=True,
            reason=(
                f"Low confidence (cat={cat.confidence:.2f}, "
                f"urg={urg.confidence:.2f}, threshold={CONFIDENCE_THRESHOLD})."
            ),
        )

    return RoutingResult(
        category=cat.label,
        category_confidence=cat.confidence,
        urgency=urg.label,
        urgency_confidence=urg.confidence,
        routing_status="AUTO_ROUTED",
        assigned_queue=cat.label,
        priority=urg.label,
        requires_human=False,
        reason=f"Auto-routed to '{cat.label}' with priority '{urg.label}'.",
    )


# ============================================================
#  Endpoint 1 — raw text → single routing decision
# ============================================================
@router.post("", response_model=RoutingResult)
def route(
    body: str = Body(
        ...,
        media_type="text/plain",
        description="Raw ticket body text.",
        examples=["My card was charged twice, please refund."],
    ),
):
    """Send raw ticket body text. Server routes it."""
    return route_ticket(body)


# ============================================================
#  Endpoint 2 — raw text → all 3 model predictions
# ============================================================
@router.post("/both", response_model=ThreeModelsResult)
def predict_all(
    body: str = Body(
        ...,
        media_type="text/plain",
        description="Raw ticket body text.",
        examples=["My card was charged twice, please refund."],
    ),
):
    """Send raw ticket body text. Returns predictions from all 3 models."""
    text = str(body).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Ticket text is empty.")

    return ThreeModelsResult(
        category_final=_to_model_prediction(predict_category(text)),
        category_rf=_to_model_prediction(predict_category_rf(text)),
        urgency_rf=_to_model_prediction(predict_urgency_rf(text)),
    )