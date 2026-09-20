# supabase_route.py

import random
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.supabase_client import get_db
from app.predictor import (
    predict_category,
    predict_category_rf,
    predict_urgency_rf,
)

router = APIRouter(prefix="/tickets", tags=["Tickets"])

CONFIDENCE_THRESHOLD = 0.70
MAX_ID_RETRIES = 10

# Model metadata used for storing in prediction tables
DEPT_MODEL_FINAL = "TFIDF_LogisticRegression"   # char+word combined, primary
DEPT_MODEL_RF    = "TFIDF_RandomForest"         # word-only baseline
URG_MODEL_RF     = "TFIDF_RandomForest"         # urgency model
MODEL_VERSION    = "1.0"


# ============================================================
#  Schemas
# ============================================================
class TicketPayload(BaseModel):
    ticket_text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ticket_text": "I was charged twice for my order"}
            ]
        }
    }


class ModelPrediction(BaseModel):
    label: str
    confidence: float


class ThreeModelsResult(BaseModel):
    category_final: ModelPrediction
    category_rf:    ModelPrediction
    urgency_rf:     ModelPrediction


class PredictionResponse(BaseModel):
    ticket_id: int
    ticket_text: str
    created_at: str
    routing_status: str
    assigned_team: Optional[str] = None
    priority: Optional[str] = None

    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float

    department_predictions: List[dict]
    urgency_predictions: List[dict]

    all_models: ThreeModelsResult


# ============================================================
#  Helpers
# ============================================================
def _to_model_prediction(result: dict) -> ModelPrediction:
    label = result.get("category") or result.get("urgency") or ""
    return ModelPrediction(label=str(label), confidence=float(result["confidence"]))


def _random_6digit_id() -> int:
    """Generate a random 6-digit integer between 100000 and 999999."""
    return random.randint(100000, 999999)


def _insert_ticket(db: Session, ticket_text: str) -> dict:
    """
    Insert a new ticket with a random 6-digit ticket_id.
    Retries on duplicate-key collisions (up to MAX_ID_RETRIES times).
    """
    last_error = None

    for _ in range(MAX_ID_RETRIES):
        ticket_id = _random_6digit_id()
        try:
            row = db.execute(
                text("""
                    insert into tickets (ticket_id, ticket_text)
                    values (:tid, :txt)
                    returning ticket_id, ticket_text, created_at
                """),
                {"tid": ticket_id, "txt": ticket_text},
            ).mappings().first()
            db.flush()
            return dict(row)

        except IntegrityError as e:
            db.rollback()
            last_error = e
            continue

        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"tickets insert failed: {e}",
            )

    raise HTTPException(
        status_code=500,
        detail=f"Could not generate a unique 6-digit ticket_id after "
               f"{MAX_ID_RETRIES} attempts: {last_error}",
    )


def _insert_department_prediction(
    db: Session,
    ticket_id: int,
    model_name: str,
    predicted_department: str,
    confidence: float,
    model_version: str = MODEL_VERSION,
) -> dict:
    try:
        row = db.execute(
            text("""
                insert into department_predictions
                    (ticket_id, model_name, model_version,
                     predicted_department, confidence)
                values
                    (:tid, :model, :ver, :dept, :conf)
                returning prediction_id, ticket_id, model_name, model_version,
                          predicted_department, confidence, created_at
            """),
            {
                "tid":   ticket_id,
                "model": model_name,
                "ver":   model_version,
                "dept":  predicted_department,
                "conf":  confidence,
            },
        ).mappings().first()
        return dict(row)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"department_predictions insert failed: {e}",
        )


def _insert_urgency_prediction(
    db: Session,
    ticket_id: int,
    model_name: str,
    predicted_urgency: str,
    confidence: float,
    model_version: str = MODEL_VERSION,
) -> dict:
    try:
        row = db.execute(
            text("""
                insert into urgency_predictions
                    (ticket_id, model_name, model_version,
                     predicted_urgency, confidence)
                values
                    (:tid, :model, :ver, :urg, :conf)
                returning prediction_id, ticket_id, model_name, model_version,
                          predicted_urgency, confidence, created_at
            """),
            {
                "tid":   ticket_id,
                "model": model_name,
                "ver":   model_version,
                "urg":   predicted_urgency,
                "conf":  confidence,
            },
        ).mappings().first()
        return dict(row)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"urgency_predictions insert failed: {e}",
        )


# ============================================================
#  POST /tickets
# ============================================================
@router.post(
    "",
    response_model=PredictionResponse,
    summary="Create ticket & store predictions",
    description="Submit a new ticket text. Runs department (2 models) + urgency "
                "(1 model) predictions, generates a random 6-digit ticket_id, "
                "and stores rows in `tickets`, `department_predictions`, "
                "and `urgency_predictions`.",
)
def create_ticket(
    payload: TicketPayload,
    db: Session = Depends(get_db),
):
    ticket_text = payload.ticket_text.strip()
    if not ticket_text:
        raise HTTPException(status_code=400, detail="Ticket text is empty.")

    # ---------- 1. Run the models ----------
    cat_final = _to_model_prediction(predict_category(ticket_text))
    cat_rf    = _to_model_prediction(predict_category_rf(ticket_text))
    urg_rf    = _to_model_prediction(predict_urgency_rf(ticket_text))

    all_models = ThreeModelsResult(
        category_final=cat_final,
        category_rf=cat_rf,
        urgency_rf=urg_rf,
    )

    # ---------- 2. Routing policy ----------
    if (cat_final.confidence < CONFIDENCE_THRESHOLD
            or urg_rf.confidence < CONFIDENCE_THRESHOLD):
        routing_status = "MANUAL_REVIEW"
        assigned_team = None
        priority = None
    else:
        routing_status = "AUTO_ROUTED"
        assigned_team = cat_final.label
        priority = urg_rf.label

    category            = cat_final.label
    category_confidence = cat_final.confidence
    urgency             = urg_rf.label
    urgency_confidence  = urg_rf.confidence

    # ---------- 3. Insert ticket ----------
    ticket_row = _insert_ticket(db, ticket_text)
    ticket_id  = ticket_row["ticket_id"]

    # ---------- 4. Insert 2 department predictions ----------
    dept_pred_rows = [
        _insert_department_prediction(
            db, ticket_id, DEPT_MODEL_FINAL,
            cat_final.label, cat_final.confidence,
        ),
        _insert_department_prediction(
            db, ticket_id, DEPT_MODEL_RF,
            cat_rf.label, cat_rf.confidence,
        ),
    ]

    # ---------- 5. Insert 1 urgency prediction ----------
    urg_pred_rows = [
        _insert_urgency_prediction(
            db, ticket_id, URG_MODEL_RF,
            urg_rf.label, urg_rf.confidence,
        )
    ]

    # ---------- 6. Commit ----------
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Commit failed: {e}")

    dept_pred_list = [
        {**r, "created_at": str(r["created_at"])} for r in dept_pred_rows
    ]
    urg_pred_list = [
        {**r, "created_at": str(r["created_at"])} for r in urg_pred_rows
    ]

    # ---------- 7. Return ----------
    return PredictionResponse(
        ticket_id=ticket_id,
        ticket_text=ticket_row["ticket_text"],
        created_at=str(ticket_row["created_at"]),
        routing_status=routing_status,
        assigned_team=assigned_team,
        priority=priority,
        category=category,
        category_confidence=category_confidence,
        urgency=urgency,
        urgency_confidence=urgency_confidence,
        department_predictions=dept_pred_list,
        urgency_predictions=urg_pred_list,
        all_models=all_models,
    )


# ============================================================
#  GET /tickets — list all
# ============================================================
@router.get(
    "",
    summary="List tickets",
    description="Return all tickets with their nested department and urgency predictions.",
)
def list_tickets(db: Session = Depends(get_db)):
    try:
        tickets_rows = db.execute(text("""
            select ticket_id, ticket_text, created_at
            from tickets
            order by created_at desc
        """)).mappings().all()

        if not tickets_rows:
            return []

        ticket_ids = [r["ticket_id"] for r in tickets_rows]

        dept_rows = db.execute(
            text("""
                select prediction_id, ticket_id, model_name, model_version,
                       predicted_department, confidence, created_at
                from department_predictions
                where ticket_id = any(:tids)
                order by prediction_id
            """),
            {"tids": ticket_ids},
        ).mappings().all()

        urg_rows = db.execute(
            text("""
                select prediction_id, ticket_id, model_name, model_version,
                       predicted_urgency, confidence, created_at
                from urgency_predictions
                where ticket_id = any(:tids)
                order by prediction_id
            """),
            {"tids": ticket_ids},
        ).mappings().all()

        dept_by_ticket: dict[int, list] = {}
        for d in dept_rows:
            d_dict = dict(d)
            d_dict["created_at"] = str(d_dict["created_at"])
            dept_by_ticket.setdefault(d["ticket_id"], []).append(d_dict)

        urg_by_ticket: dict[int, list] = {}
        for u in urg_rows:
            u_dict = dict(u)
            u_dict["created_at"] = str(u_dict["created_at"])
            urg_by_ticket.setdefault(u["ticket_id"], []).append(u_dict)

        result = []
        for t in tickets_rows:
            tid = t["ticket_id"]
            result.append({
                "ticket_id": tid,
                "ticket_text": t["ticket_text"],
                "created_at": str(t["created_at"]),
                "department_predictions": dept_by_ticket.get(tid, []),
                "urgency_predictions": urg_by_ticket.get(tid, []),
            })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")


# ============================================================
#  GET /tickets/{ticket_id}
# ============================================================
@router.get(
    "/{ticket_id}",
    summary="Get ticket by ID",
    description="Return a single ticket by its 6-digit ticket_id including all predictions.",
)
def get_ticket(
    ticket_id: int = Path(..., ge=100000, le=999999, examples=[483721]),
    db: Session = Depends(get_db),
):
    try:
        ticket = db.execute(text("""
            select ticket_id, ticket_text, created_at
            from tickets
            where ticket_id = :tid
        """), {"tid": ticket_id}).mappings().first()

        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        dept_rows = db.execute(text("""
            select prediction_id, ticket_id, model_name, model_version,
                   predicted_department, confidence, created_at
            from department_predictions
            where ticket_id = :tid
            order by prediction_id
        """), {"tid": ticket_id}).mappings().all()

        urg_rows = db.execute(text("""
            select prediction_id, ticket_id, model_name, model_version,
                   predicted_urgency, confidence, created_at
            from urgency_predictions
            where ticket_id = :tid
            order by prediction_id
        """), {"tid": ticket_id}).mappings().all()

        dept_list = [{**dict(r), "created_at": str(r["created_at"])} for r in dept_rows]
        urg_list  = [{**dict(r), "created_at": str(r["created_at"])} for r in urg_rows]

        return {
            "ticket_id": ticket["ticket_id"],
            "ticket_text": ticket["ticket_text"],
            "created_at": str(ticket["created_at"]),
            "department_predictions": dept_list,
            "urgency_predictions": urg_list,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")