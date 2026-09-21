# stats_route.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.supabase_client import get_db


router = APIRouter(prefix="/stats", tags=["Stats"])


# ============================================================
# /stats/summary
# ============================================================

@router.get("/summary", summary="Overall ticket stats")
def stats_summary(db: Session = Depends(get_db)):

    try:

        row = db.execute(
            text("""
                select
                    (select count(*) from tickets)
                        as total_tickets,

                    (select count(*) from department_predictions)
                        as total_department_predictions,

                    (select count(*) from urgency_predictions)
                        as total_urgency_predictions
            """)
        ).mappings().first()

        total_tickets = int(row["total_tickets"])

        return {
            "total_tickets": total_tickets,

            "total_department_predictions":
                int(row["total_department_predictions"]),

            "total_urgency_predictions":
                int(row["total_urgency_predictions"]),

            "avg_department_predictions_per_ticket":
                round(
                    int(row["total_department_predictions"])
                    / total_tickets,
                    2
                )
                if total_tickets else 0.0,

            "avg_urgency_predictions_per_ticket":
                round(
                    int(row["total_urgency_predictions"])
                    / total_tickets,
                    2
                )
                if total_tickets else 0.0,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"DB query failed: {e}"
        )


# ============================================================
# /stats/departments
# Department -> Model -> Tickets
#
# Expected:
#   TFIDF_LogisticRegression
#   TFIDF_RandomForest
# ============================================================

@router.get(
    "/departments",
    summary="Department stats by model",
)
def stats_departments(
    db: Session = Depends(get_db)
):

    try:

        rows = db.execute(
            text("""
                select
                    predicted_department as department,
                    model_name,
                    ticket_id,
                    round(
                        confidence::numeric,
                        4
                    ) as confidence

                from department_predictions

                order by
                    predicted_department,
                    model_name,
                    ticket_id
            """)
        ).mappings().all()


        grouped: dict[str, dict[str, list]] = {}


        for r in rows:

            grouped \
                .setdefault(r["department"], {}) \
                .setdefault(r["model_name"], []) \
                .append(
                    {
                        "ticket_id": int(r["ticket_id"]),
                        "confidence": float(r["confidence"]),
                    }
                )


        result = []


        for department, model_map in grouped.items():

            models_list = []
            all_confs = []


            for model_name in sorted(model_map.keys()):

                ticket_list = model_map[model_name]

                confs = [
                    t["confidence"]
                    for t in ticket_list
                ]

                avg_conf = (
                    round(
                        sum(confs) / len(confs),
                        4
                    )
                    if confs
                    else 0.0
                )

                all_confs.extend(confs)


                models_list.append(
                    {
                        "model_name": model_name,
                        "confidence": avg_conf,
                        "tickets": ticket_list,
                    }
                )


            department_avg = (
                round(
                    sum(all_confs) / len(all_confs),
                    4
                )
                if all_confs
                else 0.0
            )


            result.append(
                {
                    "department": department,
                    "models": models_list,
                    "avg_confidence": department_avg,
                }
            )


        result.sort(
            key=lambda d: d["avg_confidence"],
            reverse=True
        )

        return result


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"DB query failed: {e}"
        )


# ============================================================
# /stats/urgency
# Urgency -> Model -> Tickets
#
# Expected:
#   TFIDF_LogisticRegression
#   TFIDF_RandomForest
# ============================================================

@router.get(
    "/urgency",
    summary="Urgency stats by model",
)
def stats_urgency(
    db: Session = Depends(get_db)
):

    try:

        rows = db.execute(
            text("""
                select
                    predicted_urgency as urgency,
                    model_name,
                    ticket_id,
                    round(
                        confidence::numeric,
                        4
                    ) as confidence

                from urgency_predictions

                order by
                    predicted_urgency,
                    model_name,
                    ticket_id
            """)
        ).mappings().all()


        grouped: dict[str, dict[str, list]] = {}


        for r in rows:

            grouped \
                .setdefault(r["urgency"], {}) \
                .setdefault(r["model_name"], []) \
                .append(
                    {
                        "ticket_id": int(r["ticket_id"]),
                        "confidence": float(r["confidence"]),
                    }
                )


        result = []


        for urgency, model_map in grouped.items():

            models_list = []
            all_confs = []


            for model_name in sorted(model_map.keys()):

                ticket_list = model_map[model_name]

                confs = [
                    t["confidence"]
                    for t in ticket_list
                ]

                avg_conf = (
                    round(
                        sum(confs) / len(confs),
                        4
                    )
                    if confs
                    else 0.0
                )

                all_confs.extend(confs)


                models_list.append(
                    {
                        "model_name": model_name,
                        "confidence": avg_conf,
                        "tickets": ticket_list,
                    }
                )


            urgency_avg = (
                round(
                    sum(all_confs) / len(all_confs),
                    4
                )
                if all_confs
                else 0.0
            )


            result.append(
                {
                    "urgency": urgency,
                    "models": models_list,
                    "avg_confidence": urgency_avg,
                }
            )


        result.sort(
            key=lambda d: d["avg_confidence"],
            reverse=True
        )

        return result


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"DB query failed: {e}"
        )


# ============================================================
# /stats/models
# Model -> Tickets
#
# Department:
#   1. TFIDF_LogisticRegression
#   2. TFIDF_RandomForest
#
# Urgency:
#   1. TFIDF_LogisticRegression
#   2. TFIDF_RandomForest
# ============================================================

@router.get(
    "/models",
    summary="Per-model statistics",
)
def stats_models(
    db: Session = Depends(get_db)
):

    try:

        # ====================================================
        # Department models
        # ====================================================

        dept_rows = db.execute(
            text("""
                select
                    model_name,
                    ticket_id,
                    predicted_department as label,
                    round(
                        confidence::numeric,
                        4
                    ) as confidence

                from department_predictions

                order by
                    model_name,
                    ticket_id
            """)
        ).mappings().all()


        dept_grouped: dict[str, list] = {}


        for r in dept_rows:

            dept_grouped \
                .setdefault(r["model_name"], []) \
                .append(
                    {
                        "ticket_id": int(r["ticket_id"]),
                        "department": r["label"],
                        "confidence": float(r["confidence"]),
                    }
                )


        department_models = []


        for model_name in sorted(
            dept_grouped.keys()
        ):

            tickets = dept_grouped[model_name]

            confs = [
                t["confidence"]
                for t in tickets
            ]

            avg_conf = (
                round(
                    sum(confs) / len(confs),
                    4
                )
                if confs
                else 0.0
            )


            department_models.append(
                {
                    "model_name": model_name,
                    "confidence": avg_conf,
                    "tickets": tickets,
                }
            )


        # ====================================================
        # Urgency models
        # ====================================================

        urg_rows = db.execute(
            text("""
                select
                    model_name,
                    ticket_id,
                    predicted_urgency as label,
                    round(
                        confidence::numeric,
                        4
                    ) as confidence

                from urgency_predictions

                order by
                    model_name,
                    ticket_id
            """)
        ).mappings().all()


        urg_grouped: dict[str, list] = {}


        for r in urg_rows:

            urg_grouped \
                .setdefault(r["model_name"], []) \
                .append(
                    {
                        "ticket_id": int(r["ticket_id"]),
                        "urgency": r["label"],
                        "confidence": float(r["confidence"]),
                    }
                )


        urgency_models = []


        for model_name in sorted(
            urg_grouped.keys()
        ):

            tickets = urg_grouped[model_name]

            confs = [
                t["confidence"]
                for t in tickets
            ]

            avg_conf = (
                round(
                    sum(confs) / len(confs),
                    4
                )
                if confs
                else 0.0
            )


            urgency_models.append(
                {
                    "model_name": model_name,
                    "confidence": avg_conf,
                    "tickets": tickets,
                }
            )


        return {
            "department_models": department_models,
            "urgency_models": urgency_models,
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"DB query failed: {e}"
        )


# ============================================================
# /stats/confidence
# Confidence buckets + ticket drill-down
# ============================================================

@router.get(
    "/confidence",
    summary="Confidence distribution",
)
def stats_confidence(
    db: Session = Depends(get_db)
):

    try:

        # ====================================================
        # Department buckets
        # ====================================================

        dept_rows = db.execute(
            text("""
                select
                    model_name,
                    ticket_id,
                    round(
                        confidence::numeric,
                        4
                    ) as confidence

                from department_predictions

                order by
                    confidence asc,
                    ticket_id
            """)
        ).mappings().all()


        dept_buckets = {
            "below_0_3": [],
            "from_0_3_to_0_5": [],
            "from_0_5_to_0_7": [],
            "from_0_7_to_0_9": [],
            "above_0_9": [],
        }


        for r in dept_rows:

            confidence = float(r["confidence"])

            entry = {
                "ticket_id": int(r["ticket_id"]),
                "model_name": r["model_name"],
                "confidence": confidence,
            }


            if confidence < 0.3:

                dept_buckets[
                    "below_0_3"
                ].append(entry)

            elif confidence < 0.5:

                dept_buckets[
                    "from_0_3_to_0_5"
                ].append(entry)

            elif confidence < 0.7:

                dept_buckets[
                    "from_0_5_to_0_7"
                ].append(entry)

            elif confidence < 0.9:

                dept_buckets[
                    "from_0_7_to_0_9"
                ].append(entry)

            else:

                dept_buckets[
                    "above_0_9"
                ].append(entry)


        dept_output = [
            {
                "bucket": name,
                "count": len(items),
                "tickets": items,
            }
            for name, items in dept_buckets.items()
        ]


        # ====================================================
        # Urgency buckets
        # ====================================================

        urg_rows = db.execute(
            text("""
                select
                    model_name,
                    ticket_id,
                    round(
                        confidence::numeric,
                        4
                    ) as confidence

                from urgency_predictions

                order by
                    confidence asc,
                    ticket_id
            """)
        ).mappings().all()


        urg_buckets = {
            "below_0_3": [],
            "from_0_3_to_0_5": [],
            "from_0_5_to_0_7": [],
            "from_0_7_to_0_9": [],
            "above_0_9": [],
        }


        for r in urg_rows:

            confidence = float(r["confidence"])

            entry = {
                "ticket_id": int(r["ticket_id"]),
                "model_name": r["model_name"],
                "confidence": confidence,
            }


            if confidence < 0.3:

                urg_buckets[
                    "below_0_3"
                ].append(entry)

            elif confidence < 0.5:

                urg_buckets[
                    "from_0_3_to_0_5"
                ].append(entry)

            elif confidence < 0.7:

                urg_buckets[
                    "from_0_5_to_0_7"
                ].append(entry)

            elif confidence < 0.9:

                urg_buckets[
                    "from_0_7_to_0_9"
                ].append(entry)

            else:

                urg_buckets[
                    "above_0_9"
                ].append(entry)


        urg_output = [
            {
                "bucket": name,
                "count": len(items),
                "tickets": items,
            }
            for name, items in urg_buckets.items()
        ]


        # ====================================================
        # Return
        # ====================================================

        return {
            "department_confidence_buckets": dept_output,
            "urgency_confidence_buckets": urg_output,
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"DB query failed: {e}"
        )
        