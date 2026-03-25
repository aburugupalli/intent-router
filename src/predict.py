from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import load


TEMPLATES: dict[str, str] = {
    "sales": "Danke für dein Interesse! Magst du mir kurz sagen: Teamgröße, Use Case und gewünschten Startzeitpunkt?",
    "support": "Danke dir! Kannst du mir kurz sagen: Gerät/Browser, seit wann das Problem besteht und ggf. die genaue Fehlermeldung?",
    "billing": "Klar! Sag mir bitte die Rechnungsnummer (oder E-Mail) und was genau nicht passt, dann schaue ich mir das an.",
    "cancellation": "Schade, dass du gehen möchtest. Soll die Kündigung sofort oder zum Ende der Laufzeit erfolgen?",
    "feature_request": "Danke für den Vorschlag! Welches Problem löst das für dich und wie oft tritt es auf?",
    "feedback": "Danke fürs Feedback! Was genau war gut/schlecht und in welchem Schritt ist es dir aufgefallen?",
}

ROUTING: dict[str, str] = {
    "sales": "route:sales-team",
    "support": "route:support-queue",
    "billing": "route:billing",
    "cancellation": "route:retention",
    "feature_request": "route:product",
    "feedback": "route:product-feedback",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict intent + route + draft reply.")
    p.add_argument("text", type=str, help="User message text")
    p.add_argument("--model", type=Path, default=Path("artifacts/intent_router.joblib"))
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model}. Train first: uv run python src/train.py"
        )

    model = load(args.model)

    proba = model.predict_proba([args.text])[0]
    classes = list(model.classes_)
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    top1_label, top1_p = ranked[0]
    top2_label, top2_p = ranked[1] if len(ranked) > 1 else (None, None)

    if float(top1_p) < float(args.threshold):
        intent = "needs_clarification"
        route = "route:human-triage"
        reply = (
            "Ich bin nicht sicher, worum es genau geht. Geht es eher um "
            "1) Support/Problem, 2) Preise/Demo, 3) Rechnung/Zahlung oder 4) Kündigung?"
        )
    else:
        intent = str(top1_label)
        route = ROUTING.get(intent, "route:unknown")
        reply = TEMPLATES.get(intent, "Danke! Kannst du mir kurz mehr Kontext geben?")

    payload = {
        "text": args.text,
        "top1": {"label": str(top1_label), "p": float(top1_p)},
        "top2": (
            {"label": str(top2_label), "p": float(top2_p)} if top2_label is not None else None
        ),
        "intent": intent,
        "confidence": float(top1_p),
        "route": route,
        "draft_reply": reply,
        "threshold": float(args.threshold),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("\n=== Prediction ===")
    print("text:", payload["text"])
    print(f"top1: {payload['top1']['label']} ({payload['top1']['p']:.3f})")
    if payload["top2"] is not None:
        print(f"top2: {payload['top2']['label']} ({payload['top2']['p']:.3f})")
    print("intent:", payload["intent"])
    print("route:", payload["route"])
    print("\n--- draft reply ---")
    print(payload["draft_reply"])


if __name__ == "__main__":
    main()
