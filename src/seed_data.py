from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEED: list[tuple[str, str]] = [
    # sales
    ("Habt ihr eine Demo diese Woche?", "sales"),
    ("Kannst du mir was zu euren Preisen sagen?", "sales"),
    ("Gibt es einen kostenlosen Test?", "sales"),
    ("Wir brauchen das für 20 Nutzer, geht das?", "sales"),
    ("Kann ich mit jemandem sprechen wegen Angebot?", "sales"),
    # support
    ("Die App lädt nicht, ich sehe nur einen weißen Screen.", "support"),
    ("Ich bekomme einen Fehler beim Login.", "support"),
    ("Das Export-Feature funktioniert nicht mehr.", "support"),
    ("Ich kann kein Passwort zurücksetzen.", "support"),
    ("Die Integration verbindet sich nicht.", "support"),
    # billing
    ("Meine Rechnung stimmt nicht, da ist etwas doppelt.", "billing"),
    ("Kann ich die Rechnung auf Firma ausstellen lassen?", "billing"),
    ("Wo finde ich meine Rechnungen?", "billing"),
    ("Bitte Zahlungsmethode ändern.", "billing"),
    ("Warum wurde ich abgebucht?", "billing"),
    # cancellation
    ("Ich möchte kündigen.", "cancellation"),
    ("Bitte löscht meinen Account.", "cancellation"),
    ("Wie kann ich mein Abo beenden?", "cancellation"),
    ("Ich will mein Konto schließen.", "cancellation"),
    ("Abo stoppen zum Monatsende.", "cancellation"),
    # feature_request
    ("Könnt ihr eine Zapier-Integration bauen?", "feature_request"),
    ("Es wäre super, wenn es Dark Mode gäbe.", "feature_request"),
    ("Habt ihr geplant, CSV Import zu unterstützen?", "feature_request"),
    ("Bitte fügt Mehrsprachigkeit hinzu.", "feature_request"),
    ("Kann man Berichte automatisch planen?", "feature_request"),
    # feedback
    ("Die neue Oberfläche ist echt besser geworden.", "feedback"),
    ("Ich finde das Dashboard verwirrend.", "feedback"),
    ("Tolles Produkt, aber die Ladezeit ist nervig.", "feedback"),
    ("Das Onboarding war super verständlich.", "feedback"),
    ("Ich bin unzufrieden mit der Performance.", "feedback"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a small seed dataset for intent routing.")
    p.add_argument("--out", type=Path, default=Path("data/train.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(SEED, columns=["text", "label"])
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
