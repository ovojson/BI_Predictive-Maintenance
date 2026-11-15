"""Predict machine failures from the two CSV files supplied for the assignment."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from physik import PhysicalUnion


# Gemeinsames Eingabeschema, feste Aufteilungen und Kosten für die Auswertung.
TARGET = "Machine failure"
FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
SEEDS = (2006, 2026, 2007, 888, 1024)
FALSE_NEGATIVE_COST = 6
# Ein Alarm lohnt sich bei kalibriertem Risiko p, sobald 1 − p ≤ 6p gilt.
ECONOMIC_THRESHOLD = 1 / (1 + FALSE_NEGATIVE_COST)


# MODELLBEGINN: Hybridmodell (physikalische Regeln und Verschleißalarm)
class FailureModel:
    def __init__(self, wear_alarm=True, round_gap=False, pool_stress=True):
        self.wear_alarm = wear_alarm
        self.mechanisms = PhysicalUnion(
            false_negative_cost=FALSE_NEGATIVE_COST, min_precision=1.0,
            round_gap=round_gap, pool_stress=pool_stress,
        )

    def _residual_features(self, frame):
        # Der Standzeit-Effekt beginnt am gelernten Knickpunkt. Type ist ordinal kodiert.
        wear = frame["Tool wear [min]"].to_numpy(dtype=float)
        quality = frame["Type"].map({"L": 0, "M": 1, "H": 2}).to_numpy()
        return np.column_stack((np.maximum(wear - self.wear_onset_, 0), quality))

    def fit(self, frame, labels):
        # Die physikalischen Regeln bilden die erste Alarmstufe.
        self.mechanisms.fit(frame, labels)
        self.residual_ = None
        if self.wear_alarm:
            # Die zweite Stufe lernt nur aus Vorgängen ohne physikalischen Alarm.
            unexplained = self.mechanisms.predict(frame) == 0
            residual_frame = frame.loc[unexplained]
            residual_labels = labels[unexplained]
            # Das erste ausreichend besetzte 10-Minuten-Band mit erhöhter Ausfallrate setzt den Knickpunkt.
            wear_bands = (residual_frame["Tool wear [min]"] // 10) * 10
            summary = pd.Series(residual_labels, index=residual_frame.index).groupby(wear_bands)
            rate = summary.agg(["size", "mean"])
            elevated = rate[(rate["size"] >= 30) & (rate["mean"] >= .02)]
            if len(elevated) and np.unique(residual_labels).size == 2:
                self.wear_onset_ = float(elevated.index.min())
                # MODELLBEGINN: Logistische Regression für das verbleibende Verschleißrisiko
                self.residual_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
                self.residual_.fit(self._residual_features(residual_frame), residual_labels)

        self.thresholds_ = self.mechanisms.thresholds_.copy()
        if self.residual_ is not None:
            self.thresholds_.update(wear_hinge=self.wear_onset_, wear_probability=ECONOMIC_THRESHOLD)
        return self

    def rules(self, frame):
        # Ein Verschleißalarm ergänzt die vorhandenen Regeltreffer.
        rules = self.mechanisms.rules(frame)
        if self.residual_ is not None:
            at_risk = self.residual_.predict_proba(self._residual_features(frame))[:, 1]
            covered = np.logical_or.reduce(tuple(rules.values()))
            rules["wear"] = (at_risk >= ECONOMIC_THRESHOLD) & ~covered
        return rules

    def predict(self, frame):
        # Schon ein einziges aktives Alarmsignal ergibt die Vorhersage 1.
        return np.logical_or.reduce(tuple(self.rules(frame).values())).astype(int)


def read_data(path, labeled=False):
    # Spalten, Varianten und Sensorwerte prüfen. Float64-Präzision bleibt beim Einlesen erhalten.
    frame = pd.read_csv(path, float_precision="round_trip")
    expected = set(FEATURES + ([TARGET] if labeled else []))
    if frame.empty or set(frame.columns) != expected:
        raise ValueError(f"{path}: Erwartete CSV-Spalten: {sorted(expected)}")
    if not frame["Type"].isin(("L", "M", "H")).all():
        raise ValueError(f"{path}: Type muss L, M oder H sein.")
    if not np.isfinite(frame[FEATURES[1:]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: Ungültiger oder fehlender Sensorwert.")
    if labeled and set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{path}: Die Zielspalte muss beide Klassen 0 und 1 enthalten.")
    return frame


def counts(y, pred):
    # Die Konfusionsmatrix liefert die beiden kostenpflichtigen Fehlerarten.
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "cost": fp + FALSE_NEGATIVE_COST * fn}


def evaluate(frame, make_model, seeds=SEEDS):
    # Für jeden Seed werden alle Zeilen genau einmal mit einem getrennt trainierten Modell bewertet.
    x = frame[FEATURES]
    y = frame[TARGET].to_numpy(dtype=int)
    repeats = []
    for seed in seeds:
        pred = np.empty_like(y)
        folds = []
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train, test in split.split(x, y):
            estimator = make_model().fit(x.iloc[train], y[train])
            pred[test] = estimator.predict(x.iloc[test])
            folds.append(counts(y[test], pred[test]))
        type_counts = {}
        for kind in ("L", "M", "H"):
            mask = x["Type"].to_numpy() == kind
            type_counts[kind] = counts(y[mask], pred[mask])
        repeats.append({"seed": seed, **counts(y, pred), "folds": folds, "types": type_counts})
    return repeats


def forecast(repeats, n_train, n_new):
    # Alle Wiederholungen verwenden dieselben Labels. Die mittleren Fehlerraten
    # gelten als Schätzung aus n_train Fällen und werden auf die Challenge-Größe skaliert.
    fp = float(np.mean([run["fp"] for run in repeats]))
    fn = float(np.mean([run["fn"] for run in repeats]))
    expected = n_new * (fp + FALSE_NEGATIVE_COST * fn) / n_train
    # Die Simulation berücksichtigt Schwankungen in den Fehlerraten und Fallzahlen.
    rng = np.random.default_rng(371)
    rates = rng.dirichlet([fp + .5, fn + .5, n_train - fp - fn + .5], size=50_000)
    simulated = rng.multinomial(n_new, rates)
    costs = simulated[:, 0] + FALSE_NEGATIVE_COST * simulated[:, 1]
    return {
        "expected_cost": expected,
        "predictive_95": np.quantile(costs, [0.025, 0.975]).astype(int).tolist(),
    }


def main():
    # Kommandozeilenoptionen legen die Dateien, den Gruppennamen und den Validierungslauf fest.
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = Path.home() / "Downloads"
    parser.add_argument("--modeling", type=Path, default=data_dir / "modeling.csv")
    parser.add_argument("--challenge", type=Path, default=data_dir / "challenge.csv")
    parser.add_argument("--gruppe", default="GRUPPENNAME")
    parser.add_argument("--ausgabe", type=Path, default=Path(__file__).parent / "abgabe")
    parser.add_argument("--nur-prognose", action="store_true", help="Vorhersage ohne erneute CV schreiben.")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w -]+", args.gruppe) or not args.gruppe.strip():
        parser.error("--gruppe darf Buchstaben, Ziffern, Leerzeichen, _ und - enthalten.")

    # Das Abgabemodell verwendet alle gelabelten Daten. Die Validierung trainiert eigene Teilmodelle.
    frame = read_data(args.modeling, labeled=True)
    model = FailureModel().fit(frame[FEATURES], frame[TARGET].to_numpy())
    if not args.nur_prognose:
        results = evaluate(frame, FailureModel)
        core_results = evaluate(frame, lambda: FailureModel(wear_alarm=False))
        rounded_results = evaluate(frame, lambda: FailureModel(wear_alarm=False, round_gap=True))
        independent_results = evaluate(frame, lambda: FailureModel(wear_alarm=False, pool_stress=False))
    # Auswertung auf dem Vollfit für Regelabdeckung und Verschleißgruppen.
    labels = frame[TARGET].to_numpy()
    model_rules = model.rules(frame)
    triggered = np.logical_or.reduce(tuple(model_rules.values()))
    no_core_alarm = model.mechanisms.predict(frame) == 0
    wear = frame["Tool wear [min]"].to_numpy()
    bins = [(0, 194), (195, 199), (200, 209), (210, 224), (225, 234), (235, np.inf)]
    wear_residuals = [
        {"min": int(start), "max": int(end) if np.isfinite(end) else None,
         "fails": int(labels[no_core_alarm & (wear >= start) & (wear <= end)].sum()),
         "n": int(np.sum(no_core_alarm & (wear >= start) & (wear <= end)))}
        for start, end in bins
    ]

    # Die Challenge-Daten werden nach dem Training eingelesen und in Originalreihenfolge prognostiziert.
    challenge = read_data(args.challenge)
    pred = model.predict(challenge)
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    submission = args.ausgabe / f"challenge_prediction_{args.gruppe}.csv"
    pd.DataFrame({TARGET: pred}).to_csv(submission, index=False, lineterminator="\n")
    written = pd.read_csv(submission)
    if list(written.columns) != [TARGET] or not np.array_equal(written[TARGET].to_numpy(), pred):
        raise RuntimeError("Die zurückgelesene Vorhersagedatei weicht von der Vorhersage ab.")
    print(f"Vorhersagen: {submission.resolve()}")
    if args.nur_prognose:
        return

    # Auslieferungsprüfung und Validierungsdetails maschinenlesbar speichern.
    diagnostics = {
        "modeling_rows": len(frame),
        "modeling_failures": int(frame[TARGET].sum()),
        "modeling_full_fit": counts(labels, triggered),
        "modeling_rule_counts": {
            name: {"count": int(rule.sum()), "failures": int(labels[rule].sum())}
            for name, rule in model_rules.items()
        },
        "wear_residuals_full_fit": wear_residuals,
        "validation": results,
        "core_validation": core_results,
        "rounded_gap_validation": rounded_results,
        "independent_stress_validation": independent_results,
        "challenge_rows": len(challenge),
        "challenge_alarms": int(pred.sum()),
        "challenge_forecast": forecast(results, len(frame), len(challenge)),
        "thresholds": model.thresholds_,
        "challenge_rule_alarms": {name: int(rule.sum()) for name, rule in model.rules(challenge).items()},
    }
    text = json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False)
    validation_path = args.ausgabe / "validierung.json"
    validation_path.write_text(text + "\n", encoding="utf-8")
    print(f"Validierung: {validation_path.resolve()}")
    print(f"CV-Kosten: {[run['cost'] for run in results]}")
    print(f"Challenge: {len(challenge)} Vorhersagen, {pred.sum()} Alarme")


if __name__ == "__main__":
    main()
