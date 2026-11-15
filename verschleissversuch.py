"""Prüft eine stückweise Verschleißlogistik und zwei Wechselwirkungen."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from modell import ECONOMIC_THRESHOLD, SEEDS, TARGET, FailureModel, counts, read_data
from modelloptimierung import BASE, TUNED, inner_search, predict, serial_threshold, summary_table, timestamp


# Die Varianten unterscheiden sich nur in den Zusatzmerkmalen der Verschleißstufe.
BASE_CONFIG = "Verschleißlogistik C=1"
NEW_MODEL = "Verschleißkurve mit innerer Auswahl"
KNOT_OFFSET = 20
VARIANTS = {
    "Kurve": (),
    "Kurve + Drehmoment": ("torque",),
    "Kurve + Produktvariante": ("type",),
    "Kurve + beide Wechselwirkungen": ("torque", "type"),
}
GRID = {
    variant: {f"{variant} C={strength:g}": strength for strength in (.01, .1, 1., 10.)}
    for variant in VARIANTS
}
CONFIGS = {
    name: (variant, strength)
    for variant, configurations in GRID.items() for name, strength in configurations.items()
}


def wear_columns(frame, baseline, torque_mean, type_mean):
    """Baut die Kurve und die Interaktionen aus Werten einer Trainingsfalte auf."""
    wear, quality = baseline._residual_features(frame).T
    torque = frame["Torque [Nm]"].to_numpy(dtype=float) - torque_mean
    # Der zusätzliche Knick erlaubt ab Beginn + 20 min eine andere Steigung.
    curve = [wear, np.maximum(wear - KNOT_OFFSET, 0), quality]
    groups = {"torque": [torque, wear * torque], "type": [wear * (quality - type_mean)]}
    return {
        name: np.column_stack([*curve, *(column for group in terms for column in groups[group])])
        for name, terms in VARIANTS.items()
    }


# MODELLBEGINN: Physikalischer Regelkern mit einer stückweisen Verschleißlogistik
def fit_wear_scores(x, y, train, evaluation, names):
    """Lernt Regelgrenzen und trainiert Zusatzmodelle nur ohne Regelalarm."""
    x_train = x.iloc[train]
    x_eval = x.iloc[evaluation]
    baseline = FailureModel().fit(x_train, y[train])
    core = baseline.mechanisms.predict(x_eval).astype(bool)
    if baseline.residual_ is None:
        # Ohne trainierbare Verschleißstufe bleibt nur der Regelalarm aktiv.
        return core, {name: np.full(len(x_eval), -1.) for name in names}

    scores = {}
    if BASE_CONFIG in names:
        scores[BASE_CONFIG] = baseline.residual_.predict_proba(
            baseline._residual_features(x_eval)
        )[:, 1]
    new_configs = [name for name in names if name != BASE_CONFIG]
    if not new_configs:
        return core, scores

    # Zentrierung und Skalierung werden aus den ergänzend bewerteten Trainingsfällen gelernt.
    residual = baseline.mechanisms.predict(x_train) == 0
    x_residual = x_train.iloc[residual]
    torque_mean = x_residual["Torque [Nm]"].mean()
    type_mean = baseline._residual_features(x_residual)[:, 1].mean()
    train_columns = wear_columns(x_residual, baseline, torque_mean, type_mean)
    eval_columns = wear_columns(x_eval, baseline, torque_mean, type_mean)
    for name in new_configs:
        variant, strength = CONFIGS[name]
        learner = make_pipeline(StandardScaler(), LogisticRegression(C=strength))
        learner.fit(train_columns[variant], y[train][residual])
        scores[name] = learner.predict_proba(eval_columns[variant])[:, 1]
    return core, scores


def changes(y, pred, reference, in_wear_band):
    """Zählt neue und verlorene Fälle auf denselben Zeilen gegenüber der Referenz."""
    positives = y == 1
    new_alarms = pred & ~reference
    dropped_alarms = reference & ~pred
    return {
        "new_hits": int(np.sum(new_alarms & positives)),
        "lost_hits": int(np.sum(dropped_alarms & positives)),
        "new_false_alarms": int(np.sum(new_alarms & ~positives)),
        "removed_false_alarms": int(np.sum(dropped_alarms & ~positives)),
        "new_hits_200_219": int(np.sum(new_alarms & positives & in_wear_band)),
        "lost_hits_200_219": int(np.sum(dropped_alarms & positives & in_wear_band)),
    }


def validate(x, y, event):
    """Vergleicht feste Referenz, Schwellentuning und Ablationen im gleichen Split."""
    names = [BASE_CONFIG, *CONFIGS]
    versions = {name: [] for name in (BASE, TUNED, NEW_MODEL, *VARIANTS)}
    configurations = {name: [] for name in names}
    folds = []
    in_wear_band = x["Tool wear [min]"].between(200, 219).to_numpy()

    for seed in SEEDS:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        predictions = {name: np.empty(len(x), dtype=bool) for name in versions}
        candidate_preds = {name: np.empty(len(x), dtype=bool) for name in names}
        for fold, (train, test) in enumerate(outer.split(x, y), 1):
            # C und Schwelle sehen nur innere Validierungsscores der äußeren Trainingszeilen.
            inner = inner_search(x, y, train, seed + 100 * fold, names, scorer=fit_wear_scores)
            chosen = {
                name: min(grid, key=lambda candidate: inner[candidate]["cost"])
                for name, grid in GRID.items()
            }
            chosen[NEW_MODEL] = min(CONFIGS, key=lambda name: inner[name]["cost"])
            chosen[TUNED] = BASE_CONFIG
            core, scores = fit_wear_scores(x, y, train, test, names)
            predictions[BASE][test] = predict(core, scores[BASE_CONFIG], ECONOMIC_THRESHOLD)
            for name in names:
                candidate_preds[name][test] = predict(core, scores[name], inner[name]["threshold"])
            for name, config in chosen.items():
                predictions[name][test] = candidate_preds[config][test]

            # Das Protokoll bewahrt Parameterwahl und Schwelle auch bei Gleichständen.
            folds.append({
                "seed": seed, "fold": fold, "chosen": chosen,
                "inner": {name: {
                    "cost": result["cost"], "threshold": serial_threshold(result["threshold"]),
                } for name, result in inner.items()},
                "outer": {name: counts(y[test], pred[test]) for name, pred in predictions.items()},
            })

        # Jeder Fall liefert je Seed genau eine äußere Vorhersage pro Version.
        reference = predictions[BASE]
        for name, pred in predictions.items():
            versions[name].append({"seed": seed, **counts(y, pred), **changes(y, pred, reference, in_wear_band)})
        for name, pred in candidate_preds.items():
            configurations[name].append({"seed": seed, **counts(y, pred)})
        event(f"Seed {seed} abgeschlossen. Referenz {versions[BASE][-1]['cost']}, "
              f"neue Verschleißkurve {versions[NEW_MODEL][-1]['cost']} Kosteneinheiten.")
    return versions, configurations, folds


def main():
    """Liest modeling.csv ein und speichert Ranglisten und Faltenergebnisse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/verschleisskombinationen"))
    args = parser.parse_args()
    log = []

    def event(message):
        time = timestamp()
        log.append({"time": time, "event": message})
        print(f"{time} {message}", flush=True)

    event("Kompakter Verschleißversuch gestartet.")
    x = read_data(args.modeling, labeled=True)
    y = x.pop(TARGET).to_numpy(dtype=int)
    with threadpool_limits(limits=1):
        versions, configurations, folds = validate(x, y, event)
    ranking = summary_table(versions)
    for row in ranking:
        repeats = versions[row["version"]]
        for key in ("new_hits", "lost_hits", "new_false_alarms", "removed_false_alarms",
                    "new_hits_200_219", "lost_hits_200_219"):
            row[key + "_mittel"] = np.mean([repeat[key] for repeat in repeats])

    # Kurztabellen für die Auswertung. Details enthalten jede innere Wahl und ihre Testkosten.
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking).to_csv(args.ausgabe / "rangliste.csv", index=False)
    pd.DataFrame(summary_table(configurations)).to_csv(args.ausgabe / "konfigurationen.csv", index=False)
    event(f"Vergleich beendet. Niedrigste mittlere Kosten: {ranking[0]['version']} "
          f"mit {ranking[0]['kosten_mittel']:.1f}.")
    details = {
        "time_log": log, "knot_offset_min": KNOT_OFFSET, "grid": GRID,
        "versions": versions, "configurations": configurations, "folds": folds,
    }
    (args.ausgabe / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
