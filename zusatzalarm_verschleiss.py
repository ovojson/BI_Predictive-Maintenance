"""Prüft zusätzliche Alarme anhand geglätteter Verschleißraten."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from modell import SEEDS, TARGET, FailureModel, counts, read_data
from modelloptimierung import inner_search, predict, serial_threshold, summary_table, timestamp
from physik import TYPE_LEVEL
from verschleissversuch import changes


REFERENCE = "Hybrid aktuell"
EXTENDED = "Hybrid mit ergänzender Risikoschätzung"
# Der innere Vergleich wählt Glättungsbreite und Berücksichtigung der Produktstufe.
MODELS = {
    f"{family} · {width:g} min": (width, use_type)
    for family, use_type in (("Standzeit", False), ("Standzeit und Typ", True))
    for width in (1.5, 3, 6, 12)
}
PRIOR_WEIGHT = 8
OTHER_TYPE_WEIGHT = .5


# MODELLBEGINN: Geglättete empirische Ausfallrate für die bislang freigegebenen Fälle
def fit_scores(x, y, train, test, names):
    """Schätzt das lokale Risiko aus Standzeitwerten der jeweiligen Trainingsfalte."""
    x_train, x_test = x.iloc[train], x.iloc[test]
    hybrid = FailureModel().fit(x_train, y[train])
    train_released = hybrid.predict(x_train) == 0
    test_alarm = hybrid.predict(x_test).astype(bool)
    if not train_released.any():
        return test_alarm, {name: np.zeros(len(test)) for name in names}

    # Gleiche Standzeiten und Produkttypen werden zu Fall- und Ausfallzahlen zusammengefasst.
    remaining = x_train.iloc[train_released]
    remaining_y = y[train][train_released]
    grouped = remaining.assign(failure=remaining_y).groupby(["Tool wear [min]", "Type"])["failure"]
    evidence = grouped.agg(["size", "sum"]).reset_index()
    wear = evidence["Tool wear [min]"].to_numpy(dtype=float)
    quality = evidence["Type"].map(TYPE_LEVEL).to_numpy()
    test_wear, wear_index = np.unique(x_test["Tool wear [min]"], return_inverse=True)
    test_quality = x_test["Type"].map(TYPE_LEVEL).to_numpy()
    base_rate = remaining_y.mean()

    # Nahe Standzeiten tragen mehr bei. In der Typvariante zählt Evidenz anderer Typen halb.
    # Die globale Rate stabilisiert Schätzungen bei wenig lokaler Evidenz.
    difference = test_wear[:, None] - wear[None, :]
    same_type = np.arange(len(TYPE_LEVEL))[:, None] == quality[None, :]
    type_weights = np.where(same_type, 1., OTHER_TYPE_WEIGHT)
    scores = {}
    for name in names:
        width, use_type = MODELS[name]
        proximity = np.exp(-.5 * (difference / width) ** 2)
        if use_type:
            positives = proximity @ (type_weights * evidence["sum"].to_numpy()).T
            support = proximity @ (type_weights * evidence["size"].to_numpy()).T
            scores[name] = ((positives + PRIOR_WEIGHT * base_rate) / (support + PRIOR_WEIGHT))[
                wear_index, test_quality
            ]
        else:
            positives = proximity @ evidence["sum"].to_numpy()
            support = proximity @ evidence["size"].to_numpy()
            scores[name] = ((positives + PRIOR_WEIGHT * base_rate) / (support + PRIOR_WEIGHT))[wear_index]
    return test_alarm, scores


def validate(x, y, event):
    """Vergleicht die Zusatzalarme und die Referenz auf identischen äußeren Falten."""
    results = {name: [] for name in (REFERENCE, EXTENDED, *MODELS)}
    records = []
    band = x["Tool wear [min]"].between(200, 219).to_numpy()
    names = list(MODELS)
    for seed in SEEDS:
        predictions = {name: np.empty(len(y), dtype=bool) for name in results}
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(outer.split(x, y), start=1):
            # Ohne Kostenvorteil innen bleibt die Schwelle auf unendlich und ergänzt keinen Alarm.
            tuning = inner_search(x, y, train, seed + 100 * fold, names, scorer=fit_scores)
            reference, score = fit_scores(x, y, train, test, names)
            winner = min(names, key=lambda name: tuning[name]["cost"])
            predictions[REFERENCE][test] = reference
            for name in names:
                predictions[name][test] = predict(reference, score[name], tuning[name]["threshold"])
            predictions[EXTENDED][test] = predictions[winner][test]
            records.append({
                "seed": seed, "fold": fold, "auswahl": winner,
                "schwellen": {name: serial_threshold(item["threshold"]) for name, item in tuning.items()},
            })
        reference = predictions[REFERENCE]
        for name, pred in predictions.items():
            results[name].append({"seed": seed, **counts(y, pred), **changes(y, pred, reference, band)})
        event(f"Seed {seed}: Referenz {results[REFERENCE][-1]['cost']}, "
              f"Zusatzalarme {results[EXTENDED][-1]['cost']} Kosteneinheiten.")
    return results, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/zusatzalarm_verschleiss"))
    args = parser.parse_args()
    log = []

    def event(message):
        moment = timestamp()
        log.append({"time": moment, "event": message})
        print(f"{moment} {message}", flush=True)

    # Die Vorhersagedatei der Challenge wird für diesen Vergleich nicht eingelesen.
    event("Versuch gestartet.")
    x = read_data(args.modeling, labeled=True)
    y = x.pop(TARGET).to_numpy(dtype=int)
    with threadpool_limits(limits=1):
        results, records = validate(x, y, event)
    ranking = summary_table(results)
    for row in ranking:
        runs = results[row["version"]]
        for key in ("new_hits", "new_false_alarms", "new_hits_200_219"):
            row[f"{key}_mittel"] = np.mean([run[key] for run in runs])
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking).to_csv(args.ausgabe / "rangliste.csv", index=False)
    event("Vergleich abgeschlossen und Rangliste geschrieben.")
    details = {"time_log": log, "results": results, "folds": records, "grid": MODELS}
    (args.ausgabe / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
