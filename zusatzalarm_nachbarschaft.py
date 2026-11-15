"""Testet lokale Zusatzalarme, während der bisherige Hybridalarm aktiv bleibt."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from modell import SEEDS, TARGET, FailureModel, counts, read_data
from modelloptimierung import inner_search, predict, serial_threshold, summary_table, timestamp


WEIGHTS = (1., 3., 6.)
NEIGHBORS = (8, 24, 64)
FEATURES = ("Sensorwerte", "Prozessmerkmale")
# Die innere CV wählt Merkmalsgruppe, Gewicht der Standzeit und Anzahl der Nachbarn.
SETTINGS = {
    f"{feature} ×{weight:g} k={k}": (feature, weight, k)
    for feature in FEATURES
    for weight in WEIGHTS
    for k in NEIGHBORS
}


def feature_arrays(rows):
    """Baut aus jeder Zeile zwei Darstellungen des Betriebspunkts auf."""
    wear = rows["Tool wear [min]"]
    torque = rows["Torque [Nm]"]
    speed = rows["Rotational speed [rpm]"]
    air = rows["Air temperature [K]"]
    process = rows["Process temperature [K]"]
    quality = rows["Type"].map({"L": 0, "M": 1, "H": 2})
    # Gleiche Standzeit und Variante, vergleichbarer Betriebspunkt und Wärmehaushalt.
    return {
        "Sensorwerte": np.column_stack((wear, torque, speed, air, process - air, quality)),
        "Prozessmerkmale": np.column_stack((wear, torque, torque * speed, air, process - air, quality)),
    }


def neighborhoods(x, y, train, evaluation, names):
    """Schätzt die lokale Ausfallquote bei Vorgängen ohne Hybridalarm."""
    train_frame, eval_frame = x.iloc[train], x.iloc[evaluation]
    baseline = FailureModel().fit(train_frame, y[train])
    kept = baseline.predict(train_frame) == 0
    alarm = baseline.predict(eval_frame).astype(bool)
    # Ausschließlich die bisher freigegebenen Fälle erhalten Nachbarschaftsscores.
    train_arrays = feature_arrays(train_frame.iloc[kept])
    eval_arrays = feature_arrays(eval_frame.iloc[~alarm])
    labels = y[train][kept]
    requested = set(SETTINGS[name] for name in names)
    scores = {}
    for feature, train_rows in train_arrays.items():
        # Die Skala wird aus den freigegebenen Trainingsfällen gelernt.
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_rows)
        eval_data = scaler.transform(eval_arrays[feature])
        train_wear = train_data[:, 0].copy()
        eval_wear = eval_data[:, 0].copy()
        for weight in WEIGHTS:
            train_data[:, 0] = weight * train_wear
            eval_data[:, 0] = weight * eval_wear
            distances, indices = cKDTree(train_data).query(eval_data, k=max(NEIGHBORS))
            local = labels[indices]
            for k in NEIGHBORS:
                if (feature, weight, k) not in requested:
                    continue
                # Nahe Nachbarn zählen stärker als weiter entfernte Nachbarn.
                bound = np.maximum(distances[:, [k - 1]], 1e-12)
                kernel = np.exp(-2 * (distances[:, :k] / bound) ** 2)
                p = np.full(len(eval_frame), -1.)
                p[~alarm] = (kernel * local[:, :k]).sum(axis=1) / kernel.sum(axis=1)
                scores[f"{feature} ×{weight:g} k={k}"] = p
    return alarm, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/zusatzalarm_nachbarschaft"))
    args = parser.parse_args()
    log = []

    def event(message):
        time = timestamp()
        log.append({"time": time, "event": message})
        print(f"{time} {message}", flush=True)

    event("Nachbarschaftstest gestartet.")
    x = read_data(args.modeling, labeled=True)
    y = x.pop(TARGET).to_numpy(dtype=int)
    results = {name: [] for name in ("Hybrid", "KNN innere Auswahl", *SETTINGS)}
    picks = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            predictions = {name: np.empty(len(y), dtype=bool) for name in results}
            outer = StratifiedKFold(5, shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(outer.split(x, y), 1):
                # Parameter und Zusatzschwelle werden ohne die äußeren Testfälle gewählt.
                inner = inner_search(x, y, train, seed + 100 * fold, list(SETTINGS), scorer=neighborhoods)
                winner = min(SETTINGS, key=lambda name: inner[name]["cost"])
                core, scores = neighborhoods(x, y, train, test, list(SETTINGS))
                predictions["Hybrid"][test] = core
                for name, result in inner.items():
                    predictions[name][test] = predict(core, scores[name], result["threshold"])
                predictions["KNN innere Auswahl"][test] = predictions[winner][test]
                picks.append({
                    "seed": seed, "fold": fold, "configuration": winner,
                    "inner_cost": inner[winner]["cost"],
                    "threshold": serial_threshold(inner[winner]["threshold"]),
                })
            for name, prediction in predictions.items():
                results[name].append({"seed": seed, **counts(y, prediction)})
            event(f"Seed {seed}: Hybrid {results['Hybrid'][-1]['cost']}, "
                  f"mit ausgewähltem Zusatzalarm {results['KNN innere Auswahl'][-1]['cost']}.")
    ranked = summary_table(results)
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_csv(args.ausgabe / "rangliste.csv", index=False)
    event(f"Fertig. Mittlere Kosten mit innerer Auswahl: "
          f"{np.mean([row['cost'] for row in results['KNN innere Auswahl']]):.1f}.")
    details = {"time_log": log, "ranking": ranked, "results": results,
               "picks": picks, "configurations": SETTINGS}
    (args.ausgabe / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
