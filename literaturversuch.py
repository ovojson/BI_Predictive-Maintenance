"""Prüft kleine Regelbäume und gelernte Abweichungen vom Betriebspunkt."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits

from modell import SEEDS, TARGET, FailureModel, counts, read_data
from modelloptimierung import inner_search, predict, serial_threshold, summary_table, timestamp
from physik import physical_inputs
from verschleissversuch import changes


BASE = "Hybrid aktuell"
SELECTED = "Hybrid + innere Baumauswahl"
# Tiefe und Blattgröße begrenzen die Komplexität bei wenigen verbleibenden Ausfällen.
FEATURE_GROUPS = {
    "Kühlung": ["Luft", "Temperaturabstand", "Drehzahl", "Standzeit"],
    "Belastung": ["Standzeit", "Drehmoment", "Leistung", "Typ M", "Typ H"],
    "Betriebspunkt": ["Standzeit", "Drehzahlabweichung", "Temperaturabweichung", "Typ M", "Typ H"],
    "Kombiniert": ["Standzeit", "Drehmoment", "Temperaturabstand", "Leistung",
                   "Drehzahlabweichung", "Temperaturabweichung", "Typ M", "Typ H"],
}
GRID = {
    family: {
        f"{family} Tiefe {depth} Blatt {leaf}": (depth, leaf)
        for depth in (1, 2, 3) for leaf in (30, 75)
    }
    for family in FEATURE_GROUPS
}
CONFIGS = {name: (family, *params) for family, grid in GRID.items() for name, params in grid.items()}


def measurements(frame):
    """Bildet Merkmale innerhalb eines Vorgangs aus seinen CSV-Spalten."""
    power, gap, speed, _, kind = physical_inputs(frame)
    return pd.DataFrame({
        "Standzeit": frame["Tool wear [min]"],
        "Luft": frame["Air temperature [K]"],
        "Temperaturabstand": gap,
        "Drehzahl": speed,
        "Drehmoment": frame["Torque [Nm]"],
        "Leistung": power,
        "Typ M": kind == "M",
        "Typ H": kind == "H",
    }, index=frame.index)


def add_deviations(train, test, healthy):
    """Lernt typische Betriebspunkte aus regulären Trainingsvorgängen."""
    points = (
        ("Drehzahlabweichung", "Drehzahl", ["Drehmoment", "Typ M", "Typ H"]),
        ("Temperaturabweichung", "Temperaturabstand", ["Luft", "Drehzahl", "Typ M", "Typ H"]),
    )
    for feature, target, context in points:
        normal = train.loc[healthy]
        trend = make_pipeline(
            StandardScaler(), PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(), Ridge(alpha=1.0),
        )
        trend.fit(normal[context], normal[target])
        residual = normal[target] - trend.predict(normal[context])
        # Zentrierung und Streuung werden auf den normalen Trainingsfällen geschätzt.
        center = np.median(residual)
        scale = np.median(np.abs(residual - center)) or 1.0
        for part in (train, test):
            part[feature] = (part[target] - trend.predict(part[context]) - center) / scale


# MODELLBEGINN: Kleine Regelbäume für die nach dem Hybrid noch freigegebenen Fälle
def fit_scores(x, y, train, test, names, *, rules=None):
    x_train, x_test = x.iloc[train], x.iloc[test]
    hybrid = FailureModel().fit(x_train, y[train])
    admitted = hybrid.predict(x_train) == 0
    reference = hybrid.predict(x_test).astype(bool)
    labels = y[train][admitted]
    if np.unique(labels).size < 2:
        return reference, {name: np.zeros(len(test)) for name in names}
    train_features = measurements(x_train.iloc[admitted])
    test_features = measurements(x_test)
    add_deviations(train_features, test_features, labels == 0)
    scores = {}
    for name in names:
        family, depth, leaf = CONFIGS[name]
        columns = FEATURE_GROUPS[family]
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=57)
        tree.fit(train_features[columns], labels)
        scores[name] = tree.predict_proba(test_features[columns])[:, 1]
        if rules is not None:
            rules[name] = export_text(tree, feature_names=columns)
    return reference, scores


def validate(x, y, event):
    """Wählt Konfiguration und Schwelle innen und bewertet Zusatzalarme außen."""
    results = {name: [] for name in (BASE, SELECTED, *GRID, *CONFIGS)}
    folds = []
    wear_band = x["Tool wear [min]"].between(200, 219).to_numpy()
    names = list(CONFIGS)
    for seed in SEEDS:
        predictions = {name: np.empty(len(y), dtype=bool) for name in results}
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(split.split(x, y), 1):
            inner = inner_search(x, y, train, seed + 100 * fold, names, scorer=fit_scores)
            rules = {}
            reference, scores = fit_scores(x, y, train, test, names, rules=rules)
            predictions[BASE][test] = reference
            for name in names:
                predictions[name][test] = predict(reference, scores[name], inner[name]["threshold"])
            choices = {group: min(configs, key=lambda name: inner[name]["cost"]) for group, configs in GRID.items()}
            choices[SELECTED] = min(names, key=lambda name: inner[name]["cost"])
            for group, winner in choices.items():
                predictions[group][test] = predictions[winner][test]
            selected = choices[SELECTED]
            folds.append({
                "seed": seed, "fold": fold, "selection": choices,
                "thresholds": {name: serial_threshold(info["threshold"]) for name, info in inner.items()},
                "selected_tree": rules.get(selected),
            })
        reference = predictions[BASE]
        for name, pred in predictions.items():
            results[name].append({"seed": seed, **counts(y, pred), **changes(y, pred, reference, wear_band)})
        event(f"Seed {seed}: Referenz {results[BASE][-1]['cost']}, Baumauswahl {results[SELECTED][-1]['cost']}.")
    return results, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/literaturversuch"))
    args = parser.parse_args()
    log = []

    def event(message):
        now = timestamp()
        log.append({"time": now, "event": message})
        print(f"{now} {message}", flush=True)

    event("Literaturversuch gestartet. Eingabedaten aus modeling.csv.")
    x = read_data(args.modeling, labeled=True)
    y = x.pop(TARGET).to_numpy(dtype=int)
    with threadpool_limits(limits=1):
        results, folds = validate(x, y, event)

    # Die Ranglisten trennen vorab festgelegte Modellfamilien von einzelnen Parametervarianten.
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    for file, names in (
        ("rangliste.csv", [BASE, SELECTED, *GRID]),
        ("konfigurationen.csv", list(CONFIGS)),
    ):
        table = summary_table({name: results[name] for name in names})
        for row in table:
            repeats = results[row["version"]]
            for key in ("new_hits", "new_false_alarms"):
                row[key + "_mittel"] = np.mean([repeat[key] for repeat in repeats])
        pd.DataFrame(table).to_csv(args.ausgabe / file, index=False)
    event("Literaturversuch abgeschlossen.")
    (args.ausgabe / "details.json").write_text(json.dumps({
        "time_log": log, "grid": CONFIGS, "results": results, "folds": folds,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
