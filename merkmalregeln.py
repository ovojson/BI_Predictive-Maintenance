"""Prüft kurze Merkmalsregeln als Zusatzalarme zum bestehenden Hybrid."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from modell import FALSE_NEGATIVE_COST, SEEDS, TARGET, FailureModel, counts, read_data
from modelloptimierung import summary_table, timestamp
from physik import physical_inputs


REFERENCE = "Hybrid aktuell"
SELECTED = "Hybrid mit Merkmalsregel, innere Auswahl"
# Das Raster wird im jeweiligen Trainingsrest in tatsächliche Schnittpunkte übersetzt.
QUANTILES = (.05, .10, .20, .35, .50, .65, .80, .90, .95, .975, .99)
MODES = {
    f"Bis {length} Bedingung(en), mindestens {support} Treffer": (length, support)
    for length in (1, 2)
    for support in (2, 3, 5)
}


@dataclass(frozen=True)
class Predicate:
    feature: str
    relation: str
    threshold: str | float

    def evaluate(self, data):
        column = data[self.feature].to_numpy()
        if self.relation == "<=":
            return column <= self.threshold
        if self.relation == ">":
            return column > self.threshold
        return column == self.threshold


def features(rows):
    """Ergänzt die vorhandenen Sensorwerte um bereits implementierte Prozessmerkmale."""
    power, gap, speed, stress, quality = physical_inputs(rows)
    return pd.DataFrame({
        "Standzeit [min]": rows["Tool wear [min]"].to_numpy(),
        "Drehmoment [Nm]": rows["Torque [Nm]"].to_numpy(),
        "Drehzahl [U/min]": speed,
        "Lufttemperatur [K]": rows["Air temperature [K]"].to_numpy(),
        "Prozesstemperatur [K]": rows["Process temperature [K]"].to_numpy(),
        "Leistung [W]": power,
        "Temperaturdifferenz [K]": gap,
        "Verschleißlast [Nm min]": stress,
        "Produktvariante": quality,
    })


def predicates(train):
    for feature in train.columns:
        values = train[feature].to_numpy()
        if feature == "Produktvariante":
            for kind in np.unique(values):
                yield Predicate(feature, "=", str(kind))
        else:
            for cut in np.unique(np.quantile(values, QUANTILES)):
                for op in ("<=", ">"):
                    yield Predicate(feature, op, float(cut))


# MODELLBEGINN: Kostenbasierte Suche nach kurzen UND-Regeln
def fit_rules(train, label):
    """Wählt je Rasterpunkt höchstens eine Regel aus dem freigegebenen Trainingsrest."""
    if not len(train) or not np.any(label == 1):
        return dict.fromkeys(MODES)

    conditions = list(predicates(train))
    coverage = np.asarray([item.evaluate(train) for item in conditions], dtype=np.float32)
    positive = coverage[:, label == 1]
    negative = coverage[:, label == 0]

    # Das Skalarprodukt zweier Indikatorvektoren zählt ihre gemeinsame Abdeckung.
    # Die Diagonale entspricht einer Einzelbedingung. Sie hat bei Gleichstand Vorrang.
    tp = positive @ positive.T
    fp = negative @ negative.T
    gain = FALSE_NEGATIVE_COST * tp - fp
    singles = np.arange(len(conditions))
    both_i, both_j = np.triu_indices(len(conditions), k=1)
    choices = {
        1: (singles, singles),
        2: (np.r_[singles, both_i], np.r_[singles, both_j]),
    }
    fitted = {}
    for name, (length, support) in MODES.items():
        i, j = choices[length]
        eligible = (tp[i, j] >= support) & (gain[i, j] > 0)
        if not eligible.any():
            fitted[name] = None
            continue
        best = np.argmax(np.where(eligible, gain[i, j], -np.inf))
        left, right = i[best], j[best]
        rule = (conditions[left],) if left == right else (conditions[left], conditions[right])
        fitted[name] = (rule, int(tp[left, right]), int(fp[left, right]))
    return fitted


def score_fold(x, y, train, test):
    x_train, x_test = x.iloc[train], x.iloc[test]
    base = FailureModel().fit(x_train, y[train])
    reference = base.predict(x_test).astype(bool)
    free_train = base.predict(x_train) == 0
    fitted = fit_rules(features(x_train.iloc[free_train]), y[train][free_train])
    x_values = features(x_test)
    predictions = {REFERENCE: reference}
    reported_rules = {}
    for name, item in fitted.items():
        if item is None:
            predictions[name] = reference
            reported_rules[name] = None
            continue
        rule, tp, fp = item
        match = np.logical_and.reduce([condition.evaluate(x_values) for condition in rule])
        predictions[name] = reference | match
        reported_rules[name] = {
            "bedingungen": [asdict(condition) for condition in rule],
            "training_treffer": tp,
            "training_fehlalarme": fp,
        }
    return predictions, reported_rules


def select_mode(x, y, train, random_state):
    """Vergleicht den Regelsuchraum einschließlich der Variante ohne Zusatzregel."""
    inner_y = y[train]
    costs = dict.fromkeys((REFERENCE, *MODES), 0)
    splitter = StratifiedKFold(3, shuffle=True, random_state=random_state)
    for fit, valid in splitter.split(np.zeros(len(train)), inner_y):
        predicted, _ = score_fold(x, y, train[fit], train[valid])
        for name, values in predicted.items():
            costs[name] += counts(inner_y[valid], values)["cost"]

    # Bei gleichem Validierungsergebnis steht die unveränderte Alarmregel zuerst.
    return min(costs, key=costs.get), costs


def validate(x, y, log):
    results = {name: [] for name in (REFERENCE, SELECTED, *MODES)}
    folds = []
    for seed in SEEDS:
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        by_model = {name: np.empty(len(y), dtype=bool) for name in results}
        for fold, (train, test) in enumerate(outer.split(x, y), start=1):
            winner, inner = select_mode(x, y, train, seed + 100 * fold)
            predicted, rules = score_fold(x, y, train, test)
            for name, values in predicted.items():
                by_model[name][test] = values
            by_model[SELECTED][test] = predicted[winner]
            folds.append({
                "seed": seed, "fold": fold, "auswahl": winner,
                "innere_kosten": inner, "regeln": rules,
                "test_auswahl": counts(y[test], predicted[winner]),
                "test_referenz": counts(y[test], predicted[REFERENCE]),
            })

        for name, predicted in by_model.items():
            added = predicted & ~by_model[REFERENCE]
            results[name].append({
                "seed": seed, **counts(y, predicted),
                "zusätzliche_treffer": int(np.sum(added & (y == 1))),
                "zusätzliche_fehlalarme": int(np.sum(added & (y == 0))),
            })
        log(f"Seed {seed}: Hybrid {results[REFERENCE][-1]['cost']}, "
            f"innere Regelauswahl {results[SELECTED][-1]['cost']} Kosteneinheiten.")
    return results, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/merkmalregeln"))
    args = parser.parse_args()
    timeline = []

    def log(event):
        now = timestamp()
        timeline.append({"time": now, "event": event})
        print(f"{now} {event}", flush=True)

    log("Suche nach statischen Merkmalsregeln gestartet.")
    data = read_data(args.modeling, labeled=True)
    y = data.pop(TARGET).to_numpy(dtype=int)
    with threadpool_limits(limits=1):
        results, folds = validate(data, y, log)
    ranking = summary_table(results)
    for row in ranking:
        runs = results[row["version"]]
        for key in ("zusätzliche_treffer", "zusätzliche_fehlalarme"):
            row[f"{key}_summe"] = sum(repeat[key] for repeat in runs)
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking).to_csv(args.ausgabe / "rangliste.csv", index=False)
    log("Vergleich abgeschlossen und Rangliste geschrieben.")
    details = {"time_log": timeline, "raster": MODES, "quantile": QUANTILES,
               "results": results, "folds": folds}
    (args.ausgabe / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
