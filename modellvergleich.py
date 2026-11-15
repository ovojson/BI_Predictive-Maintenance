"""Vergleicht Ausfallmodelle auf modeling.csv bei Kosten von FP + 6 × FN."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from modell import FailureModel


TARGET = "Machine failure"


def feature_sets(data):
    # Rohwerte und physikalisch abgeleitete Merkmale für den Baumvergleich bereitstellen.
    quality = data["Type"].map({"L": 0, "M": 1, "H": 2})
    air = data["Air temperature [K]"]
    process = data["Process temperature [K]"]
    speed = data["Rotational speed [rpm]"]
    torque = data["Torque [Nm]"]
    wear = data["Tool wear [min]"]
    binary_gap = process - air
    rounded_gap = binary_gap.round(1)
    # Erhält die Richtung der Rundungsabweichung bei interner float32-Konvertierung.
    precision_gap = rounded_gap + 0.001 * np.sign(binary_gap - rounded_gap)
    load = wear * torque
    physical = np.column_stack([
        quality, speed, wear, rounded_gap, speed * torque * (np.pi / 30),
        load, load - 1000 * quality,
    ])
    raw = np.column_stack([quality, air, process, speed, torque, wear])
    precision = physical.copy()
    precision[:, 3] = precision_gap
    large = np.column_stack([physical, air, process, torque])
    return {"raw": raw, "physical": physical, "precision": precision, "large": large}


def outcome(y, alarm):
    # Gezählt werden Fehlalarme und übersehene Ausfälle mit dem Kostenverhältnis 1 zu 6.
    false_alarm = int(((y == 0) & alarm).sum())
    missed = int(((y == 1) & ~alarm).sum())
    return {"cost": false_alarm + 6 * missed, "FP": false_alarm, "FN": missed}


def cost_threshold(y, score):
    # Sortierte Scores zeigen den Kostenverlauf vom ersten bis zum letzten Alarm.
    order = np.argsort(-score)
    ranked = score[order]
    delta_cost = np.where(y[order] == 1, -6, 1)
    group_end = np.r_[np.flatnonzero(np.diff(ranked)), len(y) - 1]
    levels = ranked[group_end]
    cost = np.r_[0, np.cumsum(delta_cost)[group_end]] + 6 * int(y.sum())
    # Bei Kostengleichstand wird die breiteste konstante Schwellenregion gewählt.
    widths = -np.diff(np.r_[1.0, levels, 0.0])
    tied = np.flatnonzero(cost == cost.min())
    best = int(tied[np.argmax(widths[tied])])
    if best == 0:
        return float(np.nextafter(levels[0], np.inf)), int(cost[best])
    if best == len(levels):
        return float(levels[-1]), int(cost[best])
    return float((levels[best - 1] + levels[best]) / 2), int(cost[best])


def precision_models(threads, families):
    # Vergleich gleicher Modellfamilien mit gerundeter und genauerer Temperaturdifferenz.
    available = {
        # MODELLBEGINN: Random Forest mit Rohmerkmalen und Temperaturvarianten
        "RF raw": (
            "rf", "raw",
            RandomForestClassifier(n_estimators=90, random_state=57, n_jobs=threads),
        ),
        "RF physics": (
            "rf", "physical",
            RandomForestClassifier(
                n_estimators=90, random_state=57, n_jobs=threads,
                max_features=1.0, min_samples_leaf=2,
            ),
        ),
        "RF precision": (
            "rf", "precision",
            RandomForestClassifier(
                n_estimators=90, random_state=57, n_jobs=threads,
                max_features=1.0, min_samples_leaf=2,
            ),
        ),
        # MODELLBEGINN: Extra Trees mit Temperaturvarianten
        "ET physics": (
            "et", "physical",
            ExtraTreesClassifier(n_estimators=120, random_state=57, n_jobs=threads),
        ),
        "ET precision": (
            "et", "precision",
            ExtraTreesClassifier(n_estimators=120, random_state=57, n_jobs=threads),
        ),
        # MODELLBEGINN: Histogram Gradient Boosting mit Temperaturvarianten
        "HGB physics": (
            "hgb", "physical",
            HistGradientBoostingClassifier(
                max_iter=175, max_depth=3, min_samples_leaf=2, random_state=57,
            ),
        ),
        "HGB precision": (
            "hgb", "precision",
            HistGradientBoostingClassifier(
                max_iter=175, max_depth=3, min_samples_leaf=2, random_state=57,
            ),
        ),
    }
    candidates = {
        name: (features, estimator)
        for name, (family, features, estimator) in available.items()
        if family in families
    }
    # MODELLBEGINN: Ensemble aus den präzisionserhaltenden Baumvarianten
    members = [name for name in candidates if name.endswith(" precision")]
    blends = {"Ensemble precision": members} if len(members) > 1 else {}
    return candidates, blends


def extended_models(threads):
    # Die Basismodelle werden für konkrete Parameterkombinationen kopiert.
    rf = RandomForestClassifier(n_estimators=90, n_jobs=threads, random_state=57)
    et = ExtraTreesClassifier(n_estimators=120, n_jobs=threads, random_state=57)
    hist = HistGradientBoostingClassifier(max_iter=175, random_state=57)
    tree = DecisionTreeClassifier(random_state=57)
    weight = {0: 1, 1: 6}
    candidates = {
        # MODELLBEGINN: Random Forest (erweiterter Vergleich)
        "RF large leaf=1 features=sqrt": ("large", rf),
        "RF small leaf=1 features=all": (
            "physical", clone(rf).set_params(max_features=1.0),
        ),
        "RF small leaf=2 features=all": (
            "physical", clone(rf).set_params(max_features=1.0, min_samples_leaf=2),
        ),
        "RF small leaf=5 features=all": (
            "physical", clone(rf).set_params(max_features=1.0, min_samples_leaf=5),
        ),
        "RF small leaf=1 features=.65": (
            "physical", clone(rf).set_params(max_features=.65),
        ),
        "RF small leaf=2 features=.65 cost weight": (
            "physical", clone(rf).set_params(max_features=.65, min_samples_leaf=2, class_weight=weight),
        ),
        # MODELLBEGINN: Extra Trees (erweiterter Vergleich)
        "ET small leaf=1 features=all": ("physical", clone(et).set_params(max_features=1.0)),
        "ET small leaf=2 features=all": (
            "physical", clone(et).set_params(max_features=1.0, min_samples_leaf=2),
        ),
        "ET small leaf=2 features=.65 cost weight": (
            "physical", clone(et).set_params(max_features=.65, min_samples_leaf=2, class_weight=weight),
        ),
        # MODELLBEGINN: Histogram Gradient Boosting (erweiterter Vergleich)
        "HGB small leaf=2 depth=3": (
            "physical", clone(hist).set_params(max_depth=3, min_samples_leaf=2),
        ),
        "HGB small leaf=5 depth=5": (
            "physical", clone(hist).set_params(max_depth=5, min_samples_leaf=5),
        ),
        "HGB large leaf=5 depth=5": (
            "large", clone(hist).set_params(max_depth=5, min_samples_leaf=5),
        ),
        # MODELLBEGINN: Einzelner Entscheidungsbaum (erweiterter Vergleich)
        "Tree small leaf=5": (
            "physical", clone(tree).set_params(min_samples_leaf=5),
        ),
        "Tree small leaf=10 cost weight": (
            "physical", clone(tree).set_params(min_samples_leaf=10, class_weight=weight),
        ),
    }
    # MODELLBEGINN: Baum-Ensembles (Mittelwerte der Modell-Scores)
    blends = {
        "mean RF all": [
            "RF small leaf=1 features=all",
            "RF small leaf=2 features=all",
            "RF small leaf=5 features=all",
        ],
        "mean RF/ET/HGB": [
            "RF small leaf=2 features=all",
            "ET small leaf=1 features=all",
            "HGB small leaf=2 depth=3",
        ],
    }
    return candidates, blends


def catboost_model(threads):
    # MODELLBEGINN: CatBoost (lokale Rekonstruktion)
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=200, depth=5, learning_rate=0.05, random_seed=57,
        thread_count=threads, verbose=False, allow_writing_files=False,
    )


def lightgbm_model(threads):
    # MODELLBEGINN: LightGBM (lokale Rekonstruktion)
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200, max_depth=5, num_leaves=31, learning_rate=0.05,
        random_state=57, n_jobs=threads, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def all_models(threads):
    # Alle lokalen Modellfamilien und die rekonstruierbaren Cloud-Kandidaten zusammenführen.
    candidates, blends = extended_models(threads)
    precision, precision_blends = precision_models(threads, ["rf", "et", "hgb"])
    candidates.update(precision)
    blends.update(precision_blends)

    # MODELLBEGINN: Random Forest mit der früheren festen Alarmschwelle
    # Dazu wird die lokale RF-Konfiguration mit physikalischen Zusatzmerkmalen verwendet.
    candidates["RF physikalisch (Schwelle 0.399)"] = precision["RF physics"]
    thresholds = {"RF physikalisch (Schwelle 0.399)": 0.399}

    # Die optionalen Modellbibliotheken werden erst nach der Modellauswahl importiert.
    candidates["CatBoost physikalisch"] = ("physical", partial(catboost_model, threads))
    candidates["LightGBM physikalisch"] = ("physical", partial(lightgbm_model, threads))

    # MODELLBEGINN: CatBoost + LightGBM (arithmetisches Mittel, lokal rekonstruiert)
    blends["CatBoost + LightGBM"] = ["CatBoost physikalisch", "LightGBM physikalisch"]

    # MODELLBEGINN: Lokales Hybridmodell und die Varianten aus modell.py
    decisions = {
        "Hybridmodell": FailureModel,
        "Physikalisches Regelmodell": partial(FailureModel, wear_alarm=False),
        "Regelmodell mit Temperatur-Rundung": partial(
            FailureModel, wear_alarm=False, round_gap=True,
        ),
        "Regelmodell mit separaten Belastungsgrenzen": partial(
            FailureModel, wear_alarm=False, pool_stress=False,
        ),
    }
    return candidates, blends, decisions, thresholds


def score_split(model, x, y, train, test):
    # Pro Split eine neue Modellkopie trainieren und Ausfallwahrscheinlichkeiten ausgeben.
    return clone(model).fit(x.iloc[train], y[train]).predict_proba(x.iloc[test])[:, 1]


def decision_split(model_factory, data, y, train, test):
    # Regel- und Hybridmodelle liefern fest definierte Alarmentscheidungen.
    return model_factory().fit(data.iloc[train], y[train]).predict(data.iloc[test])


def fold_scores(predict, y, train, test, inner_split):
    # Innere Scores dienen der Auswahl. Äußere Scores dienen der abschließenden Bewertung.
    oof = np.empty(len(train))
    for fit, tune in inner_split:
        oof[tune] = predict(y, train[fit], train[tune])
    return oof, predict(y, train, test)


def validate(data, y, candidates, blends, seed, decisions=None, thresholds=None):
    # Für alle Kandidaten dieselben äußeren Test- und inneren Auswahlfalten verwenden.
    decisions = decisions or {}
    thresholds = thresholds or {}
    x_sets = {
        name: pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
        for name, values in feature_sets(data).items()
    }
    predictors = {}
    for name, (features, model) in candidates.items():
        predictors[name] = partial(score_split, model, x_sets[features])
    for name, model_factory in decisions.items():
        predictors[name] = partial(decision_split, model_factory, data)
    names = [*predictors, *blends]
    alarm = {name: np.zeros(len(y), dtype=bool) for name in names}
    alarm["Innere Modellauswahl"] = np.zeros(len(y), dtype=bool)
    rf_options = [name for name in names if name.startswith("RF")]
    if len(rf_options) > 1:
        alarm["Innere RF-Auswahl"] = np.zeros(len(y), dtype=bool)
    details = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(data, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=100 + fold)
        inner_split = list(inner.split(data.iloc[train], y[train]))
        scores = {
            name: fold_scores(predict, y, train, test, inner_split)
            for name, predict in predictors.items()
        }
        for blend, members in blends.items():
            # Ensemble-Scores sind Mittelwerte der Einzelmodelle innerhalb derselben Falte.
            scores[blend] = (
                np.mean([scores[name][0] for name in members], axis=0),
                np.mean([scores[name][1] for name in members], axis=0),
            )

        report = {}
        for name, (inner_score, test_score) in scores.items():
            # Nur Kandidaten ohne feste Entscheidungslogik erhalten eine gelernte Schwelle.
            threshold = 0.5 if name in decisions else thresholds.get(name)
            if threshold is None:
                threshold, inner_cost = cost_threshold(y[train], inner_score)
            else:
                inner_cost = outcome(y[train], inner_score >= threshold)["cost"]
            alarm[name][test] = test_score >= threshold
            report[name] = {
                "threshold": None if name in decisions else threshold,
                "inner_cost": inner_cost,
                **outcome(y[test], alarm[name][test]),
            }
        # Die Modellauswahl erfolgt ausschließlich anhand der inneren Kosten.
        winner = min(report, key=lambda name: report[name]["inner_cost"])
        alarm["Innere Modellauswahl"][test] = alarm[winner][test]
        if "Innere RF-Auswahl" in alarm:
            rf_winner = min(rf_options, key=lambda name: report[name]["inner_cost"])
            alarm["Innere RF-Auswahl"][test] = alarm[rf_winner][test]
        details.append({"fold": fold, "inner_winner": winner, "models": report})
        cost_summary = ", ".join(f"{name}={values['cost']}" for name, values in report.items())
        print(f"Fold {fold}/5. Innere Auswahl {winner}. {cost_summary}", file=sys.stderr, flush=True)
    summary = {name: outcome(y, predicted) for name, predicted in alarm.items()}
    return {"outer_seed": seed, "rows": len(y), "summary": summary, "folds": details}


def main():
    # Optionen erlauben den Gesamtvergleich, einen Einzeltest oder eine Modellliste.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modeling", type=Path, help="Pfad zur gelabelten modeling.csv")
    parser.add_argument("--family", action="append", choices=["rf", "et", "hgb"])
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--erweitert", action="store_true", help="Erweiterter Vergleich mit 16 Varianten")
    comparison.add_argument("--alle", action="store_true", help="Alle lokalen Modelle und rekonstruierte Cloud-Varianten")
    parser.add_argument("--modell", help="Modellname aus dem gewählten Vergleich einzeln testen")
    parser.add_argument("--liste", action="store_true", help="Modellnamen anzeigen und beenden")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--outer-seed", type=int, default=2006)
    parser.add_argument("--json", type=Path, help="Optionaler Pfad für die Ergebnisdatei")
    args = parser.parse_args()
    if (args.erweitert or args.alle) and args.family:
        parser.error("--family gehört zum Präzisionsvergleich ohne --erweitert oder --alle.")

    decisions, thresholds = {}, {}
    if args.alle:
        candidates, blends, decisions, thresholds = all_models(args.threads)
    elif args.erweitert:
        candidates, blends = extended_models(args.threads)
    else:
        candidates, blends = precision_models(args.threads, args.family or ["rf", "et", "hgb"])
    if args.liste:
        print("\n".join([*candidates, *blends, *decisions]))
        return
    if args.modell:
        if args.modell in candidates:
            candidates, blends, decisions = {args.modell: candidates[args.modell]}, {}, {}
        elif args.modell in blends:
            candidates = {name: candidates[name] for name in blends[args.modell]}
            blends, decisions = {args.modell: blends[args.modell]}, {}
        elif args.modell in decisions:
            candidates, blends, decisions = {}, {}, {args.modell: decisions[args.modell]}
        else:
            parser.error(f"Unbekanntes Modell {args.modell!r}. Verfügbare Namen zeigt --liste.")

    # Zusätzliche Bibliotheken erst laden, wenn ein ausgewähltes Modell sie benötigt.
    try:
        candidates = {
            name: (features, estimator() if isinstance(estimator, partial) else estimator)
            for name, (features, estimator) in candidates.items()
        }
    except ImportError as error:
        parser.error(f"{error}. Zusatzpakete installieren mit python -m pip install -r requirements-vergleich.txt")

    # Training und Validierung verwenden nur die gelabelte Modellierungsdatei.
    data = pd.read_csv(args.modeling, float_precision="round_trip")
    y = data.pop(TARGET).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        parser.error("Machine failure muss 0 und 1 enthalten.")
    with threadpool_limits(limits=min(args.threads, 2) if args.erweitert or args.alle else 1):
        report = validate(data, y, candidates, blends, args.outer_seed, decisions, thresholds)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(output + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
