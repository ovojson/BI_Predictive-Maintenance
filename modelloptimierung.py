"""Vergleicht Schwellen und Restmodelle mit wiederholter verschachtelter CV."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from grenzabstaende import distance_features
from modell import (
    ECONOMIC_THRESHOLD, FALSE_NEGATIVE_COST, SEEDS, TARGET, FailureModel, counts,
    read_data,
)

BASE = "Hybrid aktuell"
CORE = "Physikalische Regeln"
TUNED = "Hybrid mit optimierter Schwelle"
LOGIT = "Logistik mit Grenzabständen"
BOOST = "Gradient Boosting mit Grenzabständen"
VERSIONS = (CORE, BASE, TUNED, LOGIT, BOOST)
TARGET_RECALLS = (90, 95, 100)
FINAL_SEED = 4701
# Suchraum für die innere Validierung. Die äußeren Falten werden erst zur Bewertung genutzt.
GRIDS = {
    TUNED: {
        "Verschleißlogistik C=1": {},
    },
    LOGIT: {
        f"Logistik {features} C={strength:g}": {"features": features, "C": strength}
        for features in ("Belastung", "Alle", "Grenznähe")
        for strength in (.01, .1, 1., 10.)
    },
    BOOST: {
        f"HGB {features} Tiefe={depth} Blatt={leaf}": {
            "features": features, "max_depth": depth, "min_samples_leaf": leaf,
        }
        for features in ("Belastung", "Alle")
        for depth in (1, 2, 3)
        for leaf in (20, 60)
    },
}
CONFIGURATIONS = {
    name: (version, params)
    for version, grid in GRIDS.items() for name, params in grid.items()
}
MARGIN_COLUMNS = [
    "power_nearest_kw", "cooling_gap_k", "cooling_speed_100rpm",
    "cooling_combined", "stress_1000",
]


def timestamp():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def feature_variants(frame, baseline):
    """Kombiniert Verschleiß und Produkttyp mit drei Varianten der Grenzabstände."""
    wear = baseline._residual_features(frame)
    distance = distance_features(frame, baseline.mechanisms)
    near = distance[["power_nearest_kw", "cooling_combined", "stress_1000"]]
    return {
        "Belastung": np.column_stack((wear, distance["stress_1000"])),
        "Alle": np.column_stack((wear, distance[MARGIN_COLUMNS])),
        "Grenznähe": np.column_stack((wear, np.exp(-near.clip(lower=0)))),
    }


def make_residual(version, params):
    if version == LOGIT:
        # MODELLBEGINN: Regularisierte Logistik mit Grenzabständen
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"]))

    # MODELLBEGINN: Histogram Gradient Boosting mit Grenzabständen
    return HistGradientBoostingClassifier(
        max_iter=90, learning_rate=.06, max_bins=63, early_stopping=False,
        max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
        random_state=57,
    )


def fit_scores(x, y, train, evaluation, names):
    """Lernt Regelgrenzen und Restmodelle auf den angegebenen Trainingszeilen."""
    x_train, x_eval = x.iloc[train], x.iloc[evaluation]
    baseline = FailureModel().fit(x_train, y[train])
    core = baseline.mechanisms.predict(x_eval).astype(bool)
    if baseline.residual_ is None:
        return core, {name: np.full(len(evaluation), -1.) for name in names}

    baseline_name = next(iter(GRIDS[TUNED]))
    out = {}
    if baseline_name in names:
        out[baseline_name] = baseline.residual_.predict_proba(
            baseline._residual_features(x_eval)
        )[:, 1]
    advanced = [name for name in names if name != baseline_name]
    if not advanced:
        return core, out

    # Die zweite Stufe trainiert nur auf Vorgängen ohne physikalischen Alarm.
    residual = baseline.mechanisms.predict(x_train) == 0
    train_features = feature_variants(x_train.iloc[residual], baseline)
    eval_features = feature_variants(x_eval, baseline)
    for name in advanced:
        version, params = CONFIGURATIONS[name]
        key = params["features"]
        fitted = make_residual(version, params).fit(train_features[key], y[train][residual])
        out[name] = fitted.predict_proba(eval_features[key])[:, 1]
    return core, out


def operating_points(y, core, scores):
    """Zählt Treffer und Kosten für jede mögliche Schwelle bei aktiven Regelalarmen."""
    labels = y[~core]
    values = scores[~core]
    order = np.argsort(-values)
    ranked_labels = labels[order]
    ranked_scores = values[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores) != 0), len(values) - 1] if len(values) else []
    tp = np.r_[0, np.cumsum(ranked_labels)[ends]] + int(y[core].sum())
    fp = np.r_[0, np.cumsum(1 - ranked_labels)[ends]] + int((1 - y[core]).sum())
    fn = int(y.sum()) - tp
    return np.r_[np.inf, ranked_scores[ends]], tp, fp, fn, fp + FALSE_NEGATIVE_COST * fn


def tune(y, core, scores):
    """Wählt eine Kostenschwelle und je eine Schwelle pro Recall-Ziel."""
    levels, tp, _, _, cost = operating_points(y, core, scores)
    best = np.argmin(cost)
    # Zwischen zwei Scores trennen. Gleiche Scores erhalten dieselbe Entscheidung.
    # Unendliche Schwellen kennzeichnen Zusatzalarm für keine oder für alle Zeilen.
    def cutoff(index):
        if index == 0:
            return np.inf
        if index == len(levels) - 1:
            return -np.inf
        upper, lower = levels[index:index + 2]
        return float(max((upper + lower) / 2, np.nextafter(lower, upper)))

    recall_thresholds = {}
    for target in TARGET_RECALLS:
        minimum = (int(y.sum()) * target + 99) // 100
        first = int(np.searchsorted(tp, minimum))
        recall_thresholds[target] = cutoff(first)
    return {"threshold": cutoff(best), "cost": int(cost[best]), "recall_thresholds": recall_thresholds}


def inner_search(x, y, outer_train, inner_seed, names, *, scorer=fit_scores):
    """Sammelt Vorhersagen der inneren Testfalten und stimmt darauf die Schwellen ab."""
    core_inner = np.empty(len(outer_train), dtype=bool)
    score_inner = {name: np.empty(len(outer_train)) for name in names}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed)
    for fit, validate in cv.split(outer_train, y[outer_train]):
        core, scores = scorer(x, y, outer_train[fit], outer_train[validate], names)
        core_inner[validate] = core
        for name in names:
            score_inner[name][validate] = scores[name]
    return {name: tune(y[outer_train], core_inner, score_inner[name]) for name in names}


def predict(core, scores, threshold):
    """Verknüpft den Regelalarm und den schwellenbasierten Zusatzalarm per ODER."""
    return core | (scores >= threshold)


def serial_threshold(threshold):
    if np.isfinite(threshold):
        return float(threshold)
    return "nur_kerneingriffe" if threshold > 0 else "alle_alarme"


def summary_table(results):
    """Fasst die Wiederholungen je Modell zusammen und sortiert nach mittleren Kosten."""
    return sorted([
        {
            "version": version,
            "kosten_mittel": np.mean([row["cost"] for row in repeats]),
            "kosten_min": min(row["cost"] for row in repeats),
            "kosten_max": max(row["cost"] for row in repeats),
            "fp_mittel": np.mean([row["fp"] for row in repeats]),
            "fn_mittel": np.mean([row["fn"] for row in repeats]),
            "recall_pct": 100 * np.mean([row["tp"] / (row["tp"] + row["fn"]) for row in repeats]),
            "kosten_je_seed": "/".join(str(row["cost"]) for row in repeats),
        }
        for version, repeats in results.items()
    ], key=lambda row: row["kosten_mittel"])


def run_validation(x, y, event):
    """Bewertet Modelle und Recall-Ziele auf denselben äußeren Falten."""
    names = list(CONFIGURATIONS)
    original = next(iter(GRIDS[TUNED]))
    versions = {name: [] for name in VERSIONS}
    configs = {name: [] for name in names}
    recall_versions = (BASE, LOGIT, BOOST)
    recalls = {(name, target): [] for name in recall_versions for target in TARGET_RECALLS}
    fold_records = []
    reference_curve = None
    for seed in SEEDS:
        event(f"Äußere Validierung gestartet, Seed {seed}")
        split_pred = {name: np.empty(len(y), dtype=bool) for name in VERSIONS}
        configs_pred = {name: np.empty(len(y), dtype=bool) for name in names}
        recall_pred = {key: np.empty(len(y), dtype=bool) for key in recalls}
        curve_core = np.empty(len(y), dtype=bool)
        curve_scores = np.empty(len(y))
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(outer.split(x, y), 1):
            # Abstimmung innen, abschließender Fit auf der ganzen äußeren Trainingsfalte.
            tuned = inner_search(x, y, train, seed + 100 * fold, names)
            core, scores = fit_scores(x, y, train, test, names)
            split_pred[CORE][test] = core
            split_pred[BASE][test] = predict(core, scores[original], ECONOMIC_THRESHOLD)
            curve_core[test], curve_scores[test] = core, scores[original]
            for name in names:
                configs_pred[name][test] = predict(core, scores[name], tuned[name]["threshold"])
            winners = {
                version: min(grid, key=lambda name: tuned[name]["cost"])
                for version, grid in GRIDS.items()
            }
            for version, winner in winners.items():
                split_pred[version][test] = configs_pred[winner][test]
            recall_thresholds = {}
            for version in recall_versions:
                name = original if version == BASE else winners[version]
                recall_thresholds[version] = {}
                for target, threshold in tuned[name]["recall_thresholds"].items():
                    recall_pred[version, target][test] = predict(core, scores[name], threshold)
                    recall_thresholds[version][target] = serial_threshold(threshold)
            fold_records.append({
                "seed": seed, "fold": fold,
                "winners": winners,
                "thresholds": {name: serial_threshold(row["threshold"]) for name, row in tuned.items()},
                "inner_costs": {name: row["cost"] for name, row in tuned.items()},
                "recall_thresholds": recall_thresholds,
                "outer_metrics": {name: counts(y[test], pred[test]) for name, pred in split_pred.items()},
            })
        for key, pred in split_pred.items():
            versions[key].append({"seed": seed, **counts(y, pred)})
        for key, pred in configs_pred.items():
            configs[key].append({"seed": seed, **counts(y, pred)})
        for key, pred in recall_pred.items():
            recalls[key].append({"seed": seed, **counts(y, pred)})
        if seed == SEEDS[0]:
            # Diese Kurve beschreibt äußere Scores im Nachhinein. Sie wählt keine Schwelle für die CV.
            curve = operating_points(y, curve_core, curve_scores)
            reference_curve = pd.DataFrame(
                dict(zip(("Schwelle", "TP", "FP", "FN", "Kosten"), curve))
            )
            reference_curve["Recall"] = reference_curve["TP"] / int(y.sum())
        event(f"Seed {seed} abgeschlossen. " + ", ".join(
            f"{name}: {values[-1]['cost']}" for name, values in versions.items()
        ))
    return versions, configs, recalls, fold_records, reference_curve


def final_prediction(x, y, version, challenge_path, output, event):
    """Legt die Siegerkonfiguration fest und erzeugt auf Wunsch die Challenge-CSV."""
    indices = np.arange(len(y))
    if version in GRIDS:
        names = list(GRIDS[version])
        tuned = inner_search(x, y, indices, FINAL_SEED, names)
        selected = min(names, key=lambda name: tuned[name]["cost"])
        threshold = tuned[selected]["threshold"]
    else:
        selected = next(iter(GRIDS[TUNED]))
        threshold = ECONOMIC_THRESHOLD if version == BASE else np.inf
    final = {
        "version": version, "configuration": CORE if version == CORE else selected,
        "params": {"C": 1.0} if version in (BASE, TUNED) else CONFIGURATIONS[selected][1],
        "threshold": serial_threshold(threshold), "seed": FINAL_SEED,
    }
    event(f"Konfiguration festgelegt: {final}")
    if challenge_path is None:
        return final

    # Die Challenge wird erst nach der Wahl von Modell und Schwelle eingelesen.
    challenge = read_data(challenge_path)
    combined = pd.concat((x, challenge), ignore_index=True)
    evaluation = np.arange(len(x), len(combined))
    core, scores = fit_scores(combined, y, indices, evaluation, [selected])
    alarms = predict(core, scores[selected], threshold).astype(int)
    file = output / "challenge_prediction_GRUPPENNAME.csv"
    pd.DataFrame({TARGET: alarms}).to_csv(file, index=False, lineterminator="\n")
    written = pd.read_csv(file)
    if list(written.columns) != [TARGET] or not np.array_equal(written[TARGET].to_numpy(), alarms):
        raise RuntimeError("Die gespeicherte Challenge-Prognose stimmt nicht mit der Berechnung überein.")
    final.update(predictions=len(alarms), alarms=int(alarms.sum()))
    event(f"Challenge-Prognose geprüft: {len(alarms)} Zeilen, {alarms.sum()} Alarme")
    return final


def main():
    """Liest die Eingaben ein und schreibt Ranglisten, Prognose und Zeitprotokoll."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--challenge", type=Path, help="Optionaler Vorhersagelauf nach der Modellwahl")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/modelloptimierung"))
    args = parser.parse_args()
    log = []

    def event(message):
        moment = timestamp()
        print(f"{moment} {message}", flush=True)
        log.append({"time": moment, "event": message})

    event("Versuch gestartet. Schwellen und Modellparameter werden in inneren Falten gewählt.")
    data = read_data(args.modeling, labeled=True)
    y = data.pop(TARGET).to_numpy(dtype=int)
    args.ausgabe.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        versions, configs, recalls, folds, curve = run_validation(data, y, event)
        ranking = summary_table(versions)
        final = final_prediction(data, y, ranking[0]["version"], args.challenge, args.ausgabe, event)

    # Die CSVs sind Kurzansichten. Die JSON-Datei hält die Entscheidungen pro Falte fest.
    recall_rows = []
    for (version, target), results in recalls.items():
        row = summary_table({version: results})[0]
        recall_rows.append({"ziel_recall_pct": target, **row})
    pd.DataFrame(ranking).to_csv(args.ausgabe / "rangliste.csv", index=False)
    pd.DataFrame(summary_table(configs)).to_csv(args.ausgabe / "konfigurationen.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(args.ausgabe / "recall_ziele.csv", index=False)
    curve.to_csv(args.ausgabe / "schwellenkurve_seed2006.csv", index=False)
    event("Ranglisten und Schwellenkurven geschrieben.")
    details = {
        "time_log": log,
        "versions": versions,
        "configurations": configs,
        "folds": folds,
        "recall_runs": [
            {"version": version, "target_recall_pct": target, "repeats": results}
            for (version, target), results in recalls.items()
        ],
        "final": final,
        "grid": GRIDS,
    }
    (args.ausgabe / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
