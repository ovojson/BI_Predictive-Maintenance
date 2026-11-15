"""Inspect failures missed by the existing policy on the outer test folds."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from modell import ECONOMIC_THRESHOLD, FEATURES, SEEDS, TARGET, FailureModel, counts, read_data
from physik import physical_inputs


def folds(frame, seed):
    # Pro Seed entstehen Vorhersagen für jede Zeile aus ihrer jeweiligen äußeren Testfalte.
    rows = []
    x = frame[FEATURES]
    y = frame[TARGET].to_numpy(dtype=int)
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        model = FailureModel().fit(x.iloc[train], y[train])
        t = model.mechanisms.thresholds_
        test_frame = x.iloc[test]
        alarms = model.predict(test_frame)
        power, gap, speed, stress, variant = physical_inputs(test_frame)
        wear = test_frame["Tool wear [min]"].to_numpy(dtype=float)
        stress_cutoff = np.array([t["stress"][kind] for kind in variant])
        wear_margin = np.full(len(test), np.inf)
        residual_score = np.zeros(len(test))
        above_wear_onset = np.zeros(len(test), dtype=bool)
        if model.residual_ is not None:
            features = model._residual_features(test_frame)
            residual_score = model.residual_.predict_proba(features)[:, 1]
            above_wear_onset = wear > model.wear_onset_
            wear_coefficient = model.residual_[1].coef_[0][0] / model.residual_[0].scale_[0]
            if wear_coefficient > 0:
                # Logit-Abstand in zusätzliche Standzeitminuten umrechnen und den Knick berücksichtigen.
                alarm_logit = np.log(ECONOMIC_THRESHOLD / (1 - ECONOMIC_THRESHOLD))
                wear_margin = (alarm_logit - model.residual_.decision_function(features)) / wear_coefficient
                wear_margin += np.maximum(model.wear_onset_ - wear, 0)
        rows.append(pd.DataFrame({
            "y": y[test], "alarm": alarms, "wear": wear, "score_wear": residual_score,
            "above_wear_onset": above_wear_onset,
            "power": np.minimum(power - t["power_low"], t["power_high"] - power),
            "gap": gap - t["gap"], "speed": speed - t["speed"],
            "stress": stress_cutoff - stress,
            "wear_margin": wear_margin,
        }))
    return pd.concat(rows, ignore_index=True)


def analyze(unflagged):
    # Kennzahlen beschreiben die Verschleißwerte und Scores der untersuchten Fälle.
    result = {
        "n": len(unflagged),
        "wear_le_onset": int((~unflagged["above_wear_onset"]).sum()),
        "wear_200_219": int(unflagged["wear"].between(200, 219).sum()),
        "wear_ge_220": int((unflagged["wear"] >= 220).sum()),
        "wear_score_median": float(unflagged["score_wear"].median()),
        "wear_score_le_01": int((unflagged["score_wear"] <= .01).sum()),
        "tolerances": {},
    }
    # Halbierte, normale und verdoppelte Toleranzen prüfen die Nähe zu den Alarmgrenzen.
    for scale in (.5, 1, 2):
        power = unflagged["power"] <= (100 * scale)
        thermal = (unflagged["gap"] <= (0.1 * scale + 1e-10)) & (unflagged["speed"] <= (25 * scale))
        overload = unflagged["stress"] <= (500 * scale)
        wear_alarm = unflagged["wear_margin"] <= (10 * scale)
        physical = power | thermal | overload
        result["tolerances"][str(scale)] = {
            "near_power": int(power.sum()), "near_thermal": int(thermal.sum()),
            "near_overload": int(overload.sum()), "near_wear": int(wear_alarm.sum()),
            "near_physical_any": int(physical.sum()),
            "near_any": int((physical | wear_alarm).sum()),
        }
    return result


def main():
    # Dateipfade für die gelabelten Eingaben und die gespeicherte Diagnose festlegen.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, default=Path.home() / "Downloads" / "modeling.csv")
    parser.add_argument("--ausgabe", type=Path, default=Path("experimente/modelloptimierung/fehleranalyse.json"))
    args = parser.parse_args()
    started = datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")
    frame = read_data(args.modeling, labeled=True)
    result = {"started": started, "seeds": {}}
    for seed in SEEDS:
        folded = folds(frame, seed)
        # Übersehene Ausfälle werden mit korrekt freigegebenen Vorgängen verglichen.
        result["seeds"][str(seed)] = {
            "baseline": counts(folded["y"].to_numpy(), folded["alarm"].to_numpy()),
            "fn": analyze(folded.query("y == 1 and alarm == 0")),
            "tn": analyze(folded.query("y == 0 and alarm == 0")),
        }
    result["finished"] = datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")
    args.ausgabe.parent.mkdir(parents=True, exist_ok=True)
    args.ausgabe.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["seeds"]["2006"], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
