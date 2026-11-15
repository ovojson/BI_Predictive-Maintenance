"""Continuous distances to the boundaries of a fitted physical rule model."""

import numpy as np
import pandas as pd

from physik import physical_inputs


def distance_features(frame, mechanisms):
    """Return signed margins with positive values on the safe side of each rule.

    The boundary parameters and temperature precision come from the fitted model.
    A non-finite margin means that the corresponding rule has no finite boundary.
    Its value is set to zero so both residual learners receive finite inputs.
    """
    # Dieselben Sensorgrößen und Grenzen verwenden wie das trainierte Regelmodell.
    power, gap, speed, stress, variant = physical_inputs(frame, mechanisms.round_gap)
    thresholds = mechanisms.thresholds_
    stress_limits = np.array([thresholds["stress"][kind] for kind in variant])

    # Der Abstand ist auf der sicheren Seite positiv. Leistung wird in kW skaliert.
    low_power = (power - thresholds["power_low"]) / 1000
    high_power = (thresholds["power_high"] - power) / 1000
    cooling_gap = gap - thresholds["gap"]
    cooling_speed = (speed - thresholds["speed"]) / 100

    features = pd.DataFrame(
        {
            "power_low_kw": low_power,
            "power_high_kw": high_power,
            # Bei Leistung zählt der Abstand zur näheren Grenze.
            "power_nearest_kw": np.minimum(low_power, high_power),
            "cooling_gap_k": cooling_gap,
            "cooling_speed_100rpm": cooling_speed,
            # Der kombinierte Kühlungsabstand ist nur bei zwei negativen Abständen negativ.
            "cooling_combined": np.maximum(cooling_gap, cooling_speed),
            "stress_1000": (stress_limits - stress) / 1000,
        },
        index=frame.index,
    )
    # Nicht endliche Abstände werden als 0 kodiert, damit beide Restmodelle sie verarbeiten können.
    return features.replace([np.inf, -np.inf], 0.0)
