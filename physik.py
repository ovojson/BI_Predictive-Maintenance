import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Qualitätsstufen für gemeinsame oder getrennte Belastungsgrenzen.
TYPES = ("H", "M", "L")
TYPE_LEVEL = {"L": 0, "M": 1, "H": 2}


def physical_inputs(data, round_gap=False):
    # Aus den Sensorwerten entstehen Leistung, Temperaturdifferenz und ein Belastungsindikator.
    speed = data["Rotational speed [rpm]"].to_numpy(dtype=float)
    torque = data["Torque [Nm]"].to_numpy(dtype=float)
    wear = data["Tool wear [min]"].to_numpy(dtype=float)
    power = speed * torque * (2 * np.pi / 60)
    gap = data["Process temperature [K]"].to_numpy(dtype=float) - data[
        "Air temperature [K]"
    ].to_numpy(dtype=float)
    if round_gap:
        gap = np.round(gap, 1)
    return power, gap, speed, wear * torque, data["Type"].to_numpy()


def cutoff_in_interval(lower, upper, units):
    # Alle Kandidaten im Intervall trennen dieselben Lernfälle. Ein passendes Raster wird bevorzugt.
    if units:
        for unit in units:
            simple = (np.floor(lower / unit) + 1) * unit
            if lower < simple <= upper:
                return simple

    middle = lower + (upper - lower) / 2
    return max(middle, np.nextafter(lower, upper))


def fit_lower_cutoff(values, gain, min_precision=0, units=()):
    """Choose a cutoff that maximizes savings relative to classifying none."""
    # Nullgewichte markieren Fälle, die andere Regeln bereits abdecken.
    uncovered = gain != 0
    values = values[uncovered]
    gain = gain[uncovered]
    if len(values) == 0:
        return -np.inf, 0
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    # Gleiche Werte bleiben zusammen. Der kumulierte Gewinn bewertet jede mögliche Grenze.
    boundaries = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    scores = np.cumsum(gain[order])[boundaries]
    positives = np.cumsum(gain[order] > 0)[boundaries]
    negatives = np.cumsum(gain[order] < 0)[boundaries]
    scores[positives < min_precision * (positives + negatives)] = -1
    best = np.argmax(scores)
    if scores[best] <= 0:
        return -np.inf, 0

    last = boundaries[best]
    if last == len(values) - 1:
        return np.inf, scores[best]

    lower, upper = ordered[last:last + 2]
    return cutoff_in_interval(lower, upper, units), scores[best]


def fit_thermal(gap, speed, gain, min_precision, simplify):
    # Die Kühlungsregel alarmiert nur bei kleiner Temperaturdifferenz und niedriger Drehzahl.
    uncovered = gain != 0
    if not uncovered.any():
        return -np.inf, -np.inf

    unique_gap, gap_bin = np.unique(gap[uncovered], return_inverse=True)
    unique_speed, speed_bin = np.unique(speed[uncovered], return_inverse=True)
    shape = (len(unique_gap), len(unique_speed))
    bins = np.ravel_multi_index((gap_bin, speed_bin), shape)
    weights = gain[uncovered]
    scores = np.bincount(bins, weights=weights, minlength=np.prod(shape)).reshape(shape)
    positives = np.bincount(
        bins, weights=weights > 0, minlength=np.prod(shape)
    ).reshape(shape)
    negatives = np.bincount(
        bins, weights=weights < 0, minlength=np.prod(shape)
    ).reshape(shape)
    # Zweidimensionale Summen bewerten alle Kombinationen der beiden Untergrenzen.
    scores = scores.cumsum(axis=0).cumsum(axis=1)
    positives = positives.cumsum(axis=0).cumsum(axis=1)
    negatives = negatives.cumsum(axis=0).cumsum(axis=1)
    scores[positives < min_precision * (positives + negatives)] = -1
    best_gap, best_speed = np.unravel_index(np.argmax(scores), shape)
    if scores[best_gap, best_speed] <= 0:
        return -np.inf, -np.inf

    lower = unique_gap[best_gap]
    upper = unique_gap[best_gap + 1] if best_gap + 1 < len(unique_gap) else np.inf
    included = gap <= lower
    speed_cutoff = fit_lower_cutoff(
        speed[included], gain[included], min_precision, (10, 1) if simplify else ()
    )[0]
    gap_cutoff = cutoff_in_interval(lower, upper, (0.1,) if simplify else ())
    return gap_cutoff, speed_cutoff


def fit_pooled_stress(stress, level, gain, min_precision, simplify):
    # Ein gemeinsamer Grundwert und ein fester Abstand pro Qualitätsstufe teilen Information zwischen den Typen.
    steps = sorted(range(0, 3001, 100), key=lambda step: (
        step % 1000 != 0, step % 500 != 0, step
    ))
    best_score = -1
    for step in steps:
        cutoff, score = fit_lower_cutoff(
            step * level - stress, gain, min_precision, (1000, 100, 10) if simplify else ()
        )
        if score > best_score:
            best_score = score
            best = {kind: -cutoff + step * rank for kind, rank in TYPE_LEVEL.items()}
    return best


# MODELLBEGINN: Physikalisches Regelmodell (Leistung, Kühlung, Werkzeugbelastung)
class PhysicalUnion(ClassifierMixin, BaseEstimator):
    """Fit axis-aligned physical failure rules and combine alarms by OR."""

    def __init__(self, false_negative_cost=6, min_precision=1.0,
                 pool_stress=True, simplify=True, round_gap=False):
        self.false_negative_cost = false_negative_cost
        self.min_precision = min_precision
        self.pool_stress = pool_stress
        self.simplify = simplify
        self.round_gap = round_gap

    def fit(self, X, y):
        power, gap, speed, stress, variant = physical_inputs(X, self.round_gap)
        level = np.array([TYPE_LEVEL[kind] for kind in variant])
        gain = np.where(y, self.false_negative_cost, -1)
        discovery_gain = np.where(y, 1, -1)
        power_units = (1000, 100, 50, 10) if self.simplify else ()
        stress_units = (1000, 100, 10) if self.simplify else ()
        heat_alarm = np.zeros_like(gain, dtype=bool)
        stress_alarm = np.zeros_like(gain, dtype=bool)
        # Zuerst zählen beide Fehlerarten gleich. In zwei weiteren Durchläufen gelten die tatsächlichen Kosten.
        for weights in (discovery_gain, gain, gain):
            residual = np.where(heat_alarm | stress_alarm, 0, weights)
            power_low = fit_lower_cutoff(
                power, residual, self.min_precision, power_units
            )[0]
            power_high = -fit_lower_cutoff(
                -power, residual, self.min_precision, power_units
            )[0]
            power_alarm = (power < power_low) | (power > power_high)
            gap_cutoff, speed_cutoff = fit_thermal(
                gap, speed, np.where(power_alarm | stress_alarm, 0, weights),
                self.min_precision, self.simplify
            )
            heat_alarm = (gap < gap_cutoff) & (speed < speed_cutoff)
            # Die Belastungsregel lernt in diesem Durchlauf aus den verbleibenden Fällen.
            uncovered = ~(power_alarm | heat_alarm)
            if self.pool_stress:
                stress_cutoff = fit_pooled_stress(
                    stress[uncovered], level[uncovered], weights[uncovered],
                    self.min_precision, self.simplify
                )
            else:
                stress_cutoff = {}
                for kind in TYPES:
                    eligible = (variant == kind) & uncovered
                    stress_cutoff[kind] = -fit_lower_cutoff(
                        -stress[eligible], weights[eligible], self.min_precision, stress_units
                    )[0]
            stress_alarm = stress > np.array([stress_cutoff[kind] for kind in variant])
        self.thresholds_ = {
            "power_low": power_low, "power_high": power_high,
            "gap": gap_cutoff, "speed": speed_cutoff, "stress": stress_cutoff
        }
        self.classes_ = np.array([0, 1])
        return self

    def rules(self, X):
        # Jede Ausgabe beschreibt separat, welche gelernte Regel ausgelöst hat.
        power, gap, speed, stress, variant = physical_inputs(X, self.round_gap)
        t = self.thresholds_
        upper_stress = np.array([t["stress"][kind] for kind in variant])
        return {
            "power_low": power < t["power_low"],
            "power_high": power > t["power_high"],
            "heat": (gap < t["gap"]) & (speed < t["speed"]),
            "stress": stress > upper_stress,
        }

    def predict(self, X):
        # Ein Treffer in einer der vier Regeln reicht für einen Alarm.
        return np.logical_or.reduce(tuple(self.rules(X).values())).astype(int)
