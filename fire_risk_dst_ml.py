"""
Fire risk model with:
- Inputs: Temperature (°C), Humidity (%), IR (0/1)
- DST fusion (Temp + Hum + IR) -> BetP(Fire) + conflictK
- ML: Calibrated Logistic Regression -> risk probability
- Output: digital command 0/1 (for Arduino)


Run interactive mode:
  python fire_risk_dst_ml.py

Run serial bridge mode:
  python fire_risk_dst_ml_ir_v2.py --mode serial --port COM3 --baud 115200
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix


# =========================
# Config
# =========================
@dataclass(frozen=True)
class Config:
    seed: int = 7

    # synthetic data size & fire rarity
    synth_n: int = 12000
    synth_fire_rate: float = 0.03

    # decision settings
    threshold: float = 0.90
    consecutive_required: int = 1  # set 5–10 for fewer false triggers

    # DST sensor reliabilities (discount factors)
    r_temp: float = 0.75
    r_hum: float = 0.45
    r_ir: float = 0.75  # slightly higher so IR contributes more (still not absolute)

    # weight small real dataset higher than synthetic
    real_weight: float = 6.0
    synth_weight: float = 1.0

    # interactive delta clipping (manual input isn’t a true time-series)
    clip_dT: float = 2.0      # °C per step
    clip_dRH: float = 5.0     # %RH per step

    # hard-trip safety overrides (makes 89°C + IR=1 => command=1)
    hard_trip_ir_temp: float = 60.0   # if IR=1 and T >= this -> fire
    hard_trip_temp: float = 85.0      # if T >= this -> fire even if IR fails


CFG = Config()
RNG = np.random.default_rng(CFG.seed)

FEATURES = ["T_C", "RH_pct", "IR", "dT_per_step", "dRH_per_step", "dst_betP_fire", "dst_conflictK"]


# =========================
# Your real data (as provided)
# =========================
REAL_DATA_TEXT = r"""
Temperature: 23.90°C     Humidity: 40.50%     IR sensor reading: 0     Conclusion: no fire    initial
Temperature: 23.90°C     Humidity: 42.90%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.80°C     Humidity: 47.00%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.80°C     Humidity: 54.60%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.90°C     Humidity: 52.50%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.80°C     Humidity: 48.00%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.70°C     Humidity: 46.70%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.70°C     Humidity: 46.40%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.70°C     Humidity: 46.00%     IR sensor reading: 0     Conclusion: no fire
Temperature: 23.70°C     Humidity: 45.60%     IR sensor reading: 0     Conclusion: no fire
Temperature: 25.50°C     Humidity: 52.90%     IR sensor reading: 0     Conclusion: no fire
Temperature: 25.70°C     Humidity: 49.40%     IR sensor reading: 0     Conclusion: no fire
Temperature: 24.90°C     Humidity: 45.60%     IR sensor reading: 0     Conclusion: no fire
Temperature: 24.70°C     Humidity: 44.70%     IR sensor reading: 1     Conclusion: no fire  fire pus
Temperature: 24.60°C     Humidity: 45.40%     IR sensor reading: 1     Conclusion: no fire
Temperature: 24.80°C     Humidity: 44.40%     IR sensor reading: 1     Conclusion: no fire
Temperature: 25.10°C     Humidity: 44.20%     IR sensor reading: 1     Conclusion: no fire
Temperature: 25.40°C     Humidity: 44.60%     IR sensor reading: 1     Conclusion: no fire
Temperature: 27.00°C     Humidity: 44.50%     IR sensor reading: 1     Conclusion: no fire
Temperature: 30.10°C     Humidity: 46.20%     IR sensor reading: 1     Conclusion: fire
Temperature: 33.70°C     Humidity: 47.80%     IR sensor reading: 1     Conclusion: fire
Temperature: 36.80°C     Humidity: 48.90%     IR sensor reading: 1     Conclusion: fire
Temperature: 36.30°C     Humidity: 48.10%     IR sensor reading: 1     Conclusion: fire
Temperature: 35.50°C     Humidity: 46.90%     IR sensor reading: 1     Conclusion: fire
Temperature: 34.60°C     Humidity: 46.00%     IR sensor reading: 1     Conclusion: fire
Temperature: 56.00°C     Humidity: 57.70%     IR sensor reading: 1     Conclusion: fire
Temperature: 50.20°C     Humidity: 53.40%     IR sensor reading: 1     Conclusion: fire
Temperature: 45.70°C     Humidity: 50.20%     IR sensor reading: 1     Conclusion: fire
Temperature: 43.20°C     Humidity: 47.90%     IR sensor reading: 1     Conclusion: fire
Temperature: 41.90°C     Humidity: 46.70%     IR sensor reading: 1     Conclusion: fire
Temperature: 47.20°C     Humidity: 50.00%     IR sensor reading: 0     Conclusion: no fire
Temperature: 84.50°C     Humidity: 77.40%     IR sensor reading: 0     Conclusion: no fire      fire scos
Temperature: 94.90°C     Humidity: 85.70%     IR sensor reading: 0     Conclusion: no fire
Temperature: 81.90°C     Humidity: 75.10%     IR sensor reading: 0     Conclusion: no fire     racirea senzorului de caldura
Temperature: 70.60°C     Humidity: 64.90%     IR sensor reading: 0     Conclusion: no fire
Temperature: 62.30°C     Humidity: 58.90%     IR sensor reading: 0     Conclusion: no fire
Temperature: 56.00°C     Humidity: 54.60%     IR sensor reading: 0     Conclusion: no fire
Temperature: 50.90°C     Humidity: 50.90%     IR sensor reading: 0     Conclusion: no fire
Temperature: 46.60°C     Humidity: 47.50%     IR sensor reading: 0     Conclusion: no fire
Temperature: 43.40°C     Humidity: 44.80%     IR sensor reading: 0     Conclusion: no fire
Temperature: 40.80°C     Humidity: 42.90%     IR sensor reading: 0     Conclusion: no fire
"""


# =========================
# Numerically stable sigmoid
# =========================
def sigmoid_stable(x: float) -> float:
    # Clip avoids exp overflow and is plenty accurate for classification
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


# =========================
# Parse real data
# =========================
REAL_ROW_RE = re.compile(
    r"Temperature:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:°?C)\s*"
    r"Humidity:\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*"
    r"IR sensor reading:\s*([01])\s*"
    r"Conclusion:\s*(fire|no fire)",
    re.IGNORECASE
)

def parse_real_data(text: str) -> pd.DataFrame:
    rows: List[Tuple[float, float, int, int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = REAL_ROW_RE.search(line)
        if not m:
            continue
        T = float(m.group(1))
        RH = float(m.group(2))
        IR = int(m.group(3))
        fire = 1 if m.group(4).lower() == "fire" else 0
        rows.append((T, RH, IR, fire))

    if not rows:
        raise ValueError("No valid rows parsed from REAL_DATA_TEXT.")

    df = pd.DataFrame(rows, columns=["T_C", "RH_pct", "IR", "fire"])
    df["dT_per_step"] = df["T_C"].diff().fillna(0.0)
    df["dRH_per_step"] = df["RH_pct"].diff().fillna(0.0)
    return df


# =========================
# Synthetic generator (includes IR)
# =========================
def make_synth(n: int, fire_rate: float) -> pd.DataFrame:
    y = RNG.choice([0, 1], size=n, p=[1 - fire_rate, fire_rate])

    T = np.zeros(n)
    RH = np.zeros(n)
    IR = np.zeros(n, dtype=int)
    dT = np.zeros(n)
    dRH = np.zeros(n)

    for i in range(n):
        if y[i] == 1:
            T[i] = np.clip(RNG.normal(46, 10), 25, 85)
            dT[i] = np.clip(RNG.normal(0.55, 0.22), 0.15, 1.50)
            RH[i] = np.clip(RNG.normal(42, 11) - 6 * (1 / (1 + np.exp(-(T[i] - 45) / 8))), 10, 85)
            dRH[i] = np.clip(RNG.normal(-0.30, 0.30), -1.8, 0.8)
            IR[i] = 1 if RNG.random() < 0.92 else 0
        else:
            r = RNG.random()
            if r < 0.15:
                T[i] = np.clip(RNG.normal(36, 6), 22, 70)
                dT[i] = np.clip(RNG.normal(0.10, 0.08), -0.10, 0.50)
                RH[i] = np.clip(RNG.normal(48, 9), 15, 80)
                dRH[i] = np.clip(RNG.normal(0.05, 0.15), -0.6, 0.7)
                IR[i] = 1 if RNG.random() < 0.01 else 0
            elif r < 0.22:
                # IR false positives similar to your logs
                T[i] = np.clip(RNG.normal(25.0, 1.8), 20, 31)
                dT[i] = np.clip(RNG.normal(0.04, 0.05), -0.10, 0.25)
                RH[i] = np.clip(RNG.normal(46.0, 4.5), 25, 70)
                dRH[i] = np.clip(RNG.normal(0.00, 0.12), -0.6, 0.6)
                IR[i] = 1
            else:
                T[i] = np.clip(RNG.normal(24, 2.2), 18, 33)
                dT[i] = np.clip(RNG.normal(0.02, 0.05), -0.15, 0.20)
                RH[i] = np.clip(RNG.normal(50, 7), 20, 75)
                dRH[i] = np.clip(RNG.normal(0.00, 0.12), -0.6, 0.6)
                IR[i] = 1 if RNG.random() < 0.01 else 0

        # jitter
        if RNG.random() < 0.01:
            T[i] += RNG.normal(0, 3.0)
        if RNG.random() < 0.01:
            RH[i] += RNG.normal(0, 7.0)

    return pd.DataFrame({
        "T_C": T,
        "RH_pct": RH,
        "IR": IR,
        "dT_per_step": dT,
        "dRH_per_step": dRH,
        "fire": y,
    })


# =========================
# DST (3 sensors)
# =========================
def bpa_temperature(T: float, dT: float, r: float) -> Tuple[float, float, float]:
    # Fire evidence uses *positive* rate-of-rise (cooling shouldn't explode the math)
    dT_pos = max(0.0, float(dT))

    e_level = sigmoid_stable((T - 40) / 5.0)
    e_rate = sigmoid_stable((dT_pos - 0.20) / 0.08)

    s = float(np.clip(0.65 * e_level + 0.35 * e_rate, 0, 1))
    return r * s, r * (1 - s), 1 - r

def bpa_humidity(RH: float, dRH: float, r: float) -> Tuple[float, float, float]:
    # Humidity as weak evidence; use only *falling* humidity as fire-ish
    dRH_neg = min(0.0, float(dRH))  # <= 0
    e_low = sigmoid_stable((45 - RH) / 7.0)
    e_trend = sigmoid_stable(((-dRH_neg) - 0.10) / 0.20)  # more negative -> higher evidence

    s = float(np.clip(0.7 * e_low + 0.3 * e_trend, 0, 1))
    return r * s, r * (1 - s), 1 - r

def bpa_ir(IR: int, T: float, r: float) -> Tuple[float, float, float]:
    if IR not in (0, 1):
        raise ValueError("IR must be 0 or 1")

    if IR == 1:
        # stronger baseline than before, but still temperature-sensitive
        s = float(np.clip(0.50 + 0.50 * sigmoid_stable((T - 30) / 4.0), 0, 1))
    else:
        s = 0.08

    return r * s, r * (1 - s), 1 - r

def ds_combine_binary(m1: Tuple[float, float, float], m2: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    m1F, m1N, m1U = m1
    m2F, m2N, m2U = m2

    K = m1F * m2N + m1N * m2F
    denom = max(1e-9, 1.0 - K)

    mF = (m1F * m2F + m1F * m2U + m1U * m2F) / denom
    mN = (m1N * m2N + m1N * m2U + m1U * m2N) / denom
    mU = (m1U * m2U) / denom
    return mF, mN, mU, K

def dst_features_for_row(T: float, RH: float, IR: int, dT: float, dRH: float) -> Tuple[float, float]:
    mt = bpa_temperature(T, dT, CFG.r_temp)
    mh = bpa_humidity(RH, dRH, CFG.r_hum)
    mi = bpa_ir(IR, T, CFG.r_ir)

    mF1, mN1, mU1, K1 = ds_combine_binary(mt, mh)
    mF2, mN2, mU2, K2 = ds_combine_binary((mF1, mN1, mU1), mi)

    betP_fire = mF2 + 0.5 * mU2
    conflictK = max(K1, K2)
    return float(betP_fire), float(conflictK)

def add_dst_features(df: pd.DataFrame) -> pd.DataFrame:
    betp = np.zeros(len(df))
    conflict = np.zeros(len(df))

    arr = df[["T_C", "RH_pct", "IR", "dT_per_step", "dRH_per_step"]].to_numpy()
    for i in range(len(df)):
        T, RH, IRv, dT, dRH = float(arr[i, 0]), float(arr[i, 1]), int(arr[i, 2]), float(arr[i, 3]), float(arr[i, 4])
        betp[i], conflict[i] = dst_features_for_row(T, RH, IRv, dT, dRH)

    out = df.copy()
    out["dst_betP_fire"] = betp
    out["dst_conflictK"] = conflict
    return out


# =========================
# Training
# =========================
def build_training_table() -> Tuple[pd.DataFrame, np.ndarray]:
    real = parse_real_data(REAL_DATA_TEXT)
    synth = make_synth(CFG.synth_n, CFG.synth_fire_rate)

    df = pd.concat([synth, real], ignore_index=True)
    weights = np.ones(len(df), dtype=float) * CFG.synth_weight
    weights[-len(real):] = CFG.real_weight

    df = add_dst_features(df)
    return df, weights

def train_models(df: pd.DataFrame, weights: np.ndarray):
    X = df[FEATURES]
    y = df["fire"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.25, random_state=CFG.seed, stratify=y
    )

    base_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2500, class_weight="balanced")),
    ])

    calibrated = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train, sample_weight=w_train)

    p = calibrated.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation (held-out test) ===")
    print("ROC-AUC:", round(roc_auc_score(y_test, p), 4))
    print("PR-AUC :", round(average_precision_score(y_test, p), 4))
    print("Brier  :", round(brier_score_loss(y_test, p), 5))

    yhat = (p >= CFG.threshold).astype(int)
    cm = confusion_matrix(y_test, yhat)
    print("\nConfusion matrix @ threshold =", CFG.threshold)
    print(cm)

    # export LR for Arduino (plain LR, not calibrated)
    plain_lr = base_lr.fit(X, y, lr__sample_weight=weights)
    scaler = plain_lr.named_steps["scaler"]
    lr = plain_lr.named_steps["lr"]

    export = {
        "feature_order": FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "lr_intercept": float(lr.intercept_[0]),
        "lr_coef": lr.coef_[0].tolist(),
        "threshold": CFG.threshold,
    }
    return calibrated, export


# =========================
# Decision logic
# =========================
class DecisionFilter:
    """Consecutive-hit filter to reduce flicker."""
    def __init__(self, threshold: float, consecutive_required: int):
        self.threshold = float(threshold)
        self.N = int(max(1, consecutive_required))
        self.hits = 0

    def update(self, risk: float) -> int:
        if risk >= self.threshold:
            self.hits += 1
        else:
            self.hits = max(0, self.hits - 1)
        return 1 if self.hits >= self.N else 0

def digital_decision(risk: float, T: float, IR: int, threshold: float) -> int:
    # hard trips (what you expected for 89/89/IR=1)
    if IR == 1 and T >= CFG.hard_trip_ir_temp:
        return 1
    if T >= CFG.hard_trip_temp:
        return 1
    return 1 if risk >= threshold else 0

def predict_risk(model, T: float, RH: float, IR: int, dT: float, dRH: float) -> float:
    betp, conflict = dst_features_for_row(T, RH, IR, dT, dRH)
    X = pd.DataFrame([{
        "T_C": T, "RH_pct": RH, "IR": IR,
        "dT_per_step": dT, "dRH_per_step": dRH,
        "dst_betP_fire": betp, "dst_conflictK": conflict
    }])
    return float(model.predict_proba(X[FEATURES])[:, 1][0])


# =========================
# Modes
# =========================
def run_interactive(model, threshold: float, consecutive_required: int):
    print("\n--- Interactive mode ---")
    print("Type 'q' to quit. Type 'r' to reset baseline (dT=dRH=0).")
    print("dT/dRH are computed from the previous input and clipped for stability.\n")

    filt = DecisionFilter(threshold, consecutive_required)
    last_T: Optional[float] = None
    last_RH: Optional[float] = None

    while True:
        t_in = input("Temperature (°C): ").strip().lower()
        if t_in == "q":
            break
        if t_in == "r":
            last_T, last_RH = None, None
            print("Baseline reset.\n")
            continue

        rh_in = input("Humidity (%): ").strip().lower()
        if rh_in == "q":
            break
        if rh_in == "r":
            last_T, last_RH = None, None
            print("Baseline reset.\n")
            continue

        ir_in = input("IR (0/1): ").strip().lower()
        if ir_in == "q":
            break
        if ir_in == "r":
            last_T, last_RH = None, None
            print("Baseline reset.\n")
            continue

        try:
            T = float(t_in)
            RH = float(rh_in)
            IR = int(ir_in)
            if IR not in (0, 1):
                raise ValueError("IR must be 0 or 1.")
        except Exception as e:
            print("Invalid input:", e, "\n")
            continue

        # compute deltas, then clip (manual input is not a real time-series)
        dT = 0.0 if last_T is None else (T - last_T)
        dRH = 0.0 if last_RH is None else (RH - last_RH)
        dT = float(np.clip(dT, -CFG.clip_dT, CFG.clip_dT))
        dRH = float(np.clip(dRH, -CFG.clip_dRH, CFG.clip_dRH))

        last_T, last_RH = T, RH

        risk = predict_risk(model, T, RH, IR, dT, dRH)

        # hard-trip or ML threshold, then optional consecutive filter on the ML path
        cmd_base = digital_decision(risk, T, IR, threshold)
        cmd = 1 if cmd_base == 1 else filt.update(risk)

        print(f"\nComputed dT={dT:.3f}, dRH={dRH:.3f}")
        print(f"Risk (0..1): {risk:.3f}")
        print(f"Digital output (0/1): {cmd}")
        print("-" * 40 + "\n")


def run_serial_bridge(model, port: str, baud: int, threshold: float, consecutive_required: int):
    """
    Arduino -> Python:
      "S,<T>,<RH>,<IR>\n"  (deltas computed in Python)
    Python -> Arduino:
      "0\n" or "1\n"
    """
    try:
        import serial  # type: ignore
    except ImportError:
        raise SystemExit("pyserial not installed. Run: pip install pyserial")

    ser = serial.Serial(port, baud, timeout=1)
    print(f"\n--- Serial bridge mode ---\nListening on {port} @ {baud}\n")

    filt = DecisionFilter(threshold, consecutive_required)
    last_T: Optional[float] = None
    last_RH: Optional[float] = None

    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw or not raw.startswith("S,"):
            continue

        parts = raw.split(",")
        try:
            T = float(parts[1])
            RH = float(parts[2])
            IR = int(parts[3])
            if IR not in (0, 1):
                continue

            dT = 0.0 if last_T is None else (T - last_T)
            dRH = 0.0 if last_RH is None else (RH - last_RH)
            dT = float(np.clip(dT, -CFG.clip_dT, CFG.clip_dT))
            dRH = float(np.clip(dRH, -CFG.clip_dRH, CFG.clip_dRH))
            last_T, last_RH = T, RH

            risk = predict_risk(model, T, RH, IR, dT, dRH)
            cmd_base = digital_decision(risk, T, IR, threshold)
            cmd = 1 if cmd_base == 1 else filt.update(risk)

            ser.write(f"{cmd}\n".encode())

        except Exception:
            continue


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["interactive", "serial"], default="interactive")
    ap.add_argument("--threshold", type=float, default=CFG.threshold)
    ap.add_argument("--consecutive", type=int, default=CFG.consecutive_required)
    ap.add_argument("--export_json", type=str, default="")
    ap.add_argument("--port", type=str, default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    return ap.parse_args()

def main():
    args = parse_args()

    df, w = build_training_table()
    calibrated_model, export = train_models(df, w)

    print("\nEXPORT_JSON (for LR-on-Arduino deployment):")
    print(json.dumps(export, indent=2))

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        print(f"\nSaved export to: {args.export_json}")

    if args.mode == "interactive":
        run_interactive(calibrated_model, args.threshold, args.consecutive)
    else:
        run_serial_bridge(calibrated_model, args.port, args.baud, args.threshold, args.consecutive)

if __name__ == "__main__":
    main()