"""시계열 윈도 피처 — 과거만 보고 만든다.

센서 값 하나만으로는 '지금 이상한가'를 알기 어렵다.
'평소와 다른가' 를 보려면 최근 구간의 통계가 필요하다.

규칙은 E·F 권과 같다. i번째 시점의 피처는 i시점 '까지' 만 본다.
rolling 은 기본이 과거 방향이라 그대로 쓰면 되지만,
center=True 를 쓰면 미래가 섞인다. 절대 쓰지 않는다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/processed/labeled.parquet")
OUT = Path("data/processed/features.parquet")

# 아날로그 센서 7종. 디지털 신호는 0/1 이라 이동통계가 의미가 약하다.
ANALOG = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]
DIGITAL = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

# 10초 간격이므로 창 길이를 '행 수' 로 환산한다.
WINDOWS = {"10m": 60, "1h": 360, "6h": 2160}


def build() -> pd.DataFrame:
    df = pd.read_parquet(SRC).sort_values("timestamp").reset_index(drop=True)
    print(f"[피처] {len(df):,} 행 · 10초 간격")

    f = pd.DataFrame(
        {"timestamp": df.timestamp, "event": df.event, "is_fail": df.is_fail}
    )

    # 현재 값
    for c in ANALOG + DIGITAL:
        f[c] = df[c].astype("float32")

    # 이동 통계 — 과거 방향만
    for name, w in WINDOWS.items():
        for c in ANALOG:
            r = df[c].rolling(w, min_periods=max(w // 4, 2))
            f[f"{c}_mean_{name}"] = r.mean().astype("float32")
            f[f"{c}_std_{name}"] = r.std().astype("float32")
        print(f"  · {name} 창 ({w}행) 이동 평균·표준편차")

    # ★ 개인 기준 상대값 — '지금 값이 최근 평소 대비 얼마나 벗어났나'
    for c in ANALOG:
        m, s = f[f"{c}_mean_6h"], f[f"{c}_std_6h"]
        f[f"{c}_z"] = (
            ((f[c] - m) / s.replace(0, np.nan)).clip(-10, 10).astype("float32")
        )
    print("  · 6시간 기준 z-score")

    # 디지털 신호의 최근 점유율 — 밸브가 최근에 얼마나 켜져 있었나
    for c in DIGITAL:
        f[f"{c}_rate_1h"] = df[c].rolling(360, min_periods=90).mean().astype("float32")

    # 창을 채우지 못한 앞부분은 버린다(6시간 = 2,160행)
    before = len(f)
    f = f.iloc[WINDOWS["6h"] :].reset_index(drop=True)
    f = f.dropna()
    print(f"  · 창 미충족 앞부분 제거: {before:,} → {len(f):,} 행")

    n_feat = f.shape[1] - 3
    print(f"\n[ok] {len(f):,} 행 × {n_feat} 피처 · 고장 {int(f.is_fail.sum()):,}행")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(OUT, index=False)
    return f


if __name__ == "__main__":
    build()
