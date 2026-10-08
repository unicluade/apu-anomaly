"""a1. 라벨 없는 센서 데이터를 처음 볼 때."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plotstyle  # noqa: E402

FIG = Path("reports/figures")
plotstyle.apply()
ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/labeled.parquet")

    print("=" * 74)
    print("1. 규모와 라벨")
    print("=" * 74)
    print(f"\n  {len(df):,} 행 · 기간 {df.timestamp.min():%Y-%m-%d} ~ {df.timestamp.max():%Y-%m-%d}")
    print(f"  고장 라벨 {int(df.is_fail.sum()):,}행 ({df.is_fail.mean()*100:.2f}%) · "
          f"독립 사건 {df.event.nunique()-1}건")

    print("\n" + "=" * 74)
    print("2. 고장일 때 센서가 어떻게 달라지나")
    print("=" * 74)
    g = df.groupby(df.is_fail.map({0: "정상", 1: "고장"}))[ANALOG].mean()
    g.loc["비율"] = (g.loc["고장"] / g.loc["정상"].replace(0, np.nan))
    print("\n" + g.round(3).to_string())
    print("""
  공기 누출(air leak)이 나면 압축기가 쉬지 못합니다.
  TP2(압축기 압력)와 Motor_current 가 올라가고 오일이 뜨거워집니다.
  H1 이 0 에 붙는 것도 같은 이유입니다 — 사이클론 분리기 배출이 멈춥니다.
  ★ 도메인을 모르면 이 해석을 못 합니다. 센서 설명 문서를 먼저 읽으세요.""")

    print("\n" + "=" * 74)
    print("3. 사건별로 다르다")
    print("=" * 74)
    print("\n" + df[df.event != ""].groupby("event")[ANALOG].mean().round(3).to_string())
    print("""
  F4 만 DV_pressure 가 0 근처이고 오일온도가 84도로 유독 높습니다.
  같은 'Air leak' 이라도 양상이 다릅니다. 사건이 4건뿐인데 그 안에서도
  갈리니, 일반화할 근거가 더 얇아집니다.""")

    print("\n" + "=" * 74)
    print("4. 결측 — 연속 시계열이 아니다")
    print("=" * 74)
    gap = df.timestamp.diff().dt.total_seconds().dropna()
    print(f"\n  중앙값 간격 {gap.median():.0f}초")
    for lo in (60, 600, 3600):
        print(f"  {lo:>5}초 초과 간격  {int((gap>lo).sum()):>4}회")
    print(f"  최대 {gap.max()/3600:.1f}시간")

    # 그림
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    d = df.set_index("timestamp")
    for a, col in zip(ax.ravel()[:3], ["TP2", "Oil_temperature", "Motor_current"]):
        h = d[col].resample("1h").mean()
        a.plot(h.index, h.values, lw=.6, color="#2e7d32")
        for e in ("F1", "F2", "F3", "F4"):
            m = df.event == e
            if m.any():
                a.axvspan(df.loc[m, "timestamp"].min(), df.loc[m, "timestamp"].max(),
                          color="#c62828", alpha=.35)
        a.set_title(f"{col} (1시간 평균) · 빨강=고장 구간")
    day = df.set_index("timestamp").is_fail.resample("1D").mean() * 100
    ax[1, 1].bar(day.index, day.values, color="#c62828", width=1)
    ax[1, 1].set_title("일별 고장 라벨 비율 (%) — 사건 4건뿐")
    plt.tight_layout(); plt.savefig(FIG / "a1_eda.png"); plt.close()
    print(f"\n[저장] {FIG/'a1_eda.png'}")


if __name__ == "__main__":
    main()
