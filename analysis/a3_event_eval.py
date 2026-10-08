"""★★★ 사건 단위로 다시 재면 — 아무도 이기지 못했다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import alarms, event_report  # noqa: E402


def main():
    meta = pd.read_parquet("data/processed/test_meta.parquet")
    names = json.load(open("reports/train_meta.json"))["names"]
    ts = meta.timestamp.to_numpy()
    ev = meta.event.to_numpy()

    print("=" * 96)
    print("경보 규칙: 최근 1시간 중 절반 이상이 임계를 넘으면 경보")
    print("리드타임: 고장 시작 '몇 시간 전' 에 경보했나 (음수 = 고장이 시작된 뒤에야 울림)")
    print("=" * 96)
    rows = []
    for i, n in enumerate(names):
        s = np.load(f"data/processed/score_{i}.npy")
        for pct in (0.005, 0.01, 0.02, 0.05):
            a = alarms(s, pct)
            rep = event_report(ts, ev, a)
            row = {"모델": n, "임계(상위)": f"{pct:.1%}"}
            for e, d in rep["events"].items():
                row[e] = f"{d['lead_h']:+.1f}h" if d["detected"] else "미탐"
            row["오경보"] = rep["false_alarms"]
            row["주당 오경보"] = round(rep["fa_per_week"], 2)
            rows.append(row)
    r = pd.DataFrame(rows)
    print("\n" + r.to_string(index=False))

    print("\n" + "=" * 96)
    print("읽는 법")
    print("=" * 96)
    print("""
  ★ 두 사건을 '모두', '고장 전에', '오경보 없이' 잡은 설정이 하나도 없습니다.

  · 지도학습은 F3 를 잡지만 대개 고장이 시작된 뒤입니다(리드타임 음수). F4 는 전 구간 미탐.
  · PCA·오토인코더는 F4 를 한참 전에 잡지만 F3 를 놓칩니다.
  · 규칙(TP2)은 임계를 낮추면 뭐든 잡지만 오경보가 함께 폭증합니다.

  이게 이 프로젝트의 결론입니다. "어느 모델이 낫다" 가 아니라
  <strong>"사건 2건으로는 어느 모델이 나은지 결정할 수 없다"</strong> 입니다.""".replace("<strong>","").replace("</strong>",""))

    print("\n" + "=" * 96)
    print("그럼 사건이 몇 건이어야 하나")
    print("=" * 96)
    print("""
  탐지율을 ±10%p 이내로 추정하려면 사건이 대략 100건 필요합니다
  (비율 추정의 표준오차 √(p(1-p)/n) 가 0.05 아래가 되는 지점).
  두 모델의 탐지율 차이를 검정하려면 그보다 더 필요하고요.

  이 데이터는 7개월에 고장 4건입니다. 100건을 모으려면
    · 같은 열차에서 → 약 17년
    · 열차 100대에서 → 약 2개월
  즉 <현장에서는 '한 대를 오래' 가 아니라 '여러 대를 동시에' 가 답입니다.
  단일 설비 데이터로 예지보전 모델을 검증하겠다는 계획은 대개 실패합니다.""".replace("<",""))

    Path("reports").mkdir(exist_ok=True)
    r.to_csv("reports/a3_event_eval.csv", index=False)


if __name__ == "__main__":
    main()
