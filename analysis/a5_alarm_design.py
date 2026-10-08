"""★★ 경보 설계 — 모델보다 경보 규칙이 오경보를 정한다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import WIN_1H, alarms, event_report, group_alarms  # noqa: E402


def main():
    meta = pd.read_parquet("data/processed/test_meta.parquet")
    names = json.load(open("reports/train_meta.json"))["names"]
    ts, ev = meta.timestamp.to_numpy(), meta.event.to_numpy()
    s = np.load("data/processed/score_2.npy")
    days = (ts[-1] - ts[0]) / np.timedelta64(1, "D")

    print("=" * 84)
    print("1. 단일 시점 임계 초과로 경보하면")
    print("=" * 84)
    rows = []
    for pct in (0.005, 0.01, 0.02, 0.05):
        thr = np.quantile(s, 1 - pct)
        a = s >= thr
        gr = group_alarms(a, gap=WIN_1H)
        rows.append({"임계(상위)": f"{pct:.1%}", "경보 횟수": len(gr),
                     "하루 평균": round(len(gr) / days, 1)})
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("""
  상위 1% 임계만 써도 하루에 여러 번 울립니다. 운영팀은 사흘이면 무시하기 시작합니다.
  이걸 경보 피로(alarm fatigue)라고 하고, 실제 설비 사고의 흔한 배경입니다.""")

    print("\n" + "=" * 84)
    print("2. 지속성을 요구하면 — '최근 1시간 중 X% 이상이 초과할 때만'")
    print("=" * 84)
    rows = []
    for pct in (0.005, 0.01, 0.02, 0.05):
        for persist in (0.2, 0.5, 0.8):
            a = alarms(s, pct, persist)
            rep = event_report(ts, ev, a)
            det = {e: (f"{d['lead_h']:+.1f}h" if d["detected"] else "미탐")
                   for e, d in rep["events"].items()}
            rows.append({"임계": f"{pct:.1%}", "지속 요구": f"{persist:.0%}",
                         **det, "오경보": rep["false_alarms"],
                         "주당": round(rep["fa_per_week"], 2)})
    r = pd.DataFrame(rows)
    print("\n" + r.to_string(index=False))
    print("""
  같은 모델, 같은 점수입니다. 경보 규칙만 바꿨습니다.
  단일 시점이면 하루 수 회, 지속성 50% 를 요구하면 주 0.2회 수준까지 내려갑니다.

  ★ 대신 리드타임을 잃습니다. 1시간 지속을 기다리는 동안 고장이 진행되니까요.
    '얼마나 빨리' 와 '얼마나 조용히' 는 맞바꾸는 관계입니다.
    어디서 균형을 잡을지는 다음 장의 비용이 정합니다.""")

    print("\n" + "=" * 84)
    print("3. 실무에서 쓰는 다른 장치들")
    print("=" * 84)
    print("""
  · 히스테리시스   — 켜지는 임계와 꺼지는 임계를 다르게 둔다. 경보가 깜빡이는 걸 막는다.
  · 쿨다운        — 한 번 울리면 N시간 동안 같은 설비에 재경보하지 않는다.
  · 등급          — '주의 / 경고 / 위험' 3단계. 낮은 등급은 알림만, 높은 등급만 호출.
  · 정비 캘린더 연동 — 계획 정비 기간은 아예 경보에서 제외한다(7장의 6월 8일 문제).

  이 넷은 모델이 아니라 운영 설계입니다. 그리고 실제 현장에서 모델 성능보다
  이쪽이 도입 성패를 더 자주 가릅니다.""")

    Path("reports").mkdir(exist_ok=True)
    r.to_csv("reports/a5_alarm_design.csv", index=False)


if __name__ == "__main__":
    main()
