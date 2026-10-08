"""★★ 비용 — 오경보 한 번과 고장 한 번의 값이 다르다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import alarms, event_report  # noqa: E402

# ── 가정 (실무에서는 정비팀·운영팀에서 받는 숫자다) ──────────────
COST_FAILURE = 8_000_000      # 운행 중 고장 1건 (운행 중단·긴급 출동) — 가정
COST_INSPECT = 300_000        # 경보로 인한 점검 1회 — 가정
PREVENT_RATE = 0.7            # 미리 알았을 때 고장을 막을 확률 — 가정
MIN_LEAD_H = 6                # 이보다 늦게 알면 못 막는다 — 가정
# ─────────────────────────────────────────────────────


def main():
    meta = pd.read_parquet("data/processed/test_meta.parquet")
    names = json.load(open("reports/train_meta.json"))["names"]
    ts, ev = meta.timestamp.to_numpy(), meta.event.to_numpy()
    n_events = len([e for e in pd.unique(ev) if e])

    print("=" * 88)
    print("가정")
    print("=" * 88)
    print(f"  운행 중 고장 1건      {COST_FAILURE:>12,}원   ← 가정")
    print(f"  경보 점검 1회         {COST_INSPECT:>12,}원   ← 가정")
    print(f"  미리 알면 막을 확률    {PREVENT_RATE:>12.0%}   ← 가정")
    print(f"  최소 예지 시간        {MIN_LEAD_H:>12}시간   ← 가정 (이보다 늦으면 못 막음)")
    print(f"""
  ★ 손익 계산
    막은 고장 1건      +{COST_FAILURE*PREVENT_RATE:,.0f}원 (기댓값)
    점검 1회           −{COST_INSPECT:,}원
    → 오경보 {COST_FAILURE*PREVENT_RATE/COST_INSPECT:.0f}번까지는 고장 1건 막는 값이 큽니다.
       오경보가 무조건 나쁜 게 아닙니다. 몇 번까지 허용되는지가 계산됩니다.""")

    print("\n" + "=" * 88)
    print("설정별 손익 (평가 기간 3개월 · 고장 2건 기준)")
    print("=" * 88)
    rows = []
    for i, n in enumerate(names):
        s = np.load(f"data/processed/score_{i}.npy")
        for pct in (0.005, 0.01, 0.02, 0.05):
            a = alarms(s, pct)
            rep = event_report(ts, ev, a)
            prevented = sum(1 for d in rep["events"].values()
                            if d["detected"] and d["lead_h"] >= MIN_LEAD_H)
            missed = n_events - prevented
            n_inspect = rep["false_alarms"] + prevented
            benefit = prevented * COST_FAILURE * PREVENT_RATE
            cost = n_inspect * COST_INSPECT + missed * COST_FAILURE
            rows.append({"모델": n, "임계": f"{pct:.1%}",
                         "제때 탐지": prevented, "놓침": missed,
                         "오경보": rep["false_alarms"],
                         "순손익": benefit - cost})
    r = pd.DataFrame(rows)
    print("\n" + r.to_string(index=False))
    best = r.loc[r["순손익"].idxmax()]
    nothing = -n_events * COST_FAILURE
    print(f"""
  아무것도 안 하면   고장 {n_events}건을 그대로 맞음      {nothing:>14,.0f}원
  최선의 설정        {best['모델']} · 임계 {best['임계']}   {best['순손익']:>14,.0f}원
  ─────────────────────────────────────────────
  절감액                                        {best['순손익']-nothing:>14,.0f}원

  ★ 2건 중 1건을 6시간 이상 앞서 잡은 덕입니다. 손익은 여전히 마이너스인데,
    고장 자체가 비싸서 '덜 잃는' 게 최선인 상황이기 때문입니다.

  ★★ 그런데 이 숫자를 그대로 보고하면 안 됩니다.
     사건이 2건이라 <한 건을 더 잡거나 놓치면 결론이 통째로 뒤집힙니다>.
     실제로 표를 보면 '제때 탐지' 가 0건인 설정과 1건인 설정의 차이가
     전부입니다. 2건 중 1건을 맞힌 걸로 '탐지율 50%' 라고 쓰면 과장입니다.""".replace("<","").replace(">",""))

    print("\n" + "=" * 88)
    print("가정이 틀렸다면 — 민감도")
    print("=" * 88)
    rows = []
    s = np.load("data/processed/score_1.npy")      # PCA (F4 를 35시간 전에 잡은 것)
    for cf in (2_000_000, 8_000_000, 30_000_000):
        for ci in (100_000, 300_000, 1_000_000):
            best_p, best_v = None, -1e18
            for pct in (0.005, 0.01, 0.02, 0.05):
                a = alarms(s, pct)
                rep = event_report(ts, ev, a)
                prevented = sum(1 for d in rep["events"].values()
                                if d["detected"] and d["lead_h"] >= MIN_LEAD_H)
                missed = n_events - prevented
                v = (prevented * cf * PREVENT_RATE
                     - (rep["false_alarms"] + prevented) * ci - missed * cf)
                if v > best_v:
                    best_p, best_v = pct, v
            rows.append({"고장비용": cf, "점검비용": ci, "최적 임계": f"{best_p:.1%}",
                         "순손익": round(best_v)})
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("""
  고장이 비쌀수록 임계를 낮춰 더 많이 경보하는 게 이득입니다.
  점검이 비쌀수록 반대고요. 모델이 아니라 이 두 숫자가 운영점을 정합니다.

  ★ 한계: '막을 확률 70%' 와 '최소 예지 6시간' 은 가정입니다.
    실제 값은 정비 이력이 쌓여야 알 수 있고, 그전까지 이 계산은 방향만 보여 줍니다.""")

    Path("reports").mkdir(exist_ok=True)
    r.to_csv("reports/a6_cost.csv", index=False)
    json.dump({"COST_FAILURE": COST_FAILURE, "COST_INSPECT": COST_INSPECT,
               "PREVENT_RATE": PREVENT_RATE, "MIN_LEAD_H": MIN_LEAD_H},
              open("reports/a6_assumptions.json", "w"), indent=2)


if __name__ == "__main__":
    main()
