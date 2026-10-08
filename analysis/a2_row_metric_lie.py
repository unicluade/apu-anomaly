"""★★ 행 단위 지표는 표본이 4개인 문제를 15만 개처럼 보이게 한다.

PR-AUC 0.7745 와 0.0718. 10.8배 차이다. 결론이 날 것 같다.
그런데 그 숫자를 만든 평가 표본은 고장 사건 2건이다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    meta = pd.read_parquet("data/processed/test_meta.parquet")
    y = np.load("data/processed/y_test.npy")
    names = json.load(open("reports/train_meta.json"))["names"]

    print("=" * 80)
    print("1. 행 단위 지표 — 숫자만 보면 결론이 난 것 같다")
    print("=" * 80)
    rows = []
    for i, n in enumerate(names):
        s = np.load(f"data/processed/score_{i}.npy")
        rows.append({"모델": n, "ROC-AUC": roc_auc_score(y, s),
                     "PR-AUC": average_precision_score(y, s)})
    r = pd.DataFrame(rows)
    print("\n" + r.round(4).to_string(index=False))
    best = r.loc[r["PR-AUC"].idxmax()]
    worst = r.loc[r["PR-AUC"].idxmin()]
    print(f"""
  1등 {best['모델']}  PR-AUC {best['PR-AUC']:.4f}
  꼴등 {worst['모델']}  PR-AUC {worst['PR-AUC']:.4f}
  → {best['PR-AUC']/worst['PR-AUC']:.1f}배 차이. 지도학습의 압승으로 보인다.""")

    print("\n" + "=" * 80)
    print("2. 그런데 이 표를 만든 '표본' 이 몇 개인가")
    print("=" * 80)
    ev = [e for e in pd.unique(meta.event) if e]
    print(f"\n  평가 구간 행 수      {len(y):,}")
    print(f"  그중 고장 라벨 행    {int(y.sum()):,}  ({y.mean()*100:.2f}%)")
    print(f"  ★ 독립 고장 사건     {len(ev)}건  {ev}")
    for e in ev:
        m = meta.event == e
        h = (meta.loc[m, "timestamp"].max() - meta.loc[m, "timestamp"].min()).total_seconds() / 3600
        print(f"      {e}  {int(m.sum()):>6,}행 = {h:.1f}시간 연속")
    print(f"""
  ★★ 고장 행 {int(y.sum()):,}개는 독립 표본이 아닙니다.
     한 사건 안의 인접한 행들은 10초 간격이라 거의 같은 값입니다.
     52시간짜리 사건 하나가 17,315행을 만들 뿐, 정보량은 '사건 1건' 입니다.

     PR-AUC 는 이 17,315행을 17,315개의 독립 표본으로 셉니다.
     그래서 한 사건을 잘 맞히면 지표가 통째로 좋아지고,
     다른 사건을 통째로 놓쳐도 지표에 거의 안 나타납니다.""")

    print("\n" + "=" * 80)
    print("3. 사건별로 쪼개 보면 — 같은 모델, 완전히 다른 그림")
    print("=" * 80)
    rows = []
    for i, n in enumerate(names):
        s = np.load(f"data/processed/score_{i}.npy")
        q = np.quantile(s, 0.99)
        out = {"모델": n}
        for e in ev:
            m = (meta.event == e).to_numpy()
            out[f"{e} 초과율"] = (s[m] >= q).mean()
        out["정상 구간 초과율"] = (s[(meta.event == "").to_numpy()] >= q).mean()
        rows.append(out)
    r2 = pd.DataFrame(rows)
    print("\n  (99분위 임계를 넘은 행의 비율)")
    print(r2.round(4).to_string(index=False))
    print("""
  지도학습은 F3 에서 압도적이고 F4 에서는 약합니다.
  오토인코더는 반대입니다. 행 단위 PR-AUC 는 이 차이를 '큰 사건(F3, 52시간)' 쪽으로
  통째로 기울여 계산합니다. 사건 길이가 지표를 정하는 셈입니다.""")

    Path("reports").mkdir(exist_ok=True)
    r.to_csv("reports/a2_row_metrics.csv", index=False)
    r2.to_csv("reports/a2_by_event.csv", index=False)


if __name__ == "__main__":
    main()
