"""정상만 보고 배운다 — 비지도 이상탐지.

왜 지도학습이 아닌가. 고장 사건이 4건뿐이기 때문이다.
시간순으로 나누면 학습 구간에 2건, 평가 구간에 2건이다.
사건 2건으로 '고장의 패턴' 을 배우는 건 불가능하다.

대신 이렇게 한다.
  1) 정상 구간만으로 '평소' 를 배운다
  2) 새 데이터가 그 '평소' 로 얼마나 잘 설명되는지 잰다
  3) 설명이 안 되면 이상

오토인코더가 그 일을 한다. 입력을 좁은 층으로 압축했다가 되살리는데,
정상 데이터로만 학습하면 정상은 잘 되살리고 이상은 못 되살린다.
그 '되살리기 오차' 가 곧 이상 점수다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEAT = Path("data/processed/features.parquet")
SEED = 42
SPLIT = pd.Timestamp("2020-06-01")  # 앞: F1·F2 / 뒤: F3·F4
TRAIN_SUBSAMPLE = 10  # 학습은 10행에 1개만 써도 충분하다(10초→100초)

DROP = ["timestamp", "event", "is_fail"]


def main():
    f = pd.read_parquet(FEAT)
    cols = [c for c in f.columns if c not in DROP]
    tr = f[f.timestamp < SPLIT]
    te = f[f.timestamp >= SPLIT]

    print("[분할] 시간순")
    for name, d in [("train (~2020-05-31)", tr), ("test  (2020-06-01~)", te)]:
        ev = sorted(x for x in d.event.unique() if x)
        print(
            f"  {name}  {len(d):>9,}행 · 고장 {int(d.is_fail.sum()):>6,}행 · 사건 {ev}"
        )

    # ★ 학습에는 '정상' 만 쓴다. 고장 구간은 아예 보여주지 않는다.
    tr_norm = tr[tr.is_fail == 0]
    X_fit = tr_norm[cols].to_numpy(dtype="float32")[::TRAIN_SUBSAMPLE]
    print(
        f"\n[학습] 정상 구간 {len(tr_norm):,}행 중 {len(X_fit):,}행 사용"
        f" (1/{TRAIN_SUBSAMPLE} 추출)"
    )

    sc = StandardScaler().fit(X_fit)
    Z_fit = sc.transform(X_fit)
    Z_te = sc.transform(te[cols].to_numpy(dtype="float32"))
    y_te = te.is_fail.to_numpy()

    scores = {}

    # 0. 규칙 — 압축기 압력(TP2) 하나만 본다  - 기준선 !!
    scores["0. 규칙: TP2 값"] = te["TP2"].to_numpy(dtype="float64")

    # 1. PCA 재구성오차 — 선형 오토인코더나 마찬가지다
    pca = PCA(n_components=8, random_state=SEED).fit(Z_fit)
    rec = pca.inverse_transform(pca.transform(Z_te))
    scores["1. PCA 재구성오차(8차원)"] = np.mean((Z_te - rec) ** 2, axis=1)
    print(
        f"[PCA ] 8차원이 분산의 {pca.explained_variance_ratio_.sum() * 100:.1f}% 를 설명"
    )
    # 72개를 8개로 압축했다 되살림. 이 때 설명이 잘 안 되면 복원 오차가 큼 -> 그게 이상 점수
    # 행마다 72개 오차의 평균

    # 2. MLP 오토인코더 — 입력을 그대로 되살리게 학습한다
    ae = MLPRegressor(
        hidden_layer_sizes=(32, 8, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=40,
        early_stopping=True,  # 학습 데이터 10% 떼서 검증 손실이 5번 연속 안 줄면 멈춤
        n_iter_no_change=5,
        random_state=SEED,
    )
    ae.fit(Z_fit, Z_fit)  # ★ 목표가 입력 자신이다
    rec = ae.predict(Z_te)
    scores["2. MLP 오토인코더(32-8-32)"] = np.mean((Z_te - rec) ** 2, axis=1)
    print(f"[AE  ] {ae.n_iter_} epoch 학습 · 최종 손실 {ae.loss_:.5f}")

    # 3. 지도학습 — 학습 구간의 고장 2건으로 배워 본다
    from sklearn.linear_model import LogisticRegression

    Z_tr_all = sc.transform(tr[cols].to_numpy(dtype="float32")[::TRAIN_SUBSAMPLE])
    y_tr_all = tr.is_fail.to_numpy()[::TRAIN_SUBSAMPLE]
    sup = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    ).fit(Z_tr_all, y_tr_all)
    scores["3. 지도학습 로지스틱(사건 2건)"] = sup.predict_proba(Z_te)[:, 1]

    print("\n" + "=" * 78)
    print("행 단위 평가 (test 2020-06-01 ~ 2020-09-01)")
    print("=" * 78)
    base = y_te.mean()
    rows = []
    for name, s in scores.items():
        rows.append(
            {
                "모델": name,
                "ROC-AUC": roc_auc_score(y_te, s),
                "PR-AUC": average_precision_score(y_te, s),
            }
        )
    r = pd.DataFrame(rows)
    print("\n" + r.round(4).to_string(index=False))
    print(f"\n  ※ PR-AUC 무작위 기준선 = 고장 비율 {base:.5f}")
    print("  ※ 이 표만 보고 판단하면 안 된다. 왜인지는 5장에서.")

    Path("reports").mkdir(exist_ok=True)
    r.to_csv("reports/model_scores_row.csv", index=False)
    np.save("data/processed/y_test.npy", y_te)
    te[["timestamp", "event", "is_fail"]].to_parquet(
        "data/processed/test_meta.parquet", index=False
    )
    for i, (name, s) in enumerate(scores.items()):
        np.save(f"data/processed/score_{i}.npy", s)
    json.dump(
        {"names": list(scores.keys()), "split": str(SPLIT), "seed": SEED},
        open("reports/train_meta.json", "w"),
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    main()

# [PRINT]
#                    모델  ROC-AUC  PR-AUC
#          0. 규칙: TP2 값   0.8833  0.1092
#     1. PCA 재구성오차(8차원)   0.7305  0.0719
# 2. MLP 오토인코더(32-8-32)   0.7362  0.0658
#   3. 지도학습 로지스틱(사건 2건)   0.9144  0.7265

#   ※ PR-AUC 무작위 기준선 = 고장 비율 0.02954
