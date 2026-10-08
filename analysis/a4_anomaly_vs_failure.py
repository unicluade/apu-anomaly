"""★★★ 이상탐지는 '고장' 을 찾지 않는다. '이상' 을 찾는다.

라벨 없이 학습한 모델에게 "평소와 다른 날" 을 물으면
고장·정비·가동정지·센서교체가 전부 섞여서 나온다.
그중 무엇이 고장인지는 모델이 모른다. 우리도 문서를 봐야 안다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import daily_rank  # noqa: E402

# 공식 문서 "Failure Information" 표의 정비(Report) 열
KNOWN = {
    "2020-06-05": "고장 F3 시작",
    "2020-06-06": "고장 F3",
    "2020-06-07": "고장 F3 종료",
    "2020-06-08": "★ 정비 (문서: 8Jun 16:00)",
    "2020-07-15": "고장 F4",
    "2020-07-16": "★ 정비 (문서: 16Jul 00:00)",
    "2020-07-17": "정비 다음날",
}


def main():
    meta = pd.read_parquet("data/processed/test_meta.parquet")
    names = json.load(open("reports/train_meta.json"))["names"]
    s = np.load("data/processed/score_2.npy")      # MLP 오토인코더
    g = daily_rank(meta.timestamp.to_numpy(), s, pct=0.01)

    print("=" * 78)
    print(f"'{names[2]}' 가 꼽은 가장 이상한 날 상위 10")
    print("=" * 78)
    print()
    for day, row in g.sort_values("over_rate", ascending=False).head(10).iterrows():
        tag = KNOWN.get(str(day.date()), "")
        print(f"  {row['rank']:>2}위  {day.date()}  초과율 {row['over_rate']*100:5.1f}%   {tag}")

    print("\n" + "=" * 78)
    print("문서에 적힌 날들의 순위")
    print("=" * 78)
    print()
    for day, tag in KNOWN.items():
        d = pd.Timestamp(day)
        if d in g.index:
            print(f"  {day}  {g.loc[d,'rank']:>2}위 / {len(g)}일   {tag}")

    r1 = g.loc[pd.Timestamp("2020-06-08"), "rank"]
    f3 = [int(g.loc[pd.Timestamp(d), "rank"]) for d in
          ("2020-06-05", "2020-06-06", "2020-06-07")]
    print(f"""
  ★★ 1위가 {r1}위인 2020-06-08 — 이 날은 <고장이 아니라 정비>입니다.
     문서의 Report 열에 "Maintenance on 8Jun at 16:00" 이라고 적혀 있습니다.

     정작 고장 F3(6/5~6/7)은 91일 중 {f3[0]}위 · {f3[1]}위 · {f3[2]}위로 묻혔습니다.

  왜 이런 일이 생기나.
    정비는 '평소와 다른 상태' 입니다. 압축기를 세우고, 부품을 갈고, 밸브를 여닫습니다.
    센서로 보면 고장보다 훨씬 이상합니다. 이상탐지는 그걸 정확히 잡아냅니다.
    문제는 그게 우리가 찾던 게 아니라는 것뿐입니다.

  실무에서 이게 뜻하는 것
    · 이상탐지 결과는 그대로 경보로 쓰면 안 됩니다. 계획 정비·가동정지 일정을
      <시스템이 알고 있어야> 걸러낼 수 있습니다.
    · '이상 점수 상위 목록' 을 만들고 <사람이 분류하는> 단계가 필요합니다.
      그 분류가 쌓이면 그게 곧 라벨이 되고, 지도학습으로 넘어갈 수 있습니다.
    · 라벨 없는 상태에서 바로 자동 차단·자동 정비지시로 가면 안 됩니다.""".replace("<","").replace(">",""))

    Path("reports").mkdir(exist_ok=True)
    g.to_csv("reports/a4_daily_rank.csv")


if __name__ == "__main__":
    main()
