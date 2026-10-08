"""★★ 라벨을 직접 만든다.

이 데이터에는 정답 열이 없다. 공식 설명 문서(PDF)의 표에 고장 기간이 적혀 있을 뿐이다.
그 표를 코드로 옮기는 게 이 파일이다.

그리고 여기서 가장 중요한 사실 하나 —
  라벨이 붙는 행은 29,954개지만, 독립적인 고장 '사건' 은 4건뿐이다.
행 단위로 평가하면 3만 개의 표본이 있는 것처럼 착각하게 된다. 실제 표본은 4다.
"""

from pathlib import Path

import pandas as pd

SRC = Path("data/raw/metro.parquet")
OUT = Path("data/processed/labeled.parquet")

# 출처: MetroPT-3 공식 데이터 설명 문서 "Failure Information" 표
# 네 건 모두 공기 누출(Air leak), 심각도 High stress.
FAILURES = [
    ("F1", "2020-04-18 00:00", "2020-04-18 23:59", "정비 기록 없음"),
    (
        "F2",
        "2020-05-29 23:30",
        "2020-05-30 06:00",
        "정비 (문서상 30Apr 12:00 — 오기로 보임)",
    ),
    ("F3", "2020-06-05 10:00", "2020-06-07 14:30", "정비 8Jun 16:00"),
    ("F4", "2020-07-15 14:30", "2020-07-15 19:00", "정비 16Jul 00:00"),
]


def build() -> pd.DataFrame:
    df = pd.read_parquet(SRC)
    df["event"] = ""
    print("[라벨] 공식 문서의 고장 표를 옮긴다\n")
    for name, s, e, note in FAILURES:
        m = (df.timestamp >= pd.Timestamp(s)) & (df.timestamp <= pd.Timestamp(e))
        df.loc[m, "event"] = name
        hours = (pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 3600
        print(f"  {name}  {s} ~ {e}  {hours:>5.1f}시간  {int(m.sum()):>7,}행   {note}")

    df["is_fail"] = (df.event != "").astype("int8")
    n, nf = len(df), int(df.is_fail.sum())
    print(f"\n  전체 {n:,}행 중 고장 {nf:,}행 = {nf / n * 100:.3f}%")
    print(f"  ★ 그런데 독립 사건은 {len(FAILURES)}건뿐이다.")
    print(
        f"    행 단위로 세면 표본이 {nf:,}개처럼 보이지만, 실제로는 {len(FAILURES)}개다."
    )
    print(f"    이 구분이 5장(평가)과 6장(분할)을 통째로 결정한다.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    return df


# 고장 행은 3만 개인데 고장 사건은 4개.
# 10초 간격이라 한 사건 안의 행들은 거의 같은 값이라 3만 개가 아니라 사실상 4개의 표본.

if __name__ == "__main__":
    build()
