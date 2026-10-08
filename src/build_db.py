"""208MB CSV 를 parquet 으로 굳히고, 문서와 실제가 맞는지 확인한다.

공식 문서와 실제 파일이 어긋나는 지점이 두 개 있다. 둘 다 여기서 잡는다.
"""
from pathlib import Path

import pandas as pd

CSV = Path("data/raw/MetroPT3(AirCompressor).csv")
OUT = Path("data/raw/metro.parquet")

DOC_ROWS = 15_169_480     # 문서에 적힌 값
DOC_HZ = 1                # 문서에 적힌 샘플링 주기


def main():
    df = pd.read_csv(CSV, index_col=0, parse_dates=["timestamp"])
    print(f"[read] {len(df):,} 행 × {df.shape[1]} 열")
    print(f"       기간 {df.timestamp.min()} ~ {df.timestamp.max()}")

    gap = df.timestamp.diff().dt.total_seconds().dropna()
    span = (df.timestamp.max() - df.timestamp.min()).total_seconds()

    print("\n[문서와 대조] — 공식 설명 문서가 항상 맞지는 않는다")
    print(f"  문서: {DOC_ROWS:,}행 · {DOC_HZ}Hz(1초 간격)")
    print(f"  실제: {len(df):,}행 · 중앙값 {gap.median():.0f}초 간격")
    print(f"       → 문서의 {DOC_ROWS/len(df):.0f}분의 1. 공개본은 다운샘플된 것으로 보인다.")
    print(f"       → 1초 간격이라면 {span:,.0f}행이어야 한다(실제의 {span/len(df):.0f}배)")

    big = gap[gap > 60]
    print(f"\n  60초 초과 결측 구간 {len(big)}회 · 최대 {big.max()/3600:.1f}시간")
    print("  → 연속 시계열이 아니다. rolling 창을 '시간' 이 아니라 '행 수' 로 잡는 이유다.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    mb_csv, mb_pq = CSV.stat().st_size / 1e6, OUT.stat().st_size / 1e6
    print(f"\n[ok  ] {OUT}   CSV {mb_csv:,.0f}MB → parquet {mb_pq:,.0f}MB"
          f"  ({mb_csv/mb_pq:.1f}배)")


if __name__ == "__main__":
    main()
