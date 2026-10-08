"""전체 파이프라인을 한 줄로 재현한다.  python run_all.py"""
import subprocess, sys, time
from pathlib import Path

STEPS = [
    ("원본 다운로드·해제",   "src/download_data.py"),
    ("CSV → parquet",      "src/build_db.py"),
    ("★ 라벨 (PDF 표 → 코드)", "src/labels.py"),
    ("시계열 피처",          "src/features.py"),
    ("학습 (정상만)",        "src/train.py"),
    ("a1 EDA",              "analysis/a1_eda.py"),
    ("a2 행 단위 지표의 거짓",  "analysis/a2_row_metric_lie.py"),
    ("a3 사건 단위 평가",     "analysis/a3_event_eval.py"),
    ("a4 이상 vs 고장",      "analysis/a4_anomaly_vs_failure.py"),
    ("a5 경보 설계",         "analysis/a5_alarm_design.py"),
    ("a6 비용",              "analysis/a6_cost.py"),
]

def main() -> int:
    Path("reports").mkdir(exist_ok=True)
    log, t0 = [], time.time()
    for i, (name, script) in enumerate(STEPS, 1):
        print(f"\n{'='*78}\n[{i}/{len(STEPS)}] {name}  ({script})\n{'='*78}")
        t = time.time(); r = subprocess.run([sys.executable, script]); dt = time.time()-t
        log.append((name, script, dt))
        if r.returncode:
            print(f"\n[중단] {script} 실패(코드 {r.returncode})"); return r.returncode
        print(f"\n  … {dt:.1f}초")
    print(f"\n{'='*78}\n전체 완료 — {time.time()-t0:.1f}초\n{'='*78}")
    for n, s, d in log: print(f"  {n:24s} {d:>7.1f}초   {s}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
