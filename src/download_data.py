"""MetroPT-3 — 지하철 공기압축기 센서 로그를 받는다.

UCI #791. 인증키·계정 불필요. 라이선스 CC BY 4.0.
포르투갈 포르투 지하철 열차의 공기생산장치(APU) 센서를 2020년 2~8월 기록한 실데이터다.

zip 안에 두 개가 들어 있다.
  · MetroPT3(AirCompressor).csv   — 센서 로그 (208MB)
  · Data Description_Metro.pdf    — ★ 고장 기간 표가 여기 있다. 라벨이 CSV 에 없다.
"""
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
RAW = Path("data/raw")
ZIP = RAW / "metropt3.zip"
CSV = RAW / "MetroPT3(AirCompressor).csv"
PDF = RAW / "Data Description_Metro.pdf"

EXPECT_ZIP = 218_381_995      # 실측
EXPECT_CSV = 218_300_507      # 실측


def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and ZIP.stat().st_size == EXPECT_ZIP:
        print(f"[skip] 이미 있음 {ZIP} ({ZIP.stat().st_size:,} bytes)")
        return ZIP
    for attempt in range(1, 4):
        try:
            r = requests.get(URL, timeout=1800, stream=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            n = 0
            with open(ZIP, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            print(f"[get ] {n:,} bytes 수신 (약 218MB)")
            break
        except Exception as e:
            print(f"       시도 {attempt}/3 실패: {type(e).__name__}: {e}")
    else:
        raise SystemExit("다운로드에 3번 실패했습니다.")
    if not zipfile.is_zipfile(ZIP):
        raise SystemExit(f"{ZIP} 가 zip 이 아닙니다. 응답을 확인하세요.")
    return ZIP


def extract() -> None:
    if CSV.exists() and CSV.stat().st_size == EXPECT_CSV:
        print(f"[skip] 이미 풀림 {CSV}")
        return
    with zipfile.ZipFile(ZIP) as z:
        print(f"[zip ] 내용물 {z.namelist()}")
        z.extractall(RAW)
    print(f"[ok  ] {CSV} ({CSV.stat().st_size:,} bytes)")
    print(f"[ok  ] {PDF}  ← 고장 기간 표가 이 문서에 있다")
    print("\n★ 이 데이터에는 정답 열이 없습니다.")
    print("  PDF 의 'Failure Information' 표를 코드로 옮기는 게 src/labels.py 입니다.")


if __name__ == "__main__":
    download()
    extract()
