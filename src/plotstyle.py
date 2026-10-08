"""matplotlib 한글 폰트. 안 잡으면 그래프의 모든 한글이 네모(□)로 나온다.

matplotlib 기본 폰트(DejaVu Sans)에는 한글 글리프가 없다.
OS 마다 깔려 있는 한글 폰트 이름이 달라서, 있는 것 중 첫 번째를 쓴다.
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

CANDIDATES = [
    "AppleGothic",          # macOS 기본
    "Malgun Gothic",        # Windows 기본
    "NanumGothic",          # 리눅스에 흔히 설치
    "NanumBarunGothic",
    "Noto Sans CJK KR",
]


def use_korean() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CANDIDATES:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False   # 마이너스가 깨지는 것도 같이 막는다
            return name
    print("[warn] 한글 폰트를 못 찾았습니다. 그래프의 한글이 □ 로 나옵니다.")
    print("       리눅스: sudo apt install fonts-nanum && rm -rf ~/.cache/matplotlib")
    return matplotlib.rcParams["font.family"][0]


def apply(dpi: int = 130) -> str:
    name = use_korean()
    plt.rcParams.update({"figure.dpi": dpi, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .25, "axes.spines.top": False,
                         "axes.spines.right": False})
    return name


if __name__ == "__main__":
    print("적용된 폰트:", apply())
