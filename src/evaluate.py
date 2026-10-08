"""사건 단위 평가 도구.

행 단위 지표(PR-AUC 등)는 이 문제에서 오해를 부른다.
고장 한 건이 52시간이면 그 안에 17,315행이 들어 있다. 행으로 세면 표본이 많아 보이지만
독립적인 사건은 하나다. 그래서 '몇 건을 잡았나 / 얼마나 미리 잡았나 / 헛경보는 몇 번인가'
세 가지로 잰다. 실무 예지보전 팀이 쓰는 지표이기도 하다.
"""
import numpy as np
import pandas as pd

WIN_1H = 360          # 10초 간격이므로 1시간 = 360행


def alarms(score: np.ndarray, pct: float, persist: float = 0.5,
           win: int = WIN_1H) -> np.ndarray:
    """경보 규칙 — 단일 시점이 아니라 '최근 1시간 중 persist 이상이 임계 초과'.

    단일 시점 초과로 경보하면 하루에 수십 번 울린다(8장에서 실측).
    실제 설비 경보는 거의 항상 지속성을 요구한다.
    """
    thr = np.quantile(score, 1 - pct)
    over = (score >= thr).astype(float)
    run = pd.Series(over).rolling(win, min_periods=win).mean().to_numpy()
    return np.nan_to_num(run) >= persist


def group_alarms(alarm: np.ndarray, gap: int = WIN_1H) -> list[tuple[int, int]]:
    """연속 경보를 하나로 묶는다. gap 행 이상 떨어지면 별개 경보."""
    idx = np.flatnonzero(alarm)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def event_report(ts: np.ndarray, event: np.ndarray, alarm: np.ndarray,
                 lookahead_h: int = 48) -> dict:
    """사건별 탐지 여부와 리드타임(고장 시작 몇 시간 전에 경보했나), 그리고 오경보 수."""
    events = [e for e in pd.unique(event) if e]
    det, inside = {}, np.zeros(len(ts), bool)
    for e in events:
        m = event == e
        t0, t1 = ts[m][0], ts[m][-1]
        w = (ts >= t0 - np.timedelta64(lookahead_h, "h")) & (ts <= t1)
        inside |= w
        hit = bool(alarm[w].any())
        lead = np.nan
        if hit:
            first = ts[w][np.flatnonzero(alarm[w])[0]]
            lead = (t0 - first) / np.timedelta64(1, "h")
        det[e] = {"detected": hit, "lead_h": lead}
    groups = group_alarms(alarm)
    fa = sum(1 for s, p in groups if not inside[s:p + 1].any())
    days = (ts[-1] - ts[0]) / np.timedelta64(1, "D")
    return {"events": det, "false_alarms": fa, "fa_per_week": fa / days * 7,
            "n_alarm_groups": len(groups), "days": days}


def daily_rank(ts, score, pct: float = 0.01) -> pd.DataFrame:
    """일별 '임계 초과 비율' 과 그 순위. 어느 날이 가장 이상했나를 본다."""
    thr = np.quantile(score, 1 - pct)
    d = pd.DataFrame({"ts": pd.to_datetime(ts), "over": score >= thr})
    d["day"] = d.ts.dt.floor("D")
    g = d.groupby("day").over.mean().to_frame("over_rate")
    g["rank"] = g.over_rate.rank(ascending=False, method="min").astype(int)
    return g
