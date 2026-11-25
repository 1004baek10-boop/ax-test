"""① 공급 집계 (F1): 민방위 대피시설 CSV → 지역(구)별 시설 수·수용인원.

계획서 5.3 설계: load_shelters(), clean_shelters(), summarize_supply()
"""
import pandas as pd

REQUIRED_COLUMNS = ["구", "시설명", "수용인원"]


def load_shelters(path):
    """대피시설 CSV를 읽고 필수 열이 있는지 확인한다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"대피시설 CSV에 필요한 열이 없습니다: {missing}")
    return df


def clean_shelters(df):
    """결측치 처리 기준 (계획서 4.3 데이터 품질).

    - 지역(구) 또는 수용인원이 없는 행은 제외한다.
    - 수용인원은 숫자로 변환하고, 0 이하 값은 제외한다.
    반환: (정리된 데이터프레임, 처리 로그 dict)
    """
    out = df.copy()
    out["수용인원"] = pd.to_numeric(out["수용인원"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["구", "수용인원"])
    out = out[out["수용인원"] > 0].reset_index(drop=True)
    log = {"원본 행수": before, "제외 행수": before - len(out), "정리 후 행수": len(out)}
    return out, log


def summarize_supply(df):
    """구별 시설 수와 수용인원 합계를 집계한다 (F1 출력: 지역별 공급 데이터)."""
    summary = (
        df.groupby("구")
        .agg(시설수=("시설명", "count"), 총수용인원=("수용인원", "sum"))
        .reset_index()
    )
    return summary
