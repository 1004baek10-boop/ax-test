"""② 수요 집계 (F2): 주민등록인구 통계 → 지역(구)별 총인구·65세 이상 인구·고령 비율.

계획서 5.3 설계: load_population(), summarize_demand()
동 단위 자료를 구 단위로 합산한다.
"""
import pandas as pd

REQUIRED_COLUMNS = ["구", "동", "총인구", "65세이상"]


def load_population(path):
    """동별 인구 CSV를 읽고 필수 열이 있는지 확인한다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"인구 CSV에 필요한 열이 없습니다: {missing}")
    df["총인구"] = pd.to_numeric(df["총인구"], errors="coerce")
    df["65세이상"] = pd.to_numeric(df["65세이상"], errors="coerce")
    return df


def summarize_demand(df):
    """구별 총인구, 65세 이상 인구, 고령 비율(65세 이상 ÷ 총인구 × 100, %)을 계산한다."""
    summary = (
        df.groupby("구")[["총인구", "65세이상"]]
        .sum()
        .reset_index()
    )
    summary["고령비율(%)"] = (summary["65세이상"] / summary["총인구"] * 100).round(1)
    return summary
