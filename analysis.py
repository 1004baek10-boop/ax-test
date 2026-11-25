"""③ 병합·지표 (F3, F4) 및 ④ 시각화 (F5).

계획서 5.3 설계: merge_data(), calc_ratio(), calc_vulnerability(), rank_regions()
"""
import numpy as np
import pandas as pd

KEY = "구"  # 병합 기준 키 (계획서 부록 3번: 지역명 기준으로 확정)



def merge_data(supply, demand):
    """공급·수요를 지역 기준으로 병합한다. 매칭되지 않은 지역은 경고로 출력한다 (계획서 10.2)."""
    merged = supply.merge(demand, on=KEY, how="outer", indicator=True)
    unmatched = merged[merged["_merge"] != "both"][KEY].tolist()
    if unmatched:
        print(f"[경고] 매칭되지 않은 지역: {unmatched}")
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    return merged


def calc_ratio(df):
    """수용률(%) = 총수용인원 ÷ 총인구 × 100 (F3)."""
    out = df.copy()
    out["수용률(%)"] = (out["총수용인원"] / out["총인구"] * 100).round(1)
    return out


def calc_vulnerability(df):
    """취약도 점수(0~100) = 고령비율 ÷ 수용률을 0~100으로 정규화 (F4).

    ※ 산식은 계획서 부록 2번 '취약도 점수 산식' 확정 전 임시안이다.
       점수가 높을수록 고령 인구 비중이 크고 대피 공급이 적은 지역이다.
    """
    out = df.copy()
    raw = out["고령비율(%)"] / out["수용률(%)"]
    span = raw.max() - raw.min()
    out["취약도 점수"] = ((raw - raw.min()) / span * 100).round(1) if span else 0.0
    return out


def rank_regions(df):
    """취약도 점수가 높은 순으로 정렬하고 순위를 붙인다 (F4 출력: 취약지역 순위표)."""
    out = df.sort_values("취약도 점수", ascending=False).reset_index(drop=True)
    out.insert(0, "순위", np.arange(1, len(out) + 1))
    return out


def plot_results(ranked, save_path=None):
    """지역별 수용률과 취약도 점수를 막대 그래프로 그린다 (F5).

    matplotlib은 그래프를 그릴 때만 불러온다. 설치되어 있지 않아도
    공급·수요·병합·지표 계산은 그대로 실행된다.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[안내] matplotlib이 없어 그래프를 건너뜁니다. requirements.txt에 matplotlib을 추가하세요.")
        return None

    # 한글 폰트 후보 (코랩: NanumGothic 설치 후 사용, 로컬: 맑은 고딕 등)
    plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(ranked[KEY], ranked["수용률(%)"], color="#4C78A8")
    axes[0].set_title("지역별 대피시설 수용률 (%)")
    axes[0].set_ylabel("수용률 (%)")
    axes[0].axhline(100, color="gray", linestyle="--", linewidth=1)

    axes[1].bar(ranked[KEY], ranked["취약도 점수"], color="#E45756")
    axes[1].set_title("지역별 취약도 점수 (높을수록 취약)")
    axes[1].set_ylabel("취약도 점수 (0~100)")

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
