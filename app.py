"""청주시 재난대피소 통합 지도앱.

탭 1: 대피소 지도 · 클릭 시 상세 정보 · 꽉 찬 대피소면 가까운 여유 대피소 추천
탭 2: 지역(구)별 수요·공급 분석 · 수용률 · 취약도 순위 (supply/demand/analysis 모듈 사용)
"""
import os

from urllib.parse import quote

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

import analysis
import demand
import supply

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    """data/ 폴더 또는 app.py와 같은 폴더에서 파일을 찾는다. 없으면 None."""
    for folder in (os.path.join(BASE_DIR, "data"), BASE_DIR):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


SHELTER_CSV = find_file("shelters.csv")
POPULATION_CSV = find_file("population_dong.csv")

BBOX = (36.48, 127.28, 36.80, 127.60)  # 청주시 대략적 범위 (위도 min, 경도 min, 위도 max, 경도 max)
CENTER = (36.64, 127.49)
WALK_M_PER_MIN = 80  # 보행 속도 약 80m/분 가정
NO_DONG = "선택 안 함"


# ---------- 데이터 ----------
@st.cache_data
def load_shelters():
    df = pd.read_csv(SHELTER_CSV, encoding="utf-8-sig")
    df["상태"] = np.where(df["여유인원"] > 0, "여유", "꽉참")
    return df


@st.cache_data
def load_dongs():
    return pd.read_csv(POPULATION_CSV, encoding="utf-8-sig")


@st.cache_data
def run_analysis():
    """공급 → 수요 → 병합 → 수용률 → 취약도 → 순위 (main.py 파이프라인과 같은 순서)."""
    shelters, _ = supply.clean_shelters(supply.load_shelters(SHELTER_CSV))
    supply_df = supply.summarize_supply(shelters)
    demand_df = demand.summarize_demand(demand.load_population(POPULATION_CSV))
    merged = analysis.merge_data(supply_df, demand_df)
    merged = analysis.calc_ratio(merged)
    merged = analysis.calc_vulnerability(merged)
    return analysis.rank_regions(merged)


# ---------- 거리 · 추천 ----------
def in_cheongju(lat, lon):
    return BBOX[0] <= lat <= BBOX[2] and BBOX[1] <= lon <= BBOX[3]


def add_distance(df, lat, lon):
    """기준 좌표에서 각 대피소까지 직선거리(m)와 도보 시간(분)을 계산한다."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2 = np.radians(df["위도"].values)
    lon2 = np.radians(df["경도"].values)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * 6371000 * np.arcsin(np.sqrt(a))
    out = df.copy()
    out["거리(m)"] = d.round(0).astype(int)
    out["도보(분)"] = (d / WALK_M_PER_MIN).round(0).astype(int)
    return out


def recommend(df, lat, lon, k=5, only_available=False):
    """기준 위치에서 가까운 대피소 k개. only_available=True면 여유 인원이 있는 곳만."""
    out = add_distance(df, lat, lon)
    if only_available:
        out = out[out["여유인원"] > 0]
    return out.sort_values("거리(m)").head(k)


# ---------- 탭 1: 지도 ----------
def render_map(shelters, dongs):
    dong_labels = [NO_DONG] + (dongs["구"] + " " + dongs["동"]).tolist()
    left, right = st.columns([1, 3])
    with left:
        pick = st.selectbox("기준 동 선택", dong_labels)
        st.caption("동을 고르면 그 동 중심이 기준입니다. '선택 안 함'이면 지도를 클릭한 지점이 기준입니다.")
        st.caption("여유 인원은 시뮬레이션 배정 기준이며 실시간 혼잡도가 아닙니다.")
        st.markdown("**📍 내 위치로 찾기**")
        st.caption("버튼을 누른 뒤 브라우저의 위치 권한을 허용하세요. 동 선택이 '선택 안 함'일 때 적용됩니다.")
        loc = streamlit_geolocation()

    origin, origin_label = CENTER, "청주시 중심"
    if pick != NO_DONG:
        r = dongs.iloc[dong_labels.index(pick) - 1]
        origin, origin_label = (r["위도"], r["경도"]), pick

    # GPS 위치 (동 미선택 시). 청주 범위 밖이면 안내만 하고 기준은 바꾸지 않는다.
    gps_lat, gps_lon = (loc or {}).get("latitude"), (loc or {}).get("longitude")
    if pick == NO_DONG and gps_lat is not None and gps_lon is not None:
        if in_cheongju(gps_lat, gps_lon):
            origin, origin_label = (gps_lat, gps_lon), "내 위치 (GPS)"
        else:
            st.warning("현재 위치가 청주시 범위 밖입니다. 동을 선택하거나 지도를 클릭해 주세요.")

    m = folium.Map(location=CENTER, zoom_start=11)
    m.fit_bounds([[BBOX[0], BBOX[1]], [BBOX[2], BBOX[3]]])
    for _, row in shelters.iterrows():
        popup = folium.Popup(
            f"<b>{row['시설명']}</b><br>{row['주소']}<br>"
            f"수용 {row['수용인원']:,}명 / 여유 {row['여유인원']:,}명 ({row['상태']})",
            max_width=320,
        )
        folium.Marker(
            [row["위도"], row["경도"]],
            tooltip=row["시설명"],
            popup=popup,
            icon=folium.Icon(color="red" if row["상태"] == "꽉참" else "green", icon="home"),
        ).add_to(m)
    folium.Marker(origin, tooltip=f"기준: {origin_label}", icon=folium.Icon(color="blue", icon="user")).add_to(m)

    with right:
        out = st_folium(m, width=900, height=560, key="shelter_map")

    # 지도 클릭으로 기준 위치 지정 (동 미선택 시, 청주 범위 안만)
    click = out.get("last_clicked")
    if pick == NO_DONG and click:
        if in_cheongju(click["lat"], click["lng"]):
            origin, origin_label = (click["lat"], click["lng"]), "지도 클릭 위치"
        else:
            st.warning("청주시 범위 밖입니다. 청주시 안을 클릭해 주세요.")

    # 마커 클릭으로 대피소 선택
    sel = None
    obj = out.get("last_object_clicked")
    if obj:
        dist = (shelters["위도"] - obj["lat"]).abs() + (shelters["경도"] - obj["lng"]).abs()
        i = dist.idxmin()
        if dist[i] < 1e-6:
            sel = shelters.loc[i]

    st.subheader(f"기준 위치: {origin_label}")
    if sel is not None:
        st.markdown(f"### {sel['시설명']}")
        st.write(
            f"**주소:** {sel['주소']}  |  **지상/지하:** {sel['지상지하']}  |  "
            f"**대형시설:** {'예' if sel['대형시설'] else '아니오'}"
        )
        st.write(
            f"**수용 인원:** {sel['수용인원']:,}명  |  **배정 인원:** {sel['클래식_배정']:,}명  |  "
            f"**여유:** {sel['여유인원']:,}명"
        )
        if sel["상태"] == "꽉참":
            st.error("이 대피소는 꽉 찼습니다. 아래 가까운 여유 대피소를 이용해 주세요.")
            alt = recommend(
                shelters[shelters["시설명"] != sel["시설명"]], origin[0], origin[1], k=3, only_available=True
            )
            st.markdown("**추천 대피소 (기준 위치에서 가까운 여유 대피소)**")
            st.dataframe(alt[["시설명", "주소", "여유인원", "거리(m)", "도보(분)"]], hide_index=True)
        else:
            st.success("여유가 있는 대피소입니다.")

    if sel is not None:
        st.markdown("**🚶 길찾기 (외부 지도 앱)**")
        d_lat, d_lon, d_name = sel["위도"], sel["경도"], sel["시설명"]
        g_url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={origin[0]},{origin[1]}&destination={d_lat},{d_lon}&travelmode=walking"
        )
        k_url = f"https://map.kakao.com/link/to/{quote(d_name)},{d_lat},{d_lon}"
        c1, c2 = st.columns(2)
        with c1:
            st.link_button("Google 지도 도보 길찾기", g_url)
        with c2:
            st.link_button("카카오맵 길찾기", k_url)
        st.caption(f"Google 지도는 출발지를 현재 기준 위치({origin_label})로 설정합니다. 카카오맵은 별도 출발지 설정이 필요하면 앱에서 현재 위치를 사용하세요.")

    st.subheader("가까운 대피소 목록")
    near = recommend(shelters, origin[0], origin[1], k=10)
    st.dataframe(
        near[["시설명", "주소", "수용인원", "여유인원", "상태", "거리(m)", "도보(분)"]],
        hide_index=True,
    )


# ---------- 탭 2: 지역 분석 ----------
def render_stats():
    ranked = run_analysis()
    st.subheader("지역(구)별 결과 · 취약지역 순위")
    st.dataframe(ranked, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**지역별 수용률 (%)**")
        st.bar_chart(ranked.set_index("구")["수용률(%)"])
    with c2:
        st.markdown("**지역별 취약도 점수 (높을수록 취약)**")
        st.bar_chart(ranked.set_index("구")["취약도 점수"])

    st.caption(
        "수용률 = 총수용인원 ÷ 총인구 × 100. 취약도 = 고령비율 ÷ 수용률을 0~100으로 정규화한 임시 산식입니다."
    )


# ---------- 메인 ----------
def main():
    st.set_page_config(page_title="청주 재난대피소 통합 지도", layout="wide")
    st.title("청주시 재난대피소 통합 지도")

    # 데이터 파일이 없으면 화면에 바로 알려준다 (Streamlit Cloud 로그의 마스킹 방지)
    missing = [n for n, p in [("shelters.csv", SHELTER_CSV), ("population_dong.csv", POPULATION_CSV)] if p is None]
    if missing:
        st.error("다음 파일을 찾지 못했습니다: " + ", ".join(missing))
        st.info("저장소에 data/ 폴더를 만들고 그 안에 CSV 두 개를 올리거나, app.py와 같은 위치에 올려 주세요.")
        st.stop()

    shelters = load_shelters()
    dongs = load_dongs()

    tab_map, tab_stat = st.tabs(["🗺 대피소 지도·추천", "📊 지역 수요·공급 분석"])
    with tab_map:
        render_map(shelters, dongs)
    with tab_stat:
        render_stats()


if __name__ == "__main__":
    main()
