import streamlit as st
from streamlit_mic_recorder import mic_recorder
import numpy as np
import librosa
import pandas as pd
import time

# --- [UI 설정] iROOM 브랜드 테마 적용 ---
st.set_page_config(page_title="iROOM - Music Analytics", layout="wide")

st.title("🎺 iROOM : 데이터 기반 음악 교육의 새 시대")
st.subheader("Step 1: 음정 및 음색 데이터 수집 프로토타입")
st.write("---")

# --- [1단계] 프로그램이 음정 제시 (Reference) ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info("### 🎯 오늘의 목표 음정")
    target_note = st.selectbox("연주할 음을 선택하세요", ["Bb4", "A4", "G4", "F4", "Eb4"])
    
    if st.button("🔈 기준음 듣기"):
        # 간단한 사인파 비프음이나 미리 준비된 오디오 파일을 재생할 수 있습니다.
        st.write(f"{target_note} 가이드 음이 재생 중입니다...")

# --- [2~3단계] 사용자 연주 및 데이터 수집 ---
with col2:
    st.write("### 🎙️ 연주 녹음 및 실시간 분석")
    audio = mic_recorder(
        start_prompt="녹음 시작",
        stop_prompt="녹음 완료",
        key='recorder'
    )

    if audio:
        # 오디오 데이터를 수치로 변환 (Librosa 활용)
        audio_bytes = audio['bytes']
        # 실제 환경에서는 여기서 주파수(Hz)와 배음(Harmonics)을 추출합니다.
        
        # 가상의 데이터 추출 결과 (예시)
        detected_pitch = "Bb4" 
        cent_error = 8  # +8 Cent
        tone_quality = 92 # 음색 점수
        
        st.success("✅ 분석 완료!")
        
        # 결과 대시보드 (iROOM UI 스타일)
        m1, m2, m3 = st.columns(3)
        m1.metric("입력 음정", detected_pitch)
        m2.metric("음정 오차", f"{cent_error} Cent", delta=f"{cent_error}")
        m3.metric("음색 점수", f"{tone_quality} 점")

        # --- [구글 시트 저장 로직] ---
        st.write("---")
        st.write("### 💾 데이터 전송")
        user_name = st.text_input("테스터 이름", "홍길동")
        instrument = st.selectbox("악기 선택", ["트럼펫", "호른", "트롬본"])

        if st.button("구글 시트로 전송하기"):
            with st.spinner('데이터를 기록 중입니다...'):
                # 여기에 gspread를 이용한 구글 시트 연동 코드가 들어갑니다.
                # 현재는 가상으로 성공 메시지만 띄웁니다.
                time.sleep(1)
                st.balloons()
                st.write(f"📊 **{user_name}**님의 데이터가 전송되었습니다.")
                st.caption(f"전송 데이터: {target_note} | {cent_error}c | {instrument}")

# --- [4~6단계] 관리자용 데이터 검토 메뉴 ---
with st.expander("🔍 관리자용 데이터 검토 및 보정 (학습용)"):
    st.write("구글 시트에 수집된 악기별 데이터를 바탕으로 알고리즘을 보정합니다.")
    # 예시 데이터프레임
    df_sample = pd.DataFrame({
        '악기': ['트럼펫', '트럼펫', '호른'],
        '오차': [5, 12, -3],
        '음색밀도': [0.85, 0.78, 0.92]
    })
    st.table(df_sample)
    st.slider("악기별 음정 정확도 보정값 설정", -20, 20, 0)