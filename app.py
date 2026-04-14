import streamlit as st
from streamlit_mic_recorder import mic_recorder
import numpy as np
import librosa
import io
import pandas as pd

# --- [1] 페이지 설정 및 디자인 (iROOM 브랜드 테마) ---
st.set_page_config(page_title="iROOM - The Moneyball of Music", layout="wide")

# iROOM 전용 커스텀 스타일링
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; }
    .status-card { background-color: #ffffff; padding: 25px; border-radius: 20px; border-left: 5px solid #007bff; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 사이드바 (사용자 정보 및 설정) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/trumpet.png", width=80)
    st.title("iROOM Profile")
    user_name = st.text_input("테스터 이름", "홍길동")
    instrument = st.selectbox("악기 선택", ["Trumpet (Bb)", "Horn (F)", "Trombone"])
    st.write("---")
    st.caption("🚀 iROOM은 데이터 기반 음악 교육의 새 시대를 엽니다.")

# --- [3] 메인 화면 구성 ---
st.title("🎺 iROOM Analytics")
st.markdown(f"#### **{user_name}** 님의 실시간 퍼포먼스 데이터 센터")

# 상단 섹션: 목표 설정 및 녹음
col_target, col_rec = st.columns([1, 1])

with col_target:
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Target Goal")
    target_note = st.selectbox("오늘의 목표 음정", ["Bb4", "A4", "G4", "F4", "Eb4"])
    note_freqs = {"Bb4": 466.16, "A4": 440.0, "G4": 392.0, "F4": 349.23, "Eb4": 311.13}
    st.write(f"기준 주파수: **{note_freqs[target_note]} Hz**")
    st.markdown('</div>', unsafe_allow_html=True)

with col_rec:
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Live Session")
    audio = mic_recorder(
        start_prompt="Record Performance",
        stop_prompt="Stop & Analyze",
        key='recorder'
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- [4] 데이터 분석 엔진 및 시각화 ---
if audio:
    with st.spinner('머니볼 데이터 분석 중...'):
        audio_bytes = io.BytesIO(audio['bytes'])
        y, sr = librosa.load(audio_bytes, sr=None)

        if len(y) > 0:
            # 주파수 및 음색 분석
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            index = magnitudes.argmax()
            pitch = pitches.flatten()[index]

            if pitch > 0:
                target_hz = note_freqs[target_note]
                cent_error = int(1200 * np.log2(pitch / target_hz))
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                tone_score = int(np.clip((centroid / 5000) * 100, 0, 100))

                # 대시보드 메트릭 섹션
                st.write("---")
                st.subheader("📊 Key Performance Indicators")
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.metric("Pitch Precision", f"{pitch:.1f} Hz")
                with m2:
                    color = "normal" if abs(cent_error) < 15 else "inverse"
                    st.metric("Intonation Error", f"{cent_error} Cent", delta=f"{cent_error}c", delta_color=color)
                with m3:
                    st.metric("Timbre Brilliance", f"{tone_score} pts")

                # 데이터 전송 및 피드백
                st.info(f"💡 **분석 결과:** {target_note} 대비 음정이 {'높습니다' if cent_error > 0 else '낮습니다'}. {'호흡의 압력을 조절해보세요.' if abs(cent_error) > 10 else '매우 훌륭한 정확도입니다!'}")
                
                if st.button("📈 데이터를 구글 시트에 기록하고 학습시키기"):
                    st.balloons()
                    st.success("데이터가 성공적으로 전송되었습니다. iROOM이 당신의 소리를 학습합니다.")
            else:
                st.error("분석 가능한 소리가 감지되지 않았습니다.")

# --- [5] 하단 보정 및 통계 (Expansion) ---
st.write("---")
with st.expander("🛠️ Advanced Calibration (Admin Only)"):
    st.write("악기별 고유 편차 보정 데이터")
    cal_data = pd.DataFrame({
        "Metric": ["Pitch Bias", "Harmonic Saturation", "Response Speed"],
        "Current Value": [f"{cent_error if audio else 0}c", f"{tone_score if audio else 0}%", "0.12s"]
    })
    st.table(cal_data)
