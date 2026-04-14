import streamlit as st
from streamlit_mic_recorder import mic_recorder
import numpy as np
import librosa
import io

# --- [UI 설정] ---
st.set_page_config(page_title="iROOM - Music Analytics", layout="wide")

st.title("🎺 iROOM : 데이터 기반 음악 교육의 새 시대")
st.subheader("Step 1: 음정 및 음색 실시간 분석 엔진")
st.write("---")

# --- [1단계] 음정 제시 ---
col1, col2 = st.columns([1, 2])
with col1:
    st.info("### 🎯 오늘의 목표 음정")
    target_note = st.selectbox("연주할 음을 선택하세요", ["Bb4", "A4", "G4", "F4", "Eb4"])
    # 음 이름별 주파수 기준 (보정 학습용 데이터)
    note_freqs = {"Bb4": 466.16, "A4": 440.0, "G4": 392.0, "F4": 349.23, "Eb4": 311.13}

# --- [2~3단계] 사용자 연주 및 데이터 분석 ---
with col2:
    st.write("### 🎙️ 연주 녹음 및 실시간 분석")
    audio = mic_recorder(start_prompt="🔴 녹음 시작 (트럼펫 연주)", stop_prompt="⏹️ 녹음 완료", key='recorder')

    if audio:
        # 1. 오디오 로드
        audio_bytes = io.BytesIO(audio['bytes'])
        y, sr = librosa.load(audio_bytes, sr=None)

        if len(y) > 0:
            # 2. 음정(Pitch) 분석
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            # 가장 강한 주파수 추출
            index = magnitudes.argmax()
            pitch = pitches.flatten()[index]

            if pitch > 0:
                # 3. 음색(Tone) 분석 - 배음 구조 (Spectral Centroid)
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                
                # 4. 결과 계산
                target_hz = note_freqs[target_note]
                # Cent 오차 계산 공식: 1200 * log2(f2/f1)
                cent_error = int(1200 * np.log2(pitch / target_hz))
                
                # 음색 점수화 (단순화: Centroid가 높을수록 화려한 소리로 가정)
                tone_score = int(np.clip((centroid / 5000) * 100, 0, 100))

                st.success("✅ 분석이 완료되었습니다!")
                
                # 대시보드 출력
                m1, m2, m3 = st.columns(3)
                m1.metric("실측 주파수", f"{pitch:.2f} Hz")
                
                # 오차 범위에 따른 색상 표시
                delta_color = "normal" if abs(cent_error) < 10 else "inverse"
                m2.metric("음정 오차", f"{cent_error} Cent", delta=f"{cent_error}c", delta_color=delta_color)
                m3.metric("음색 화려함", f"{tone_score} 점")

                # --- 데이터 전송 섹션 ---
                st.write("---")
                user_name = st.text_input("테스터 이름", "홍길동")
                if st.button("📊 이 데이터를 구글 시트로 전송"):
                    st.balloons()
                    st.write(f"[{user_name}]님의 {target_note} 데이터를 서버로 보냈습니다! (구글 시트 연동 대기 중)")
            else:
                st.warning("소리가 너무 작거나 감지되지 않았습니다. 다시 시도해주세요.")
