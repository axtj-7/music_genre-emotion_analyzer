import streamlit as st
import os
import sys
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from predict_genre import predict as predict_genre

# ------------------ CONFIG ------------------
st.set_page_config(layout="centered")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from g.audio_feature import extract_audio_features
from e.predict_emo import predict_emotion_timeline

# load models
emotion_model = joblib.load("models/emotion_model.pkl")

import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(10,10,20,0.75), 
            rgba(10,10,20,0.95)
        ),
        url("data:image/png;base64,{encoded}");
        
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Optional: remove white containers */
    .block-container {{
        background: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("bg.jpeg")

st.markdown("""
<style>
.css-1d391kg {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container { padding-top: 2rem; }

.card {
    background: rgba(255,255,255,0.05);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
}

.big-text {
    font-size: 2rem;
    font-weight: bold;
    background: linear-gradient(90deg, #a855f7, #6366f1);
    -webkit-background-clip: text;
    color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="
        white-space: nowrap;
        font-size: clamp(24px, 3vw, 36px);
        text-align: center;
    ">
    🎵 Music Genre & Emotion Analyzer
    </h1>
    """,
    unsafe_allow_html=True
)

# ------------------ FUNCTIONS ------------------

def get_distribution(emotions):
    count = Counter(emotions)
    total = sum(count.values())
    return {k: (v / total) * 100 for k, v in count.items()}


def plot_centered(fig_func):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fig_func()


# ------------------ MAIN ------------------

file = st.file_uploader("Upload audio", type=["wav", "mp3"])

if file:

    with open("temp.wav", "wb") as f:
        f.write(file.read())

    st.audio("temp.wav")

    y, sr = librosa.load("temp.wav", sr=22050)

    # ------------------ PREDICTIONS ------------------
    time_points, emotions = predict_emotion_timeline("temp.wav")
    percent = get_distribution(emotions)

    all_emotions = ['sad', 'calm', 'happy', 'angry']
    values = [percent.get(e, 0) for e in all_emotions]

    overall_emotion = max(percent, key=percent.get)
    emotion_conf = percent[overall_emotion]

    genre, genre_conf = predict_genre("temp.wav")

    # ------------------ CARDS ------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div>GENRE</div>
            <div class="big-text">{genre}</div>
            <div>{genre_conf*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div>EMOTION</div>
            <div class="big-text">{overall_emotion}</div>
            <div>{emotion_conf:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # timeline
    st.subheader("📈 Emotion Timeline")

    emotion_map = {'sad': 0, 'calm': 1, 'happy': 2, 'angry': 3}
    numeric = [emotion_map[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(time_points, numeric, color="#a855f7", marker='o')
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(all_emotions)

    st.pyplot(fig)
    
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🎭 Emotion Profile")

        angles = np.linspace(0, 2*np.pi, len(all_emotions), endpoint=False)
        values_cycle = values + [values[0]]
        angles_cycle = np.append(angles, angles[0])

        fig = plt.figure(figsize=(3,3))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles_cycle, values_cycle, color="#a855f7", linewidth=2)
        ax.fill(angles_cycle, values_cycle, alpha=0.3, color="#a855f7")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Emotion")
        ax.set_xticks(angles)
        ax.set_xticklabels(all_emotions)

        st.pyplot(fig)

    with col4:
        st.subheader("🎼 Audio Characteristics")
    
        feature_dict, _ = extract_audio_features(y=y, sr=sr)
    
        # Normalize values to percentage scale (0–100)
        energy = feature_dict["energy"] * 100
        brightness = feature_dict["spectral_centroid"] / 50   # scaled
        bandwidth = feature_dict["spectral_bandwidth"] / 50   # scaled
        zcr = feature_dict["zcr"] * 100
        contrast = np.mean(feature_dict["spectral_contrast"])  # already reasonable
    
        labels = ["Energy", "Brightness", "Bandwidth", "Noisiness", "Contrast"]
        vals = [energy, brightness, bandwidth, zcr, contrast]
    
        # Clip values to max 100 for clean UI
        vals = [min(100, v) for v in vals]
    
        fig, ax = plt.subplots(figsize=(5,3))
    
        colors = ["#ff6b6b", "#4ecdc4", "#a855f7", "#f59e0b", "#ec4899"]
    
        ax.barh(labels, vals, color=colors)
    
        # % labels
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.0f}%", va='center', color="white", fontsize=9)
    
        ax.set_xlim(0, 100)
    
        # Styling (dark UI)
        ax.set_facecolor("#0b0f19")
        fig.patch.set_facecolor("#0b0f19")
    
        ax.tick_params(colors='white')
    
        for spine in ax.spines.values():
            spine.set_color("#444")
    
        st.pyplot(fig)

    # ------------------ AUDIO VISUALS ------------------
    st.subheader("🎛 Audio Visualizations")

    tabs = st.tabs(["Waveform", "Spectrogram", "Chromagram", "MFCC"])

    with tabs[0]:
        def plot_wave():
            fig, ax = plt.subplots(figsize=(8,3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
        
            ax.set_title("Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
        
            st.pyplot(fig)
        
        plot_wave()
        st.caption("The waveform shows the amplitude of the audio signal over time. Peaks indicate louder sections.")

    with tabs[1]:
        def spec():
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S)
            fig, ax = plt.subplots(figsize=(8,3))
            img = librosa.display.specshow(
                S_db,
                sr=sr,
                x_axis='time',
                y_axis='mel',
                ax=ax
           )

            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
            ax.set_title("Mel Spectrogram")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Frequency (Hz)")
        
            st.pyplot(fig)
        spec()
        st.caption("The Mel spectrogram displays frequency content over time. Brighter areas = more energy at that frequency.")

    with tabs[2]:
        def chroma():
            c = librosa.feature.chroma_stft(y=y, sr=sr)
        
            fig, ax = plt.subplots(figsize=(8,3))  # 🔥 FIX SIZE
        
            img = librosa.display.specshow(
                c,
                sr=sr,
                x_axis='time',
                y_axis='chroma',
                ax=ax
            )
        
            fig.colorbar(img, ax=ax)
        
            ax.set_title("Chromagram")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Pitch Class")
        
            ax.tick_params(labelsize=8)  # 🔥 readable labels
        
            plt.tight_layout()  # 🔥 prevents cutting
        
            st.pyplot(fig)
        chroma()
        st.caption("The chromagram shows the intensity of each of the 12 pitch classes (C, C#, D, ..., B) over time.")

    with tabs[3]:
        def mfcc():
            m = librosa.feature.mfcc(y=y, sr=sr)
        
            fig, ax = plt.subplots(figsize=(8,3))  # 🔥 consistent size
        
            img = librosa.display.specshow(
                m,
                sr=sr,                    # 🔥 IMPORTANT FIX
                x_axis='time',
                ax=ax
            )
        
            fig.colorbar(img, ax=ax)
        
            ax.set_title("MFCCs (Timbral Features)")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("MFCC Coefficients")
        
            ax.tick_params(labelsize=8)
        
            plt.tight_layout()
        
            st.pyplot(fig)
        mfcc()
        st.caption("MFCCs capture the timbral texture of sound — what makes a guitar sound different from a piano.")

    # ------------------ HOW AI SEES ------------------
    st.subheader("🧠 How AI Analyzed Your Audio")

    col5, col6 = st.columns([1,2])
    
    def section(title):
        st.markdown(
            f"<h3 style='font-size:22px; font-weight:600; margin-bottom:10px;'>{title}</h3>",
            unsafe_allow_html=True
        )
    with col5:
        section("🎧 What the AI Sees")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S)
    
        fig, ax = plt.subplots(figsize=(4,3))
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
    
        ax.set_title("Spectrogram")
        plt.tight_layout()
    
        st.pyplot(fig)
    
        st.caption("This visual representation of sound is used by the CNN to detect genre.")

        section("🔍 What the AI Found")
        feature_dict, _ = extract_audio_features(y=y, sr=sr)
    
        tempo = feature_dict["tempo"]
        energy = feature_dict["energy"]
        insights = []
    
        if tempo > 120:
            insights.append("In this track, the fast tempo suggests a high-energy composition.")
        else:
            insights.append("In this track, the slower tempo suggests a calm and relaxed feel.")
    
        if energy > 0.1:
            insights.append("In this track, the high energy suggests strong and intense sections.")
        else:
            insights.append("In this track, the low energy suggests soft and smooth audio.")
    
        for i in insights:
            st.markdown(f"- {i}")

    with col6:
        # ----------- SECTION 1 -----------
        section("⚙️ What the AI Did")
        st.markdown("""
        - The audio was split into smaller segments to capture changes over time  
        - Each segment was converted into a spectrogram (visual sound image)  
        - A CNN analyzed these images to identify the genre  
        - Audio features like tempo and energy were extracted  
        - A Random Forest model used these features to detect emotion  
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # ----------- SECTION 2 -----------
        section("🎯 Final Decision")
        st.markdown(f"""
        - **Genre:** `{genre}` ({genre_conf*100:.1f}% confidence)  
        - **Emotion:** `{overall_emotion}` ({emotion_conf:.1f}% confidence)  
        """)
    
        st.markdown("""
        The genre model uses deep learning on spectrograms, while the emotion model 
        uses feature-based learning, enabling accurate and reliable predictions.
        """)