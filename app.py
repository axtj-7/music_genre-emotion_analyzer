import streamlit as st
import time
import os
import sys
import librosa
import librosa.display
import numpy as np
import joblib
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from collections import Counter
from predict_genre import predict as predict_genre
from cognitive.context_engine import analyze_music_context
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
    
    /* Waveform */
    .waveform {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 80px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    overflow: hidden;
    z-index: 999;
    pointer-events: none;
    opacity: 0.8;
   }

    .wave-track {
        display: flex;
        gap: 3px;
        width: max-content;
        animation: moveWave 30s linear infinite;
    }
    
    .bar {
        width: 4px;
        height: 80px;
        background: linear-gradient(to top, #ff00cc, #4facfe);
        animation: equalizer 1s infinite ease-in-out;
        transform-origin: bottom;   
    }
    
    .bar:nth-child(2n) { animation-duration: 0.9s; }
    .bar:nth-child(3n) { animation-duration: 1.1s; }
    .bar:nth-child(4n) { animation-duration: 0.8s; }
    .bar:nth-child(5n) { animation-duration: 1.2s; }
    
    @keyframes equalizer {
    0%, 100% { transform: scaleY(0.3); }
    50% { transform: scaleY(1.2); }
    }
    @keyframes moveWave {
    0% { transform: translateX(0); }
    100% { transform: translateX(50%); }
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
st.subheader("🧠 User Context")

col_a, col_b, col_c = st.columns(3)

with col_a:
    user_mood = st.selectbox(
        "Current Mood",
        [
            "stressed",
            "sad",
            "calm",
            "happy",
            "motivated",
            "tired"
        ]
    )

with col_b:
    activity = st.selectbox(
        "Current Activity",
        [
            "studying",
            "working",
            "gym",
            "sleeping",
            "meditation",
            "relaxation"
        ]
    )

with col_c:
    goal = st.selectbox(
        "Desired Outcome",
        [
            "improve focus",
            "relax",
            "boost energy",
            "maintain mood"
        ]
    )
file = st.file_uploader("Upload audio", type=["wav", "mp3"])

if file:

    loader = st.empty()
    status_text = st.empty()

    with open("temp.wav", "wb") as f:
        f.write(file.read()) #save file

    loader.markdown("""
    <div class="waveform">
        <div class="wave-track">
            """ + "".join([f'<div class="bar" style="--i:{i}"></div>' for i in range(1500)]) + """
            """ + "".join([f'<div class="bar" style="--i:{i}"></div>' for i in range(1500)]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    status_text.markdown("### 🎧 Extracting audio signal...")
    y, sr = librosa.load("temp.wav", sr=22050)
    time.sleep(0.5)
    status_text.markdown("### ⚡ Extracting features...")
    time_points, emotions = predict_emotion_timeline("temp.wav")
    percent = get_distribution(emotions)
    time.sleep(0.5)
    status_text.markdown("### 🧠 Analyzing emotion...")
    all_emotions = ['sad', 'calm', 'happy', 'angry']
    values = [percent.get(e, 0) for e in all_emotions]
    time.sleep(0.5)
    overall_emotion = max(percent, key=percent.get)
    emotion_conf = percent[overall_emotion]
    time.sleep(0.5)
    status_text.markdown("### 🎼 Classifying genre...")
    genre, genre_conf = predict_genre("temp.wav")
    time.sleep(0.5)
    feature_dict, _ = extract_audio_features(y=y, sr=sr)
    context_result = analyze_music_context(
        feature_dict=feature_dict,
        emotions=emotions,
        emotion=overall_emotion,
        genre=genre,
    
        user_mood=user_mood,
        activity=activity,
        goal=goal
    )
    status_text.markdown("### ✅ Analysis complete!")
    time.sleep(0.5)
    loader.empty()
    status_text.empty()
    
    audio_bytes = open("temp.wav", "rb").read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    st.markdown("---")
    components.html(f"""
        <audio id="audio" controls style="width:100%">
           <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        <br>
        <canvas id="eq" width="800" height="120" style="width:100%; margin-top:10px;"></canvas>
        
        <script>
        const audio = document.getElementById("audio");

        const canvas = document.getElementById('eq');
        const ctx = canvas.getContext('2d');
        
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioCtx.createAnalyser();
        analyser.smoothingTimeConstant = 0.9;
        
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        function draw() {{
            requestAnimationFrame(draw);
        
            analyser.getByteTimeDomainData(dataArray);
        
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        
            let gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, "#ff00cc");   // pink
            gradient.addColorStop(1, "#7b2ff7");   // purple
            ctx.shadowBlur = 10;
            ctx.shadowColor = "#00f2fe";
            const barWidth = canvas.width / bufferLength;
        
            for (let i = 0; i < bufferLength; i++) {{
                let value = dataArray[i] / 128.0;
                let amplitude = value - 1.0;
        
                let height = Math.abs(amplitude) * canvas.height * 0.9;
                let x = i * barWidth;
        
                ctx.fillStyle = gradient;
        
                ctx.fillRect(x, canvas.height/2 - height, barWidth - 2, height);
                ctx.fillRect(x, canvas.height/2, barWidth - 2, height);
            }}
        }}
        
        let started = false;

        function startVisualizer() {{
            if (started) return;
            started = true;
        
            const source = audioCtx.createMediaElementSource(audio);
        
            source.connect(analyser);
            analyser.connect(audioCtx.destination);
        
            audioCtx.resume();
            draw();
        }}
        
        // trigger on play
        audio.addEventListener("play", startVisualizer);
        
        // ALSO trigger on click (important for Streamlit)
        document.addEventListener("click", startVisualizer);
        </script>
        
        """, height=200)
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
    # ---------------- CONTEXTUAL AI ----------------

    st.subheader("🧠 Contextual Music Intelligence")
    
    score = context_result["compatibility_score"]
    
    if score >= 80:
        color = "#22c55e"
    
    elif score >= 60:
        color = "#eab308"
    
    else:
        color = "#ef4444"
    
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.05);
            padding: 1.5rem;
            border-radius: 18px;
            margin-top: 10px;
        ">
    
        <h2>
            Compatibility Score:
            <span style="color:{color}">
                {score}%
            </span>
        </h2>
    
        <h3>
            {context_result["alignment"]}
        </h3>
    
        <p>
            {context_result["summary"]}
        </p>
    
        <p>
            <strong>Recommendation:</strong>
            {context_result["recommendation"]}
        </p>
    
        </div>
        """,
        unsafe_allow_html=True
    )

    # timeline
    st.subheader("📈 Emotion Timeline")

    emotion_map = {'sad': 0, 'calm': 1, 'happy': 2, 'angry': 3}
    numeric = [emotion_map[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(time_points, numeric, color="#a855f7", marker='o')
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(all_emotions)

    st.pyplot(fig)
    st.markdown("### 🧠 Behavioral Music Analysis")

    for point in context_result["behavioral_analysis"]:
        st.markdown(f"- {point}")
    st.subheader("🧠 Music Cognition Metrics")

    scores = context_result["cognition_scores"]
    
    metric_cols = st.columns(len(scores))
    
    for col, (label, value) in zip(metric_cols, scores.items()):
        with col:
            radius = 38
            circumference = 2 * 3.1416 * radius
            offset = circumference - (value / 100) * circumference
    
            components.html(f"""
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    background: transparent;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    width: 100%;
                }}
                .card {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 20px;
                    width: 100%;
                    height: 190px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                }}
                svg {{
                    display: block;
                }}
                p {{
                    font-size: 14px;
                    font-weight: 500;
                    color: white;
                    text-align: center;
                    padding: 0 6px;
                }}
            </style>
    
            <div class="card">
                <svg width="100" height="100" viewBox="0 0 100 100">
    
                    <circle
                        cx="50" cy="50" r="{radius}"
                        stroke="rgba(255,255,255,0.1)"
                        stroke-width="12"
                        fill="none"
                    />
    
                    <circle
                        cx="50" cy="50" r="{radius}"
                        stroke="#22c55e"
                        stroke-width="12"
                        fill="none"
                        stroke-linecap="butt"
                        stroke-dasharray="{circumference}"
                        stroke-dashoffset="{offset}"
                        transform="rotate(-90 50 50)"
                    />
    
                    <text
                        x="50" y="50"
                        dominant-baseline="middle"
                        text-anchor="middle"
                        fill="white"
                        font-size="16"
                        font-weight="bold"
                    >
                        {value}%
                    </text>
    
                </svg>
    
                <p>{label}</p>
            </div>
            """, height=210)
        
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
        
            fig, ax = plt.subplots(figsize=(8,3))
        
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
        
            ax.tick_params(labelsize=8)  
        
            plt.tight_layout()  
        
            st.pyplot(fig)
        chroma()
        st.caption("The chromagram shows the intensity of each of the 12 pitch classes (C, C#, D, ..., B) over time.")

    with tabs[3]:
        def mfcc():
            m = librosa.feature.mfcc(y=y, sr=sr)
        
            fig, ax = plt.subplots(figsize=(8,3))  
        
            img = librosa.display.specshow(
                m,
                sr=sr,                    
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
        
        