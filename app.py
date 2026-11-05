
import streamlit as st
import torch
import tempfile
import os

from model import load_model
from video_processor import extract_frames


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Fall Detection",
    page_icon="🚨",
    layout="wide"
)


# ==========================================
# PROFESSIONAL CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0b1120, #111827);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 35px;
}

.section-heading {
    font-size: 25px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 15px;
}

.dashboard-card {
    background: #172033;
    border: 1px solid #263449;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    transition: 0.3s;
}

.dashboard-card:hover {
    border-color: #38bdf8;
    transform: translateY(-3px);
}

.card-label {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 8px;
}

.card-value {
    font-size: 20px;
    font-weight: 700;
    color: #38bdf8;
}

.upload-container {
    background: #151e2e;
    border: 1px solid #263449;
    border-radius: 15px;
    padding: 25px;
}

.result-card {
    padding: 35px 25px;
    border-radius: 18px;
    text-align: center;
    margin: 20px 0;
    animation: fadeIn 0.6s ease-in-out;
}

.fall-card {
    background: linear-gradient(135deg, #3b1720, #27151d);
    border: 2px solid #ef4444;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.12);
}

.normal-card {
    background: linear-gradient(135deg, #12352d, #142820);
    border: 2px solid #10b981;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.12);
}

.result-icon {
    font-size: 60px;
    margin-bottom: 12px;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 10px;
}

.result-confidence {
    font-size: 18px;
    color: #cbd5e1;
}

.alert-box {
    background: #451a1a;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 18px;
    margin: 15px 0;
    color: #fecaca;
    font-size: 15px;
}

.success-box {
    background: #123329;
    border-left: 5px solid #10b981;
    border-radius: 10px;
    padding: 18px;
    margin: 15px 0;
    color: #a7f3d0;
    font-size: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #0891b2, #06b6d4);
    color: white;
    border: none;
    border-radius: 10px;
    height: 52px;
    font-size: 17px;
    font-weight: 700;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0e7490, #0891b2);
    border: 1px solid #38bdf8;
    transform: scale(1.01);
}

div[data-testid="stFileUploader"] {
    background: #151e2e;
    border-radius: 12px;
    padding: 10px;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #06b6d4, #38bdf8);
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding-top: 30px;
}

.disclaimer-box {
    background: #292719;
    border: 1px solid #51451c;
    border-radius: 10px;
    padding: 15px;
    color: #facc15;
    font-size: 13px;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Bright white Streamlit metric numbers */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Light-colored metric labels */
[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)







# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_PATH = "models/best_32frame.pth"


@st.cache_resource
def load_trained_model():

    model = load_model(MODEL_PATH)

    return model


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🚨 AI-Based Fall Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Human Activity Monitoring Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# DASHBOARD INFORMATION
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="dashboard-card">
        <div class="card-label">🧠 Deep Learning Model</div>
        <div class="card-value">ResNet18 + LSTM</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="dashboard-card">
        <div class="card-label">🎞️ Video Processing</div>
        <div class="card-value">32 Frames</div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="dashboard-card">
        <div class="card-label">⚡ Classification</div>
        <div class="card-value">Normal / Fall</div>
    </div>
    """, unsafe_allow_html=True)


st.divider()


# ==========================================
# VIDEO UPLOAD
# ==========================================

st.markdown(
    '<div class="section-heading">📹 Upload Your Video</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a video containing human activity "
    "to analyze possible fall-related activity."
)

uploaded_video = st.file_uploader(
    "Select your video",
    type=["mp4", "avi", "mov"],
    help="Supported formats: MP4, AVI, MOV"
)


# ==========================================
# VIDEO ANALYSIS
# ==========================================

if uploaded_video is not None:

    st.video(uploaded_video)

    st.success("✅ Video uploaded successfully!")

    st.write("Click below to start AI-powered analysis.")

    if st.button(
        "🔍  ANALYZE VIDEO",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 Analyzing video with ResNet18 + LSTM..."
        ):

            video_path = None

            try:

                # Save video temporarily
                suffix = os.path.splitext(
                    uploaded_video.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_video.getbuffer()
                    )

                    video_path = temp_file.name


                # Load model
                model = load_trained_model()


                # Extract frames
                video_tensor = extract_frames(
                    video_path
                )


                # Prediction
                with torch.no_grad():

                    output = model(video_tensor)

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )

                    predicted_class = torch.argmax(
                        probabilities,
                        dim=1
                    ).item()


                # Class names
                classes = ["Normal", "Fall"]

                prediction = classes[predicted_class]

                confidence = (
                    probabilities[0][predicted_class].item()
                    * 100
                )

                normal_probability = (
                    probabilities[0][0].item()
                    * 100
                )

                fall_probability = (
                    probabilities[0][1].item()
                    * 100
                )


            except Exception as error:

                st.error(
                    f"❌ Error processing video: {error}"
                )

                st.stop()


            finally:

                if video_path and os.path.exists(video_path):

                    os.remove(video_path)


        # ======================================
        # RESULTS
        # ======================================

        st.divider()

        st.markdown(
            '<div class="section-heading">'
            '📊 Analysis Results'
            '</div>',
            unsafe_allow_html=True
        )


        # ======================================
        # FALL DETECTED
        # ======================================

        # ======================================
        # RESULTS
        # ======================================

        st.divider()

        st.markdown("## 📊 Analysis Results")

        # FALL DETECTED
        if prediction == "Fall":

            st.error("🚨 FALL DETECTED")

            st.metric(
                label="🚨 Fall Detection Confidence",
                value=f"{confidence:.2f}%"
            )

            st.warning(
                "⚠️ SAFETY ALERT: A possible fall has been detected. "
                "Please immediately check on the person and "
                "contact emergency assistance if needed."
            )

        # NORMAL ACTIVITY
        else:

            st.success("✅ NORMAL ACTIVITY DETECTED")

            st.metric(
                label="✅ Normal Activity Confidence",
                value=f"{confidence:.2f}%"
            )

            st.success(
                "No fall was detected in the analyzed video."
            )

        # ======================================
        # CONFIDENCE VISUALIZATION
        # ======================================

        st.markdown("## 📈 Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="🟢 Normal Activity",
                value=f"{normal_probability:.2f}%"
            )

            st.progress(
                int(normal_probability)
            )

        with col2:

            st.metric(
                label="🔴 Fall Detection",
                value=f"{fall_probability:.2f}%"
            )

            st.progress(
                int(fall_probability)
            )

        # ======================================
        # DISCLAIMER
        # ======================================

        st.warning(
            "DISCLAIMER: This is an educational and research "
            "prototype, not a certified medical or emergency "
            "monitoring system. Do not rely on it as the sole "
            "basis for real-world emergency decisions."
        )



# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        AI-Based Fall Detection System<br>
        Powered by PyTorch | ResNet18 | LSTM | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
