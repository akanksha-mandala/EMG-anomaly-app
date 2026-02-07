import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ---- PRELOADED NORMAL EMG BASELINES ----
BASELINE_EMG = {
    muscle: pd.to_numeric(pd.read_csv(f"baseline_emg/baseline_{muscle.lower()}.csv")['emg'], errors='coerce').dropna().values
    for muscle in ["Biceps", "Triceps", "Forearm", "Thigh", "Calf"]
}

# ---- Session Memory ----
if "baseline_emg" not in st.session_state:
    st.session_state.baseline_emg = None

def sliding_window(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size, step_size):
        windows.append(signal[start:start + window_size])
    return windows


def extract_features(window):
    mean = np.mean(window)
    rms = np.sqrt(np.mean(window ** 2))
    var = np.var(window)
    peak = np.max(np.abs(window))
    return [mean, rms, var, peak]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EMG Anomaly Detection System",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------------- WELCOME PAGE ----------------
if st.session_state.page == "welcome":
    st.title("🧬 EMG Anomaly Detection System")
    st.markdown("""
    **Clinical tool for time-series analysis of muscle activity.**

    This system:
    - Learns *normal EMG behavior*
    - Detects deviations over time
    - Assists doctors in identifying abnormalities
    """)

    st.markdown("---")

    if st.button("🩺 Start New Analysis"):
        st.session_state.page = "upload"
        st.rerun()

# ---------------- UPLOAD PAGE ----------------
elif st.session_state.page == "upload":
    st.sidebar.header("👨‍⚕️ Doctor Panel")

    # Patient info
    patient_id = st.sidebar.text_input("Patient ID")
    muscle_type = st.sidebar.selectbox(
        "Muscle Type",
        ["Biceps", "Triceps", "Forearm", "Thigh", "Calf"]
    )
    # Auto-load baseline based on muscle
    st.session_state.baseline_emg = BASELINE_EMG[muscle_type]

    st.title("🧬 EMG Anomaly Detection – Clinical View")

    # ---------------- PATIENT ----------------
    st.subheader("📤 Step 2: Upload Patient EMG")

    uploaded_file = st.file_uploader(
        "Upload PATIENT EMG (.csv)",
        type=["csv"],
        key="patient"
    )

    if uploaded_file is not None:
        st.success("✅ Patient EMG uploaded")

        if st.button("🔍 Compare with Baseline"):
            st.session_state.file = uploaded_file
            st.session_state.patient_id = patient_id
            st.session_state.muscle_type = muscle_type
            st.session_state.page = "analysis"
            st.rerun()

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "welcome"
        st.rerun()

# ---------------- ANALYSIS PAGE (TEMP VIEW) ----------------
elif st.session_state.page == "analysis":
    st.title("📊 EMG Comparison & Anomaly Detection")

    # 🔒 HARD GUARD (prevents empty CSV crash)
    if (
        "file" not in st.session_state
        or st.session_state.file is None
        or st.session_state.baseline_emg is None
    ):
        st.warning("⚠️ No patient EMG loaded. Please upload again.")
        if st.button("⬅️ Back to Doctor Panel"):
            st.session_state.page = "upload"
            st.rerun()
        st.stop()

    # Load data
    baseline = st.session_state.baseline_emg
    try:
        patient_df = pd.read_csv(st.session_state.file)
    except pd.errors.EmptyDataError:
        st.error("⚠️ Uploaded CSV is empty or invalid. Please re-upload the patient EMG file.")
        if st.button("⬅️ Back to Doctor Panel"):
            st.session_state.page = "upload"
            st.session_state.file = None
            st.rerun()
        st.stop()
    patient_signal = patient_df.iloc[:, 0].values

    # Align length
    min_len = min(len(baseline), len(patient_signal))
    baseline = baseline[:min_len]
    patient_signal = patient_signal[:min_len]

    # Statistics
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    # Z-score anomaly detection
    z_scores = np.abs((patient_signal - baseline_mean) / baseline_std)
    anomaly_threshold = 3
    anomalies = z_scores > anomaly_threshold

    # Plot
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=patient_signal,
        mode="lines",
        name="Patient EMG"
    ))

    fig.add_trace(go.Scatter(
        y=baseline,
        mode="lines",
        name="Baseline EMG",
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=np.where(anomalies)[0],
        y=patient_signal[anomalies],
        mode="markers",
        name="Anomalies",
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="EMG Signal vs Baseline with Detected Anomalies",
        xaxis_title="Time",
        yaxis_title="Amplitude"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.subheader("🧾 Clinical Summary")
    st.write(f"**Patient ID:** {st.session_state.patient_id}")
    st.write(f"**Muscle:** {st.session_state.muscle_type}")
    st.write(f"**Anomalies detected:** {np.sum(anomalies)}")
    # 🧠 Clinical Interpretation
    st.subheader("🩺 Interpretation")

    if np.sum(anomalies) == 0:
        st.success(
        "The EMG signal closely follows the normal muscle activation pattern. "
        "No abnormal muscle behavior was detected during the observed period."
    )
    else:
        st.error(
        "The EMG signal shows significant deviations from normal muscle behavior. "
        "Red markers indicate time points where muscle activity exceeds expected limits, "
        "which may suggest muscle fatigue, nerve dysfunction, or abnormal contractions. "
        "Further clinical evaluation is recommended."
    )

    if st.button("⬅️ Back to Doctor Panel"):
        st.session_state.page = "upload"
        st.session_state.file = None
        st.rerun()

    