import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EMG Anomaly Dashboard",
    layout="wide",
)

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def set_bg(image_file):
    try:
        with open(image_file, "rb") as img:
            b64_string = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64_string}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found.")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "file" not in st.session_state:
    st.session_state.file = None

# ---------------- PAGE BACKGROUNDS ----------------
page_backgrounds = {
    "welcome": "doctor-doing-an-emg-test-on-a-woman.jpg",
    "upload": "185673536-signal-wallpaper.jpg",
    "analysis": "bk.jpg"
}

if st.session_state.page in page_backgrounds:
    set_bg(page_backgrounds[st.session_state.page])

# ---------------- WELCOME PAGE ----------------
if st.session_state.page == "welcome":
    st.title("🧬 EMG Anomaly Detection System")
    st.markdown("""
    <div style="color:black; font-size:25px; line-height:1.6;">
        <strong>Clinical tool to analyze muscle EMG signals:</strong>
        <ul style="margin-top:5px;">
            <li>Detect abnormal muscle activity</li>
            <li>Compare patient EMG with healthy baseline</li>
            <li>Generate a PDF report for doctors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🩺 Start Analysis"):
        st.session_state.page = "upload"
        st.rerun()

# ---------------- UPLOAD PAGE ----------------
elif st.session_state.page == "upload":
    st.sidebar.header("👨‍⚕️ Doctor Panel")
    st.session_state.patient_id = st.sidebar.text_input("Patient ID")
    st.session_state.muscle_type = st.sidebar.selectbox(
        "Muscle Type", ["Biceps", "Triceps", "Forearm", "Thigh", "Calf"]
    )

    uploaded_file = st.file_uploader("Upload Patient EMG (.csv)", type=["csv"])
    if uploaded_file is not None:
        st.session_state.file = uploaded_file
        st.success("✅ Patient EMG uploaded")
        if st.button("🔍 Analyze EMG"):
            st.session_state.page = "analysis"
            st.rerun()

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "welcome"
        st.rerun()

# ---------------- ANALYSIS PAGE ----------------
elif st.session_state.page == "analysis":
    if st.session_state.file is None:
        st.warning("⚠️ No patient data loaded. Please upload EMG again.")
        if st.button("⬅️ Back to Doctor Panel"):
            st.session_state.page = "upload"
            st.rerun()
        st.stop()

    st.session_state.file.seek(0)
    try:
        patient_df = pd.read_csv(st.session_state.file)
    except Exception as e:
        st.error(f"⚠️ Uploaded CSV is empty or invalid. {e}")
        if st.button("⬅️ Back to Doctor Panel"):
            st.session_state.page = "upload"
            st.session_state.file = None
            st.rerun()
        st.stop()

    patient_signal = patient_df.iloc[:,0].values

    # Moving average baseline
    window = 5
    baseline = np.convolve(patient_signal, np.ones(window)/window, mode='same')

    # Z-score anomaly detection
    z_scores = np.abs((patient_signal - baseline) / (np.std(patient_signal[:window])+1e-6))
    anomalies = z_scores > 2
    anomaly_percent = round(np.sum(anomalies)/len(patient_signal)*100, 2)

    # Severity classification
    severity = np.where(z_scores > 5, "Severe",
               np.where(z_scores > 4, "Moderate",
               np.where(z_scores > 3, "Mild", "Normal")))

    anomaly_info = [{"Timepoint": i, "EMG": patient_signal[i], "Severity": severity[i]}
                    for i in np.where(anomalies)[0]]

    # ---------------- PLOT ----------------
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(patient_signal, label='Patient EMG')
    ax.plot(baseline, label='Baseline', alpha=0.5)
    ax.scatter(np.where(anomalies)[0], patient_signal[anomalies], color='red', s=50, label='Anomalies')
    ax.set_title("EMG Signal vs Baseline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()

    tab1, tab2 = st.tabs(["📈 EMG Plot", "📊 Summary & Interpretation"])
    with tab1:
        st.pyplot(fig)
        st.metric("Anomaly %", f"{anomaly_percent}%")

    with tab2:
        st.subheader("🧾 Clinical Summary")
        st.write(f"**Patient ID:** {st.session_state.patient_id}")
        st.write(f"**Muscle:** {st.session_state.muscle_type}")
        st.write(f"**Anomalies detected:** {np.sum(anomalies)}")
        if anomaly_info:
            st.subheader("🔴 Anomaly Details")
            st.table(pd.DataFrame(anomaly_info))

       # ---------------- MUSCLE-SPECIFIC INTERPRETATION ----------------
    muscle_alerts = {
        "Biceps": "Possible fatigue or repetitive strain injury.",
        "Triceps": "Could indicate overuse or nerve irritation.",
        "Forearm": "May suggest tremors, overexertion, or nerve issues.",
        "Thigh": "Could point to muscle imbalance or strain.",
        "Calf": "Might indicate cramps or nerve conduction issues."
    }

    suggested_actions = {
        "Biceps": ["Recommend physiotherapy for biceps strengthening.", 
                "Consider EMG follow-up if symptoms persist."],
        "Triceps": ["Targeted triceps exercises suggested.",
                    "Monitor for nerve irritation."],
        "Forearm": ["Recommend physiotherapy or targeted forearm exercises.", 
                    "Consider further nerve conduction study or EMG follow-up."],
        "Thigh": ["Strengthening exercises for quadriceps/hamstrings.", 
                "Check for muscle imbalance."],
        "Calf": ["Stretching and calf strengthening exercises.", 
                "Consider EMG if cramps/fatigue persist."]
    }

    # Show muscle-specific alert
    alert_msg = muscle_alerts.get(st.session_state.muscle_type, "")
    if alert_msg:
        st.info(f"⚠️ Interpretation: {alert_msg}")

    # Show suggested next steps
    actions = suggested_actions.get(st.session_state.muscle_type, [])
    if actions:
        st.subheader("💡 Suggested Next Steps")
        for act in actions:
            st.write(f"- {act}")

    # ---------------- DOWNLOAD PDF ----------------
    def create_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "EMG Analysis Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Patient ID: {st.session_state.patient_id}", ln=True)
        pdf.cell(0, 10, f"Muscle: {st.session_state.muscle_type}", ln=True)
        pdf.cell(0, 10, f"Anomalies detected: {np.sum(anomalies)}", ln=True)
        pdf.cell(0, 10, f"Anomaly %: {anomaly_percent}%", ln=True)
        pdf.ln(10)

        # Table of anomalies
        if anomaly_info:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(40, 10, "Time", border=1)
            pdf.cell(40, 10, "EMG", border=1)
            pdf.cell(40, 10, "Severity", border=1)
            pdf.ln()
            pdf.set_font("Arial", '', 12)
            for item in anomaly_info:
                pdf.cell(40, 10, str(item["Timepoint"]), border=1)
                pdf.cell(40, 10, f"{item['EMG']:.2f}", border=1)
                pdf.cell(40, 10, item["Severity"], border=1)
                pdf.ln()

        # Save figure to temp file and embed
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig.savefig(tmpfile.name, format='PNG')
            pdf.image(tmpfile.name, x=10, w=pdf.w - 20)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return pdf_bytes

    pdf_bytes = create_pdf()
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"EMG_Report_{st.session_state.patient_id}.pdf",
        mime="application/pdf"
    )

    if st.button("⬅️ Back to Doctor Panel"):
        st.session_state.page = "upload"
        st.session_state.file = None
        st.rerun() 
