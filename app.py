import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from agent import ask_agent

# Page config
st.set_page_config(
    page_title="NewYork-Presbyterian | Leukemia AI Screening",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f4f6f9; }

    /* Sidebar background only */
    [data-testid="stSidebar"] { background-color: #0a2463; }
    [data-testid="stSidebar"] * { color: white !important; }

    /* Main area text — force dark */
    .main * { color: #1a1a2e !important; }
    .main h1, .main h2, .main h3 { color: #0a2463 !important; }
    
    /* Override metric text specifically */
    [data-testid="stMetricValue"] { color: #0a2463 !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #555 !important; }

    /* Header — white text on dark background */
    .nyp-header { background-color: #0a2463; padding: 20px 30px; border-radius: 8px; margin-bottom: 24px; }
    .nyp-header h1 { color: white !important; font-size: 22px; font-weight: 700; margin: 0; }
    .nyp-header p { color: #a8c4e0 !important; font-size: 13px; margin: 4px 0 0 0; }

    /* Chat bubbles */
    .chat-user {
        background-color: #0a2463;
        color: white !important;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 14px;
    }
    .chat-assistant {
        background-color: white;
        color: #1a1a2e !important;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 75%;
        border: 1px solid #e0e6ed;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Confidence card */
    .confidence-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #0a2463;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .confidence-card h3 { color: #0a2463 !important; margin: 0 0 8px 0; font-size: 16px; }
    .confidence-value { font-size: 36px; font-weight: 700; }
    .detected { color: #c0392b !important; }
    .not-detected { color: #27ae60 !important; }

    /* Sidebar section labels */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        color: #a8c4e0 !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Input box */
    .stTextInput input {
        border-radius: 24px;
        border: 2px solid #0a2463;
        padding: 10px 18px;
        font-size: 14px;
        color: #1a1a2e !important;
        background: white !important;
    }

    hr { border: none; border-top: 1px solid #e0e6ed; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="nyp-header">
    <h1>🏥 NewYork-Presbyterian Hospital</h1>
    <p>Leukemia AI Screening System &nbsp;|&nbsp; Powered by Deep Learning &nbsp;|&nbsp; For Physician Use Only</p>
</div>
""", unsafe_allow_html=True)

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quick_query" not in st.session_state:
    st.session_state.quick_query = ""
if "uploaded_patients" not in st.session_state:
    st.session_state.uploaded_patients = {}
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient Records")
    st.markdown("---")

    st.markdown('<div class="section-label">Upload Patient Data</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload CSV files (one per patient)",
        type=["csv"],
        accept_multiple_files=True,
        help="CSV must contain FilePath and Patient_New_ID columns"
    )

    if uploaded_files:
        for f in uploaded_files:
            df_upload = pd.read_csv(f)

            if "FilePath" not in df_upload.columns or "Patient_New_ID" not in df_upload.columns:
                st.error(f"❌ {f.name} missing required columns (FilePath, Patient_New_ID)")
                continue

            # Only keep FilePath and Patient_New_ID
            df_clean = df_upload[["FilePath", "Patient_New_ID"]].copy()
            pid = int(df_clean["Patient_New_ID"].iloc[0])
            st.session_state.uploaded_patients[pid] = df_clean

        st.success(f"{len(st.session_state.uploaded_patients)} patient(s) loaded")

    st.markdown("---")

    if st.session_state.uploaded_patients:
        st.markdown('<div class="section-label">Loaded Patients</div>', unsafe_allow_html=True)
        patient_ids = list(st.session_state.uploaded_patients.keys())

        selected_patient = st.selectbox(
            "Active Patient",
            options=patient_ids,
            format_func=lambda x: f"Patient {x}"
        )

        st.markdown(f"""
        <div style='background:#1a3a7a; border-radius:8px; padding:12px; margin:8px 0;'>
            <div style='color:#a8c4e0; font-size:11px; font-weight:700; letter-spacing:1px;'>ACTIVE PATIENT</div>
            <div style='color:white; font-size:20px; font-weight:700; margin:4px 0;'>Patient {selected_patient}</div>
            <div style='color:#a8c4e0; font-size:12px;'>{len(st.session_state.uploaded_patients[selected_patient])} cell images loaded</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)

        if st.button("🔬 Full Prediction", use_container_width=True):
            st.session_state.quick_query = f"What is the prediction for patient {selected_patient}?"

        if st.button("🧬 Top Cancer Cells", use_container_width=True):
            st.session_state.quick_query = f"Show me the top 5 cancer cells for patient {selected_patient}"

        if st.button("✅ Top Healthy Cells", use_container_width=True):
            st.session_state.quick_query = f"Show me the top 5 healthy cells for patient {selected_patient}"

        if st.button("🗺️ Grad-CAM Heatmaps", use_container_width=True):
            st.session_state.quick_query = f"Show me the gradcam heatmaps for patient {selected_patient}"

        if len(patient_ids) > 1:
            st.markdown("---")
            st.markdown('<div class="section-label">Compare Patients</div>', unsafe_allow_html=True)
            compare_options = [p for p in patient_ids if p != selected_patient]
            compare_patient = st.selectbox(
                "Compare With",
                options=compare_options,
                format_func=lambda x: f"Patient {x}"
            )
            if st.button("⚖️ Compare", use_container_width=True):
                st.session_state.quick_query = f"Compare patient {selected_patient} and patient {compare_patient}"

    else:
        st.markdown("""
        <div style='color:#a8c4e0; font-size:13px; text-align:center; padding:20px 0;'>
            📂 Upload a patient CSV<br>to get started
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#a8c4e0; line-height:1.6;'>
    ⚠️ <b>Clinical Disclaimer</b><br>
    This tool assists physicians and does not replace clinical judgment.
    All results must be verified by a licensed medical professional.
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Clinical Assistant")

    if not st.session_state.uploaded_patients:
        st.info("👈 Upload a patient CSV file in the sidebar to begin screening.")
    else:
        for entry in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">{entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-assistant">🤖 {entry["assistant"]}</div>', unsafe_allow_html=True)
            if entry.get("fig"):
                st.pyplot(entry["fig"])
            if entry.get("result") and "confidence" in entry["result"]:
                conf = entry["result"]["confidence"]
                pred = entry["result"]["prediction"]
                detected_class = "detected" if conf > 0.5 else "not-detected"
                st.markdown(f"""
                <div class="confidence-card">
                    <h3>Patient {entry["result"]["patient_id"]} — Screening Result</h3>
                    <div class="confidence-value {detected_class}">{conf:.1%}</div>
                    <div style="font-size:14px; margin-top:6px; color:#555;">{pred}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

        user_input = st.text_area(
            "Ask a question...",
            value=st.session_state.quick_query,
            placeholder="e.g. Show me the top 5 cancer cells for patient 86",
            label_visibility="collapsed",
            height=100,
            key="chat_input"
        )

        if st.button("Send →", type="primary") and user_input.strip():
            st.session_state.quick_query = ""
            with st.spinner("Analysing..."):
                fig, result, narration = ask_agent(
                    user_input,
                    uploaded_patients=st.session_state.uploaded_patients
                )
            st.session_state.chat_history.append({
                "user": user_input,
                "assistant": narration,
                "fig": fig,
                "result": result
            })
            st.rerun()

with col2:
    st.markdown("### 📊 Session Summary")

    st.metric("Queries This Session", len(st.session_state.chat_history))
    st.metric("Patients Loaded", len(st.session_state.uploaded_patients))

    if st.session_state.uploaded_patients:
        st.markdown("**Loaded Patients:**")
        for pid in st.session_state.uploaded_patients.keys():
            st.markdown(f"• Patient {pid}")

    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.uploaded_patients = {}
        st.rerun()