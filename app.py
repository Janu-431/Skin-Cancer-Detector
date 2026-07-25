import io
import os
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2

# Import modules
from feature_extractor import extract_features
from image_validator import validate_skin_lesion_image
from train_model import train_rf, train_cnn, load_data

# Import Grad-CAM utilities
from utils.gradcam import generate_gradcam, build_gradcam_model
from utils.preprocessing import preprocess_for_display, pil_to_png_bytes

# Configuration
MODEL_DIR = "model_assets"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
CNN_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Cancer Detection Using Machine Learning",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"   # sidebar hidden by default
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
/* ── ROOT & BODY ─────────────────────────────────────────────── */
html, body { background: #0a0e17 !important; margin: 0; padding: 0; }

.stApp,
.stApp > div,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"],
section.main { background: #0a0e17 !important; color: #f1f5f9 !important; }

/* Kill Streamlit's white toolbar strip */
header[data-testid="stHeader"] {
    background: #0a0e17 !important;
    background-color: #0a0e17 !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
header[data-testid="stHeader"]::before,
header[data-testid="stHeader"]::after { display: none !important; }

div[data-testid="stToolbar"] { background: transparent !important; }

/* Collapse sidebar button invisible */
button[data-testid="collapsedControl"] { display: none !important; }

/* ── BLOCK CONTAINER – enough top space so header isn't clipped ── */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
}

/* ── 3-D PERSPECTIVE ROOT ─────────────────────────────────────── */
.perspective-root { perspective: 1200px; perspective-origin: 50% 40%; }

/* ── APP HEADER CARD ──────────────────────────────────────────── */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.15rem 1.8rem;
    background: linear-gradient(135deg, rgba(20,32,55,0.75) 0%, rgba(14,22,42,0.85) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 18px;
    margin-bottom: 1.4rem;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.45),
        0 0 0 1px rgba(34,211,238,0.08),
        inset 0 1px 0 rgba(255,255,255,0.06);
    transform: translateZ(0);
    transition: box-shadow 0.3s ease;
}
.app-header:hover {
    box-shadow:
        0 12px 48px rgba(0,0,0,0.5),
        0 0 30px rgba(34,211,238,0.08),
        inset 0 1px 0 rgba(255,255,255,0.08);
}
.header-left { display: flex; align-items: center; gap: 1.1rem; }
.header-icon {
    font-size: 2.4rem;
    color: #22d3ee;
    filter: drop-shadow(0 0 10px rgba(34,211,238,0.55));
    flex-shrink: 0;
}
.header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0; padding: 0; line-height: 1.2;
    text-shadow: 0 2px 12px rgba(34,211,238,0.18);
}
.header-subtitle {
    font-size: 0.88rem;
    color: #22d3ee;
    font-weight: 500;
    margin: 3px 0 0 0;
    letter-spacing: 0.3px;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.4);
    color: #22d3ee;
    padding: 0.45rem 1rem;
    border-radius: 30px;
    font-size: 0.83rem;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    white-space: nowrap;
    box-shadow: 0 0 14px rgba(34,211,238,0.12);
}

/* ── STREAMLIT TABS ───────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: rgba(16,24,42,0.7) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 5px 6px !important;
    gap: 6px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    margin-bottom: 1.5rem !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    border-radius: 10px !important;
    padding: 9px 22px !important;
    transition: all 0.22s ease !important;
    letter-spacing: 0.2px !important;
}
button[data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(255,255,255,0.04) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #22d3ee, #06b6d4) !important;
    color: #0a0e17 !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 18px rgba(34,211,238,0.45) !important;
}
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] { display: none !important; }

/* ── 3-D GLASS CARD BASE ──────────────────────────────────────── */
.card-3d {
    background: linear-gradient(145deg, rgba(22,33,58,0.7) 0%, rgba(14,22,40,0.8) 100%);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem 1.6rem;
    box-shadow:
        0 10px 40px rgba(0,0,0,0.4),
        0 1px 0 rgba(255,255,255,0.07) inset,
        0 -1px 0 rgba(0,0,0,0.2) inset;
    transform: perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0);
    transform-style: preserve-3d;
    transition: transform 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94),
                box-shadow 0.35s ease;
    position: relative;
    overflow: hidden;
}
.card-3d::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(34,211,238,0.03) 0%, transparent 60%);
    pointer-events: none;
}
.card-3d:hover {
    transform: perspective(900px) rotateX(-1.5deg) rotateY(1.5deg) translateZ(8px);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.55),
        0 0 40px rgba(34,211,238,0.07),
        0 1px 0 rgba(255,255,255,0.1) inset;
    border-color: rgba(34,211,238,0.2);
}

/* ── COLUMN PANELS ────────────────────────────────────────────── */
div[data-testid="column"] {
    background: linear-gradient(145deg, rgba(22,33,58,0.65) 0%, rgba(14,22,40,0.75) 100%) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
    padding: 1.5rem 1.6rem !important;
    box-shadow:
        0 10px 40px rgba(0,0,0,0.4),
        0 1px 0 rgba(255,255,255,0.06) inset !important;
    transform: perspective(900px) translateZ(0) !important;
    transform-style: preserve-3d !important;
    transition: transform 0.35s ease, box-shadow 0.35s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
div[data-testid="column"]:hover {
    transform: perspective(900px) rotateX(-1deg) rotateY(1deg) translateZ(6px) !important;
    box-shadow:
        0 20px 60px rgba(0,0,0,0.5),
        0 0 30px rgba(34,211,238,0.06),
        0 1px 0 rgba(255,255,255,0.09) inset !important;
    border-color: rgba(34,211,238,0.18) !important;
}

/* ── CARD HEADERS ─────────────────────────────────────────────── */
.card-hdr {
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.1rem;
}
.card-hdr h2 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 0 3px 0 !important;
    display: flex; align-items: center; gap: 0.45rem;
}
.card-hdr p {
    font-size: 0.82rem !important;
    color: #64748b !important;
    margin: 0 !important;
}

/* ── TOGGLE BUTTONS (File Upload / Live Camera) ───────────────── */
.toggle-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1.1rem;
}
.toggle-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 0.7rem 1rem;
    border-radius: 12px;
    font-family: 'Outfit', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    color: #64748b;
    transition: all 0.22s ease;
    user-select: none;
    transform: perspective(400px) translateZ(0);
}
.toggle-btn.active {
    background: rgba(34,211,238,0.08);
    border: 1.5px solid #22d3ee;
    color: #22d3ee;
    box-shadow: 0 0 18px rgba(34,211,238,0.15), inset 0 1px 0 rgba(34,211,238,0.1);
    transform: perspective(400px) translateZ(4px);
}
.toggle-btn:hover:not(.active) {
    border-color: rgba(34,211,238,0.25);
    color: #cbd5e1;
    background: rgba(255,255,255,0.04);
}

/* ── DRAG & DROP UPLOAD ZONE ──────────────────────────────────── */
.upload-zone {
    border: 2px dashed rgba(34,211,238,0.45);
    border-radius: 16px;
    padding: 2.75rem 1.5rem;
    text-align: center;
    background: rgba(10,20,38,0.55);
    cursor: pointer;
    transition: all 0.28s cubic-bezier(0.4,0,0.2,1);
    transform: perspective(600px) translateZ(0);
    box-shadow: inset 0 0 40px rgba(34,211,238,0.02);
    margin-bottom: 0.6rem;
    position: relative;
}
.upload-zone:hover {
    border-color: #22d3ee;
    background: rgba(34,211,238,0.04);
    box-shadow:
        0 0 30px rgba(34,211,238,0.1),
        inset 0 0 40px rgba(34,211,238,0.04);
    transform: perspective(600px) translateZ(6px);
}
.upload-zone-icon {
    font-size: 3.2rem;
    color: #22d3ee;
    filter: drop-shadow(0 0 12px rgba(34,211,238,0.5));
    margin-bottom: 1rem;
    display: block;
    animation: float-icon 3s ease-in-out infinite;
}
@keyframes float-icon {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-7px); }
}
.upload-zone h3 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 0 6px 0 !important;
}
.upload-zone p {
    font-size: 0.84rem !important;
    color: #64748b !important;
    margin: 0 !important;
}
.upload-caption {
    font-size: 0.73rem;
    color: #475569;
    text-align: center;
    margin-top: 6px;
    font-style: italic;
}

/* Completely hide Streamlit's native file uploader widget visually,
   but keep it functional so uploads still work */
div[data-testid="stFileUploader"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}
div[data-testid="stFileUploader"] > label { display: none !important; }
div[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
    padding: 4px 0 !important;
}
div[data-testid="stFileUploader"] section > button {
    background: rgba(34,211,238,0.08) !important;
    border: 1px solid rgba(34,211,238,0.3) !important;
    color: #22d3ee !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stFileUploader"] section > div small {
    color: #475569 !important;
}

/* ── HIDE STREAMLIT RADIO WIDGET (replaced by custom HTML) ──── */
div[data-testid="stRadio"] { display: none !important; }

/* ── RESULTS PLACEHOLDER ─────────────────────────────────────── */
.results-placeholder {
    padding: 2.5rem 1rem;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 340px;
}
.placeholder-icon {
    font-size: 3.2rem;
    color: #334155;
    margin-bottom: 1.1rem;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));
}
.results-placeholder h3 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin-bottom: 0.4rem !important;
}
.results-placeholder > p {
    font-size: 0.84rem !important;
    color: #64748b !important;
    max-width: 290px;
    line-height: 1.6;
    margin-bottom: 1.5rem !important;
}

/* ── QUICK STATS BOX ─────────────────────────────────────────── */
.quick-stats-box {
    width: 100%;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: left;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.quick-stats-box h4 {
    font-family: 'Outfit', sans-serif;
    font-size: 0.72rem;
    color: #22d3ee;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 0.6rem 0;
}
.stats-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.4rem; }
.stat-item { display: flex; flex-direction: column; }
.stat-label { font-size: 0.63rem; color: #475569; }
.stat-val { font-family:'Outfit',sans-serif; font-size:0.84rem; font-weight:600; color:#f1f5f9; }

/* ── RESULT BOX ──────────────────────────────────────────────── */
.result-box {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin: 0.75rem 0;
    transform: perspective(600px) translateZ(4px);
}
.result-cancer {
    background: rgba(225,29,72,0.08) !important;
    border: 1px solid rgba(225,29,72,0.3) !important;
    color: #fda4af !important;
    box-shadow: 0 0 30px rgba(225,29,72,0.08) !important;
}
.result-benign {
    background: rgba(5,150,105,0.08) !important;
    border: 1px solid rgba(5,150,105,0.3) !important;
    color: #a7f3d0 !important;
    box-shadow: 0 0 30px rgba(5,150,105,0.08) !important;
}
.result-box h3 { font-family:'Outfit',sans-serif; font-size:1.05rem; font-weight:700; margin-bottom:0.35rem; }
.result-box p  { font-size:0.85rem; line-height:1.55; }

/* ── PIPELINE ARCH CARDS ─────────────────────────────────────── */
.arch-card {
    background: linear-gradient(145deg, rgba(22,33,58,0.6), rgba(14,22,40,0.75));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    height: 100%;
    transform: perspective(700px) translateZ(0);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}
.arch-card:hover {
    transform: perspective(700px) rotateX(-2deg) rotateY(2deg) translateZ(10px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.45), 0 0 20px rgba(34,211,238,0.08);
    border-color: rgba(34,211,238,0.28);
}
.arch-num {
    font-family:'Outfit',sans-serif;
    font-size:1.5rem; font-weight:800;
    color:#22d3ee; opacity:0.5; margin-bottom:0.2rem;
}
.arch-card h3 {
    font-family:'Outfit',sans-serif;
    font-size:0.95rem; font-weight:700;
    color:#fff; margin-bottom:0.4rem;
}
.arch-card p { font-size:0.81rem; color:#64748b; line-height:1.6; }

/* ── SECTION HEADERS ─────────────────────────────────────────── */
.section-hdr {
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.3rem;
}
.section-hdr h2 {
    font-family:'Outfit',sans-serif !important;
    font-size:1.2rem !important; font-weight:700 !important;
    color:#fff !important; margin:0 0 3px 0 !important;
    display:flex; align-items:center; gap:0.45rem;
}
.section-hdr h2 i { color:#22d3ee; }
.section-hdr p { font-size:0.82rem; color:#64748b; margin:0; }

/* ── COMPARISON TABLE ────────────────────────────────────────── */
.comparison-table { width:100%; border-collapse:collapse; font-size:0.83rem; margin-top:1rem; }
.comparison-table th {
    background:rgba(34,211,238,0.08); color:#22d3ee;
    font-family:'Outfit',sans-serif; font-weight:700;
    padding:0.65rem 0.9rem; text-align:left;
    border-bottom:1px solid rgba(34,211,238,0.2);
}
.comparison-table td {
    padding:0.6rem 0.9rem;
    border-bottom:1px solid rgba(255,255,255,0.05);
    color:#94a3b8; vertical-align:top;
}
.comparison-table tr:hover td { background:rgba(255,255,255,0.02); }
.text-green { color:#34d399; font-weight:600; }
.text-red   { color:#f87171; font-weight:600; }

/* ── GLOBAL TEXT OVERRIDES ───────────────────────────────────── */
h1,h2,h3,h4,h5,h6 { color:#f1f5f9 !important; font-family:'Outfit',sans-serif !important; }
p, span, li, td, th, div { color:#94a3b8; }
.stMarkdown p { color:#94a3b8 !important; }

/* metrics / info / warning widgets */
div[data-testid="stMetric"] {
    background:rgba(20,30,50,0.5) !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:12px !important; padding:0.75rem 1rem !important;
}
div[data-testid="stMetricValue"] { color:#22d3ee !important; }
.stInfo    { background:rgba(34,211,238,0.06)!important; border:1px solid rgba(34,211,238,0.2)!important; }
.stWarning { background:rgba(217,119,6,0.09)!important; border:1px solid rgba(217,119,6,0.2)!important; }
.stError   { background:rgba(225,29,72,0.09)!important; border:1px solid rgba(225,29,72,0.2)!important; }
.stSuccess { background:rgba(5,150,105,0.09)!important; border:1px solid rgba(5,150,105,0.2)!important; }

div[data-testid="stExpander"] {
    background:rgba(20,30,50,0.4)!important;
    border:1px solid rgba(255,255,255,0.07)!important;
    border-radius:10px!important;
}

/* Tables */
thead tr th {
    background:rgba(34,211,238,0.08)!important; color:#22d3ee!important;
    font-family:'Outfit',sans-serif!important; font-weight:700!important;
    padding:0.55rem 0.8rem!important;
    border-bottom:1px solid rgba(34,211,238,0.2)!important;
}
tbody tr td {
    color:#94a3b8!important;
    border-bottom:1px solid rgba(255,255,255,0.05)!important;
    padding:0.5rem 0.8rem!important;
}
tbody tr:hover td { background:rgba(255,255,255,0.02)!important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0a0e17; }
::-webkit-scrollbar-thumb { background:rgba(34,211,238,0.3); border-radius:4px; }

/* ── GRAD-CAM SECTION ────────────────────────────────────────── */
.gradcam-section {
    margin-top: 2rem;
    padding: 1.8rem 1.8rem 1.4rem;
    background: linear-gradient(145deg, rgba(22,33,58,0.72) 0%, rgba(14,22,40,0.82) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,165,0,0.18);
    border-radius: 22px;
    box-shadow:
        0 12px 48px rgba(0,0,0,0.45),
        0 0 0 1px rgba(255,165,0,0.06),
        inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.gradcam-section::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(255,130,0,0.04) 0%, transparent 55%);
    pointer-events: none;
}
.gradcam-hdr {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.4rem;
}
.gradcam-hdr-icon {
    font-size: 1.55rem;
    color: #fb923c;
    filter: drop-shadow(0 0 10px rgba(251,146,60,0.5));
}
.gradcam-hdr-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.18rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
    line-height: 1.2;
}
.gradcam-hdr-sub {
    font-size: 0.80rem;
    color: #64748b;
    margin: 2px 0 0 0;
}
.gradcam-badge {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(251,146,60,0.09);
    border: 1px solid rgba(251,146,60,0.35);
    color: #fb923c;
    padding: 0.35rem 0.85rem;
    border-radius: 30px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    white-space: nowrap;
}
.gradcam-img-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.80rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    text-align: center;
    margin-bottom: 0.55rem;
}
.gradcam-img-wrap {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.gradcam-img-wrap:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.5), 0 0 20px rgba(251,146,60,0.08);
    border-color: rgba(251,146,60,0.25);
}
.gradcam-conf-bar-track {
    width: 100%;
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.gradcam-conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
    background: linear-gradient(90deg, #fb923c, #f97316);
    box-shadow: 0 0 8px rgba(251,146,60,0.4);
}
.gradcam-explanation {
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    margin-top: 1.2rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
}
.gradcam-explanation h4 {
    font-family: 'Outfit', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #fb923c;
    margin: 0 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.gradcam-explanation p {
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.65;
    margin: 0;
}
.gradcam-download-btn {
    margin-top: 1rem;
    display: flex;
    justify-content: center;
}
/* Style the Streamlit download button to match our theme */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(251,146,60,0.12), rgba(249,115,22,0.12)) !important;
    border: 1.5px solid rgba(251,146,60,0.45) !important;
    color: #fb923c !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 0 14px rgba(251,146,60,0.1) !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, rgba(251,146,60,0.22), rgba(249,115,22,0.22)) !important;
    box-shadow: 0 0 24px rgba(251,146,60,0.25) !important;
    border-color: #fb923c !important;
}
.gradcam-metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.gradcam-metric {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.75rem 1rem;
}
.gradcam-metric-label {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 3px;
}
.gradcam-metric-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #fb923c;
}
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load Random Forest model, scaler, metadata, and optional legacy CNN."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None, None
    try:
        model   = joblib.load(MODEL_PATH)
        scaler  = joblib.load(SCALER_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        cnn_model = None
        if os.path.exists(CNN_PATH):
            try:
                import tensorflow as tf
                cnn_model = tf.keras.models.load_model(CNN_PATH)
            except Exception as e:
                print(f"[Warning] CNN load failed: {e}")
        return model, scaler, cnn_model, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


@st.cache_resource
def load_gradcam_model(num_classes: int):
    """
    Load and cache the MobileNetV2-based Grad-CAM model.

    This function is decorated with @st.cache_resource so the model is
    built only ONCE per Streamlit session (or server restart), keeping
    Grad-CAM inference fast on subsequent image uploads.

    Args:
        num_classes: Number of output classes from metadata.

    Returns:
        tf.keras.Model or None if TensorFlow is not available.
    """
    try:
        model = build_gradcam_model(num_classes)
        return model
    except ImportError:
        return None   # PyTorch not installed — Grad-CAM section will show a warning
    except Exception as e:
        print(f"[Warning] Grad-CAM model build failed: {e}")
        return None


def get_recommendation(class_name, is_cancer, confidence, risk_level):
    if is_cancer:
        if risk_level == "High":
            return (f"CRITICAL ASSESSMENT: Features highly consistent with {class_name} (Cancer Detected). "
                    f"Urgent dermatologist appointment and potential biopsy recommended. Do not delay.")
        else:
            return (f"MODERATE ASSESSMENT: Model detects signs of {class_name} (Cancer Detected). "
                    f"Please consult a dermatologist and monitor the lesion closely for any changes.")
    else:
        if risk_level == "Medium":
            return (f"MONITORING ADVISED: Classified as {class_name} (No Cancer Detected). "
                    f"Moderate confidence – seek dermatologist opinion if itching, bleeding, or asymmetrical growth occurs.")
        else:
            return (f"ROUTINE CARE: Classified as {class_name} (No Cancer Detected). "
                    f"Typical benign pattern. Use SPF 30+ sunscreen and perform regular self-exams.")


def run_prediction_pipeline(img, model, scaler, cnn_model, metadata):
    is_valid, val_msg = validate_skin_lesion_image(img)
    if not is_valid:
        return {"success": False, "error": val_msg}

    features, feature_dict = extract_features(img)
    features_scaled = scaler.transform(features.reshape(1, -1))
    rf_probs = model.predict_proba(features_scaled)[0]
    rf_idx   = np.argmax(rf_probs)
    rf_conf  = float(rf_probs[rf_idx] * 100)
    rf_class = metadata["classes"][rf_idx]

    cnn_results = None
    if cnn_model is not None:
        try:
            img_cv    = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_r     = cv2.resize(img_cv, (64, 64)) / 255.0
            cnn_probs = cnn_model.predict(np.expand_dims(img_r, axis=0))[0]
            cnn_idx   = np.argmax(cnn_probs)
            cnn_results = {
                "class": metadata["classes"][cnn_idx],
                "confidence": round(float(cnn_probs[cnn_idx]) * 100, 2),
                "probs": {metadata["classes"][i]: round(float(p)*100, 2) for i, p in enumerate(cnn_probs)}
            }
        except Exception as e:
            print(f"[Warning] CNN prediction failed: {e}")

    is_cancer  = metadata["cancer_mapping"].get(rf_class, False)
    risk_level = "High" if (is_cancer and rf_conf >= 60) else ("Medium" if (is_cancer or rf_conf >= 85) else "Low")
    recommendation = get_recommendation(rf_class, is_cancer, rf_conf, risk_level)

    return {
        "success": True,
        "class": rf_class,
        "confidence": round(rf_conf, 2),
        "is_cancer": is_cancer,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "features": feature_dict,
        "rf_probs": {metadata["classes"][i]: round(float(p)*100, 2) for i, p in enumerate(rf_probs)},
        "cnn": cnn_results
    }


# ─── APP HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
    <header class="app-header">
        <div class="header-left">
            <i class="fa-solid fa-microscope header-icon"></i>
            <div>
                <div class="header-title">Skin Cancer Detection Using Machine Learning</div>
                <div class="header-subtitle">Random Forest Based Intelligent Skin Lesion Analysis</div>
            </div>
        </div>
        <div class="header-badge">
            <i class="fa-solid fa-shield-halved"></i> Clinical Assistant V1.0
        </div>
    </header>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
model, scaler, cnn_model, metadata = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1.05, 0.95], gap="medium")

# ── Left Card: Lesion Input Center ────────────────────────────────────────
with col1:
    # Card header
    st.markdown("""
        <div class="card-hdr">
            <h2><i class="fa-solid fa-camera-rotate" style="color:#22d3ee;font-size:1rem;"></i>
                Lesion Input Center</h2>
            <p>Select input mode and upload or capture skin lesion image</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Custom styled toggle buttons ─────────────────────────────────────
    # We still use st.radio but hide it with CSS; we drive the UX with JS-free HTML buttons
    # The session state tracks which mode is active
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "upload"

    col_u, col_c = st.columns(2)
    with col_u:
        if st.button("☁  File Upload", key="btn_upload", use_container_width=True):
            st.session_state.input_mode = "upload"
    with col_c:
        if st.button("🎥  Live Camera", key="btn_cam", use_container_width=True):
            st.session_state.input_mode = "webcam"

    # Inject CSS to highlight whichever button is active
    active_mode = st.session_state.input_mode
    st.markdown(f"""
        <style>
        /* Style all buttons dark by default */
        div[data-testid="column"] button[kind="secondary"] {{
            background: rgba(255,255,255,0.02) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            color: #64748b !important;
            border-radius: 12px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding: 0.65rem 1rem !important;
            transition: all 0.22s ease !important;
        }}
        div[data-testid="column"] button[kind="secondary"]:hover {{
            border-color: rgba(34,211,238,0.3) !important;
            color: #cbd5e1 !important;
        }}
        /* Active button – highlight whichever key matches */
        button[data-testid="baseButton-secondary"][aria-label="☁  File Upload"]
            {"" if active_mode == "upload" else "display:none;display:block;"} {{}}
        </style>
    """, unsafe_allow_html=True)

    # Inject active state via targeted CSS using button order
    if active_mode == "upload":
        st.markdown("""
            <style>
            div[data-testid="column"]:nth-of-type(1) button[kind="secondary"] {
                background: rgba(34,211,238,0.08) !important;
                border: 1.5px solid #22d3ee !important;
                color: #22d3ee !important;
                box-shadow: 0 0 16px rgba(34,211,238,0.14), inset 0 1px 0 rgba(34,211,238,0.08) !important;
                transform: perspective(400px) translateZ(3px) !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            div[data-testid="column"]:nth-of-type(2) button[kind="secondary"] {
                background: rgba(34,211,238,0.08) !important;
                border: 1.5px solid #22d3ee !important;
                color: #22d3ee !important;
                box-shadow: 0 0 16px rgba(34,211,238,0.14), inset 0 1px 0 rgba(34,211,238,0.08) !important;
                transform: perspective(400px) translateZ(3px) !important;
            }
            </style>
        """, unsafe_allow_html=True)

    uploaded_image = None

    if active_mode == "upload":
        # ── Custom styled drag-and-drop zone (cosmetic) ──────────────────
        st.markdown("""
            <div class="upload-zone">
                <i class="fa-solid fa-images upload-zone-icon"></i>
                <h3>Drag &amp; Drop Image Here</h3>
                <p>or click to browse local files (JPG, PNG, BMP)</p>
            </div>
            <p class="upload-caption">Optimized for high-resolution macro lesion photos</p>
        """, unsafe_allow_html=True)

        # Native uploader (functional, visually minimal)
        file_upload = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )
        if file_upload is not None:
            uploaded_image = Image.open(file_upload)
    else:
        # Camera permission guide
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(34,211,238,0.06), rgba(6,182,212,0.04));
                border: 1px solid rgba(34,211,238,0.25);
                border-radius: 14px;
                padding: 0.85rem 1.1rem;
                margin-bottom: 0.85rem;
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
            ">
                <i class="fa-solid fa-circle-info" style="color:#22d3ee;font-size:1.1rem;margin-top:2px;flex-shrink:0;"></i>
                <div>
                    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:0.88rem;color:#22d3ee;margin-bottom:3px;">Camera Permission Required</div>
                    <div style="font-size:0.80rem;color:#64748b;line-height:1.6;">
                        When prompted by your browser, click <strong style="color:#e2e8f0;">Allow</strong> to grant camera access.
                        If blocked, click the <strong style="color:#e2e8f0;">🔒 lock icon</strong> in your browser address bar → Camera → Allow → Reload page.
                    </div>
                </div>
            </div>

            <style>
            /* Style the camera widget to match dark theme */
            div[data-testid="stCameraInput"] {
                background: rgba(10,20,38,0.55) !important;
                border: 2px dashed rgba(34,211,238,0.35) !important;
                border-radius: 16px !important;
                overflow: hidden !important;
            }
            div[data-testid="stCameraInput"] > label {
                font-family: 'Outfit', sans-serif !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                color: #22d3ee !important;
                padding: 0.5rem 0 !important;
            }
            div[data-testid="stCameraInput"] video {
                border-radius: 12px !important;
            }
            div[data-testid="stCameraInput"] button {
                background: linear-gradient(135deg, #22d3ee, #06b6d4) !important;
                color: #0a0e17 !important;
                font-family: 'Outfit', sans-serif !important;
                font-weight: 700 !important;
                border: none !important;
                border-radius: 10px !important;
                font-size: 0.88rem !important;
            }
            </style>
        """, unsafe_allow_html=True)

        camera_upload = st.camera_input("📷  Take Live Photo of Lesion", label_visibility="visible")
        if camera_upload is not None:
            uploaded_image = Image.open(camera_upload)

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Selected Lesion Preview", use_container_width=True)

# ── Right Card: Diagnostic Assessment ─────────────────────────────────────
with col2:
    st.markdown("""
        <div class="card-hdr">
            <h2><i class="fa-solid fa-heart-pulse" style="color:#22d3ee;font-size:1rem;"></i>
                Diagnostic Assessment</h2>
            <p>Real-time ML analysis and classifier outputs</p>
        </div>
    """, unsafe_allow_html=True)

    if uploaded_image is None:
        accuracy_str = f"{metadata['rf_accuracy']*100:.1f}%" if metadata else "N/A"
        st.markdown(f"""
            <div class="results-placeholder">
                <i class="fa-solid fa-receipt placeholder-icon"></i>
                <h3>Awaiting Lesion Input</h3>
                <p>Upload a skin lesion image or start the live camera to trigger
                the Traditional Machine Learning diagnosis sequence.</p>
                <div class="quick-stats-box">
                    <h4>Model Calibration Details</h4>
                    <div class="stats-row">
                        <div class="stat-item">
                            <span class="stat-label">Model Classifier</span>
                            <span class="stat-val">Random Forest</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Feature Pipeline</span>
                            <span class="stat-val">HSV + GLCM + Contours</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Cross-Val Accuracy</span>
                            <span class="stat-val">{accuracy_str}</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        if model is None or scaler is None:
            st.error("Model assets not found. Please train the model first.")
        else:
            with st.spinner("Running diagnostic sequence..."):
                results = run_prediction_pipeline(uploaded_image, model, scaler, cnn_model, metadata)

            if not results["success"]:
                st.error(results["error"])
            else:
                box_cls     = "result-cancer" if results["is_cancer"] else "result-benign"
                status_text = "Cancer Detected" if results["is_cancer"] else "No Cancer Detected"

                st.markdown(f"""
                    <div class="result-box {box_cls}">
                        <h3>{status_text} &mdash; {results['class']}</h3>
                        <p><strong>Risk Index:</strong> {results['risk_level']}
                           &nbsp;|&nbsp;
                           <strong>RF Confidence:</strong> {results['confidence']}%</p>
                        <p style="margin-bottom:0">{results['recommendation']}</p>
                    </div>
                """, unsafe_allow_html=True)

                if results["cnn"] is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Random Forest Probabilities**")
                        st.bar_chart(results["rf_probs"])
                    with c2:
                        st.write(f"**CNN Probabilities** (Top: {results['cnn']['class']})")
                        st.bar_chart(results["cnn"]["probs"])
                else:
                    st.write("**Classifier Probabilities**")
                    st.bar_chart(results["rf_probs"])

                with st.expander("🔬 View Extracted Feature Vector (Color · Texture · Shape)"):
                    feat_df = pd.DataFrame([
                        {
                            "Feature Group": "Color" if "color" in k else ("Texture" if "texture" in k else "Shape"),
                            "Descriptor Name": k,
                            "Normalized Value": round(float(v), 5)
                        }
                        for k, v in results["features"].items()
                    ])
                    st.dataframe(feat_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM VISUALIZATION SECTION
# Appears below the main diagnostic layout whenever an image is uploaded
# and the Random Forest prediction succeeds.
# ══════════════════════════════════════════════════════════════════════════════

if uploaded_image is not None and model is not None and scaler is not None:
    # Re-run prediction (cached in session state to avoid double compute)
    if "gradcam_results" not in st.session_state or st.session_state.get("gradcam_img_id") != id(uploaded_image):
        prediction = run_prediction_pipeline(uploaded_image, model, scaler, cnn_model, metadata)
        st.session_state["gradcam_results"]  = prediction
        st.session_state["gradcam_img_id"]   = id(uploaded_image)
    else:
        prediction = st.session_state["gradcam_results"]

    if prediction.get("success"):
        # ── Section Header ────────────────────────────────────────────────────
        st.markdown("""
            <div class="gradcam-section">
                <div class="gradcam-hdr">
                    <i class="fa-solid fa-fire-flame-curved gradcam-hdr-icon"></i>
                    <div>
                        <div class="gradcam-hdr-title">Grad-CAM Explainability Analysis</div>
                        <div class="gradcam-hdr-sub">
                            Gradient-weighted Class Activation Mapping · MobileNetV2 Backbone
                        </div>
                    </div>
                    <span class="gradcam-badge">
                        <i class="fa-solid fa-eye"></i> XAI Visualization
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Load / retrieve cached Grad-CAM model ─────────────────────────────
        num_classes    = len(metadata["classes"])
        gradcam_model  = load_gradcam_model(num_classes)

        # Determine RF-predicted class index for Grad-CAM guidance
        rf_class_name  = prediction["class"]
        rf_class_idx   = metadata["classes"].index(rf_class_name) \
                         if rf_class_name in metadata["classes"] else 0

        if gradcam_model is None:
            # TensorFlow not installed — show informative warning
            st.warning(
                "⚠️ **Grad-CAM unavailable:** PyTorch is not installed. "
                "Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` "
                "and restart the app to enable explainability visualization."
            )
        else:
            # ── Run Grad-CAM pipeline ─────────────────────────────────────────
            with st.spinner("🔥 Generating Grad-CAM heatmap via MobileNetV2..."):
                gc_result = generate_gradcam(
                    pil_img     = uploaded_image,
                    class_idx   = rf_class_idx,
                    num_classes = num_classes,
                    cached_model= gradcam_model
                )

            if not gc_result["success"]:
                # Grad-CAM failed — show error, RF results still visible above
                st.error(
                    f"⚠️ **Grad-CAM Error:** {gc_result['error']} "
                    "The Random Forest prediction above is still valid."
                )
            else:
                # ── Metrics row ───────────────────────────────────────────────
                is_cancer     = prediction["is_cancer"]
                conf_pct      = prediction["confidence"]
                risk_color    = "#f87171" if is_cancer else "#34d399"
                risk_label    = prediction["risk_level"]
                status_txt    = "⚠️ Cancer Detected" if is_cancer else "✅ No Cancer Detected"

                st.markdown(f"""
                    <div class="gradcam-metric-row">
                        <div class="gradcam-metric">
                            <div class="gradcam-metric-label">Diagnosis</div>
                            <div class="gradcam-metric-val" style="color:{risk_color};font-size:0.92rem">
                                {status_txt}
                            </div>
                        </div>
                        <div class="gradcam-metric">
                            <div class="gradcam-metric-label">Classified As</div>
                            <div class="gradcam-metric-val" style="font-size:0.85rem">
                                {prediction['class']}
                            </div>
                        </div>
                        <div class="gradcam-metric">
                            <div class="gradcam-metric-label">RF Confidence</div>
                            <div class="gradcam-metric-val">{conf_pct:.2f}%</div>
                            <div class="gradcam-conf-bar-track">
                                <div class="gradcam-conf-bar-fill" style="width:{min(conf_pct,100):.1f}%;"></div>
                            </div>
                        </div>
                        <div class="gradcam-metric">
                            <div class="gradcam-metric-label">Risk Index</div>
                            <div class="gradcam-metric-val" style="color:{risk_color};">{risk_label}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # ── Three-column image display ────────────────────────────────
                gc1, gc2, gc3 = st.columns(3, gap="medium")

                with gc1:
                    st.markdown('<div class="gradcam-img-label">📷 Original Image</div>',
                                unsafe_allow_html=True)
                    display_img = preprocess_for_display(uploaded_image)
                    st.image(display_img, use_container_width=True, caption="Input lesion image")

                with gc2:
                    st.markdown('<div class="gradcam-img-label">🔥 Grad-CAM Heatmap</div>',
                                unsafe_allow_html=True)
                    st.image(
                        gc_result["heatmap_pil"],
                        use_container_width=True,
                        caption="Class activation map (red = high focus)"
                    )

                with gc3:
                    st.markdown('<div class="gradcam-img-label">🩺 Suspicious Region</div>',
                                unsafe_allow_html=True)
                    st.image(
                        gc_result["annotated_pil"],
                        use_container_width=True,
                        caption="Overlay with detected suspicious region"
                    )

                # ── Medical Explanation ───────────────────────────────────────
                explanation = prediction["recommendation"]

                # Build gradient color for the explanation box
                if is_cancer:
                    expl_border = "rgba(248,113,113,0.25)"
                    expl_bg     = "rgba(248,113,113,0.05)"
                    icon_color  = "#f87171"
                    expl_icon   = "fa-triangle-exclamation"
                else:
                    expl_border = "rgba(52,211,153,0.25)"
                    expl_bg     = "rgba(52,211,153,0.05)"
                    icon_color  = "#34d399"
                    expl_icon   = "fa-circle-check"

                st.markdown(f"""
                    <div class="gradcam-explanation" style="
                        border-color:{expl_border};
                        background:{expl_bg};
                    ">
                        <h4>
                            <i class="fa-solid {expl_icon}" style="color:{icon_color};"></i>
                            Medical Explanation
                        </h4>
                        <p>{explanation}</p>
                        <p style="margin-top:0.55rem;color:#64748b;font-size:0.78rem;">
                            <strong style="color:#fb923c;">How to read this:</strong>
                            The heatmap highlights regions the model focused on.
                            <span style="color:#ef4444;">Red/warm areas</span> = high activation (model attention),
                            <span style="color:#3b82f6;">blue/cool areas</span> = low activation.
                            The cyan contour marks the primary suspicious region.
                            This visualization does <strong>not</strong> replace clinical diagnosis.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # ── Download Button ───────────────────────────────────────────
                st.markdown(
                    '<div class="gradcam-download-btn">',
                    unsafe_allow_html=True
                )
                try:
                    overlay_bytes = pil_to_png_bytes(gc_result["annotated_pil"])
                    st.download_button(
                        label     = "⬇  Download Grad-CAM Result (PNG)",
                        data      = overlay_bytes,
                        file_name = "gradcam_result.png",
                        mime      = "image/png",
                        key       = "gradcam_download"
                    )
                except Exception as dl_err:
                    st.warning(f"Download preparation failed: {dl_err}")
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Heatmap Interpretation Guide ──────────────────────────────
                with st.expander("📖 Grad-CAM Interpretation Guide & Technical Details"):
                    st.markdown("""
                    **What is Grad-CAM?**
                    Gradient-weighted Class Activation Mapping (Grad-CAM) is an
                    XAI (Explainable AI) technique that produces visual explanations
                    for CNN predictions. It computes the gradient of the predicted
                    class score with respect to the final convolutional layer's
                    feature maps, then uses those gradients as channel weights.

                    **Model Architecture:**
                    - Backbone: **MobileNetV2** (ImageNet pre-trained, frozen)
                    - Target layer: `block_16_project` (last spatial conv layer)
                    - Input: 224×224 RGB, normalized to [0, 1]
                    - Preprocessing: Median blur → CLAHE enhancement → resize

                    **Color Scale:**
                    | Color | Meaning |
                    |---|---|
                    | 🔴 Red / Yellow | High activation — model focused heavily here |
                    | 🟢 Green | Moderate activation |
                    | 🔵 Blue | Low activation — model ignored this region |

                    **Important Disclaimer:**
                    This Grad-CAM visualization is a *decision support tool* and
                    is not a substitute for professional medical diagnosis. Always
                    consult a certified dermatologist for clinical evaluation.
                    """)

# ── Footer spacer ──────────────────────────────────────────────────────────────
st.markdown("<div style='margin-bottom:2.5rem'></div>", unsafe_allow_html=True)
