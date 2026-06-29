import streamlit as st
import os
import re
import random
import hashlib
import time
import sqlite3
import json
import uuid
import io
import httpx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 1. PERFORMANCE CACHING ---
st.set_page_config(page_title="MedSim Academy", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

def _secret(key, default=None):
    """Read from env var first (Railway), fall back to st.secrets (Streamlit Cloud)."""
    val = os.environ.get(key)
    if val is not None:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

def load_scope_reference(standard: str = "PWC") -> str:
    """Load the appropriate scope file — PWC local protocols or NREMT national standards."""
    fname = "NREMT_scope.txt" if standard == "NREMT" else "PWC_scope.txt"
    scope_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scope", fname)
    try:
        with open(scope_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

@st.cache_resource
def load_resources():
    # Cache the 'Brain' so it stays in RAM and never reloads during the session
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("protocol_db", embedder, allow_dangerous_deserialization=True)
    # Using 2.0-Flash as the stable fallback to bypass the 404 errors on '3' and '1.5'
    llm = ChatGoogleGenerativeAI(model=_secret("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.7, timeout=60)
    return embedder, db, llm

embeddings, vector_db, llm_engine = load_resources()

# --- LLM RETRY HELPER ---
def _invoke_with_retry(messages, max_attempts=4):
    """
    Invoke llm_engine with exponential back-off for transient API errors.
    Handles Google 503/429/UNAVAILABLE and httpx network errors.
    Delays: 3s → 6s → 12s before giving up.
    """
    delay = 3
    last_err = None
    for attempt in range(max_attempts):
        try:
            return llm_engine.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            retryable = any(x in err_str for x in [
                "503", "502", "529", "unavailable", "overloaded",
                "429", "resource exhausted", "quota", "try again",
                "remotedisconnected", "connectionreset", "connectionerror",
            ])
            if retryable and attempt < max_attempts - 1:
                last_err = e
                time.sleep(delay * (2 ** attempt))  # 3s, 6s, 12s
            else:
                raise
    raise last_err or RuntimeError("_invoke_with_retry: no attempts made")

# --- 2. ACCESS CONTROL ---
def _auth_token():
    return hashlib.sha256(_secret("APP_PASSWORD", "").encode()).hexdigest()[:20]

def check_password():
    # Auto-pass if valid token is in URL (survives WebSocket reconnects)
    if st.query_params.get("auth") == _auth_token():
        st.session_state["password_correct"] = True

    if "password_correct" not in st.session_state:
        st.markdown('<div style="text-align:center; padding: 50px;"><h1>🚑 MedSim Academy</h1><p>Restricted Access - Enter Credentials</p></div>', unsafe_allow_html=True)
        pwd = st.text_input("Access Code", type="password")
        if pwd:
            if pwd == _secret("APP_PASSWORD"):
                st.session_state["password_correct"] = True
                st.query_params["auth"] = _auth_token()
                st.rerun()

            else:
                st.error("😕 Access Denied.")
        return False
    return True

if not check_password():
    st.stop()

# --- CSS: THE "STAY PUT" MONITOR ---
st.markdown("""
    <style>
        /* 1. Allow sticky to work — no overflow clipping on ancestors */
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stVerticalBlock"] {
            overflow: visible !important;
        }

        /* 2. Parent flex row: align-items:flex-start so columns shrink to content
              height instead of stretching — required for sticky to activate */
        [data-testid="stHorizontalBlock"] {
            overflow: visible !important;
            align-items: flex-start !important;
        }

        /* 3. Sticky monitor: target the last column (always col_data regardless
              of render order). align-self:flex-start reinforces the parent rule. */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            position: -webkit-sticky !important;
            position: sticky !important;
            top: 2rem !important;
            z-index: 1000 !important;
            align-self: flex-start !important;
        }

        /* 4. Cap the inner content so a long timeline can't push vitals off screen */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > div {
            max-height: calc(100vh - 5rem) !important;
            overflow-y: auto !important;
        }

        /* 3. The Monitor "Rugged Tablet" Look */
        .vital-card {
            background-color: #000;
            border: 2px solid #444;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }
        .vital-label { color: #00FF00; font-size: 0.7rem; text-transform: uppercase; opacity: 0.8; }
        .vital-value { color: #00FF00; font-size: 1.4rem; font-weight: bold; font-family: 'Courier New', monospace; }
        .log-entry { font-size: 0.85rem; color: #00FF00; font-family: monospace; border-bottom: 1px solid #222; padding: 4px 0; }

        /* ── Mobile layout ───────────────────────────────────────────────── */
        @media (max-width: 768px) {
            /* Disable sticky monitor panel — columns are stacked, not side-by-side */
            div[data-testid="stColumn"]:nth-of-type(3) > div {
                position: relative !important;
                top: 0 !important;
            }

            /* Compact vital cards so 2-up grid fits on a phone */
            .vital-card {
                padding: 8px 6px;
                margin-bottom: 5px;
                border-radius: 7px;
            }
            .vital-label { font-size: 0.6rem; }
            .vital-value { font-size: 1.05rem; }

            /* Tighter timeline entries */
            .log-entry { font-size: 0.75rem; padding: 3px 0; }

            /* Hint pills — let them wrap on narrow screens */
            div[data-testid="stMarkdown"] > div > p {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
            }
        }

        /* ── Leaderboard navbar ──────────────────────────────────────────── */

        /* Desktop: hide Streamlit header — sidebar toggle is handled by the drawer tab below */
        @media (min-width: 769px) {
            header[data-testid="stHeader"] { display: none !important; }
            .main .block-container { padding-top: 3.5rem !important; }
        }

        /* Mobile: keep header transparent so hamburger stays tappable above our navbar */
        @media (max-width: 768px) {
            header[data-testid="stHeader"] {
                background: transparent !important;
                box-shadow: none !important;
                border-bottom: none !important;
                z-index: 10000 !important;
            }
            #ms-navbar { padding-left: 56px !important; }
        }

        /* Sidebar drawer tab — visible navy pull handle on all screen sizes */
        [data-testid="collapsedControl"] {
            background-color: #1A2E5A !important;
            border-radius: 0 8px 8px 0 !important;
            width: 20px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 9998 !important;
        }
        [data-testid="collapsedControl"] button {
            background: transparent !important;
            border-radius: 0 8px 8px 0 !important;
            width: 20px !important;
            padding: 0 !important;
        }
        [data-testid="collapsedControl"] svg,
        [data-testid="collapsedControl"] button {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }

        #ms-chk { display: none; }

        #ms-navbar {
            position: fixed;
            top: 0; left: 0; right: 0;
            z-index: 9999;
            background: #1A2E5A;
            padding: 0 20px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: sans-serif;
        }
        #ms-navbar .nav-title { color: #fff; font-size: 15px; font-weight: 600; }
        #ms-navbar .nav-right { display: flex; align-items: center; gap: 12px; }
        #ms-navbar .nav-station { color: #A8CFDA; font-size: 12px; }
        label[for="ms-chk"] {
            background: rgba(255,255,255,0.15);
            border: none;
            border-radius: 6px;
            width: 32px;
            height: 28px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3px;
            padding: 0;
        }
        label[for="ms-chk"]:hover { background: rgba(255,255,255,0.25); }
        .ms-dot { width: 4px; height: 4px; border-radius: 50%; background: #fff; display: inline-block; }
        #ms-lb-panel {
            display: none;
            position: fixed;
            top: 48px; right: 20px;
            width: 280px;
            background: #fff;
            border: 1px solid rgba(0,0,0,0.12);
            border-radius: 12px;
            overflow: hidden;
            z-index: 9998;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            font-family: sans-serif;
        }
        #ms-chk:checked ~ #ms-lb-panel { display: block; }
        #ms-lb-panel .lb-head { background: #1A2E5A; padding: 14px 16px; }
        #ms-lb-panel .lb-rank-big { font-size: 30px; font-weight: 700; color: #fff; line-height: 1; }
        #ms-lb-panel .lb-rank-sub { font-size: 12px; color: #A8CFDA; margin-top: 3px; }
        .lb-row { display: flex; align-items: center; padding: 8px 16px; gap: 10px; border-bottom: 0.5px solid rgba(0,0,0,0.08); font-size: 13px; color: #222; }
        .lb-row.lb-me { background: #E1F5EE; }
        .lb-row .r-pos { width: 22px; text-align: center; color: #888; font-size: 12px; font-weight: 500; }
        .lb-row .r-pos.r-me { color: #0F6E56; font-weight: 700; }
        .lb-row .r-name { flex: 1; }
        .lb-row .r-count { color: #888; font-size: 12px; }
        .lb-row .r-medal { width: 22px; text-align: center; font-size: 14px; }
        .lb-you { font-size: 10px; background: #1D9E75; color: #fff; padding: 2px 6px; border-radius: 4px; margin-left: 5px; vertical-align: middle; }
        .lb-sep { padding: 3px 16px; text-align: center; font-size: 11px; color: #aaa; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<input type="checkbox" id="ms-chk">
<div id="ms-navbar">
    <span class="nav-title">MedSim Academy</span>
    <div class="nav-right">
        <span class="nav-station">Station 516 &middot; B shift</span>
        <label for="ms-chk" aria-label="Leaderboard">
            <span class="ms-dot"></span><span class="ms-dot"></span><span class="ms-dot"></span>
        </label>
    </div>
</div>

<div id="ms-lb-panel">
    <div class="lb-head">
        <div style="display:flex;align-items:baseline;gap:10px;">
            <div class="lb-rank-big">#14</div>
            <div>
                <div class="lb-rank-sub">Station 516 B shift</div>
                <div class="lb-rank-sub" style="margin-top:2px;">34 calls this month</div>
            </div>
        </div>
    </div>
    <div class="lb-row"><div class="r-medal">&#129351;</div><div class="r-name">St. 508 A shift</div><div class="r-count">89 calls</div></div>
    <div class="lb-row"><div class="r-medal">&#129352;</div><div class="r-name">St. 503 C shift</div><div class="r-count">81 calls</div></div>
    <div class="lb-row"><div class="r-medal">&#129353;</div><div class="r-name">St. 512 B shift</div><div class="r-count">76 calls</div></div>
    <div class="lb-sep">&middot; &middot; &middot;</div>
    <div class="lb-row"><div class="r-pos">13</div><div class="r-name">St. 521 A shift</div><div class="r-count">36 calls</div></div>
    <div class="lb-row lb-me"><div class="r-pos r-me">14</div><div class="r-name">St. 516 B shift<span class="lb-you">you</span></div><div class="r-count">34 calls</div></div>
    <div class="lb-row"><div class="r-pos">15</div><div class="r-name">St. 519 C shift</div><div class="r-count">31 calls</div></div>
</div>
""", unsafe_allow_html=True)

# --- 4. ENGINE FUNCTIONS ---
def get_protocol_context(query):
    # Retrieve top 2 chunks to minimize prompt size and maximize speed
    docs = vector_db.similarity_search(query, k=4)
    return "\n---\n".join([d.page_content for d in docs])

def process_medsim_turn(text):
    # 1. Update Vitals (Same as before)
    v_pattern = r"\[VITAL\]\s*(HR|BP|SPO2|RR|BGL|TEMP)[:\-]\s*([^\[\n\r]+)"
    for label, val in re.findall(v_pattern, text, re.IGNORECASE):
        st.session_state.vitals[label.upper()] = val.strip().split(' ')[0]
    
    # 2. Update Timeline with Deduplication
    l_matches = re.findall(r"\[LOG\]\s*([^\[\n\r]+)", text, re.IGNORECASE)
    for entry in l_matches:
        clean_entry = entry.strip()
        
        # Check if this exact action already exists in the timeline
        # This prevents the 'Echo' effect if the AI repeats itself
        is_duplicate = any(clean_entry in existing_log for existing_log in st.session_state.timeline)
        
        if not is_duplicate:
            ts = datetime.now() - st.session_state.get("start_time", datetime.now())
            timestamp = f"T+{ts.seconds//60:02d}:{ts.seconds%60:02d}"
            st.session_state.timeline.append(f"{timestamp} | {clean_entry}")
    
    # 3. Clean the text for display
    return re.sub(r"\[VITAL\].*?(\n|$)|\[LOG\].*?(\n|$)", "", text, flags=re.IGNORECASE).strip()

_VITALS_REQUEST_KW = re.compile(
    r"\bvital|pulse|heart rate|\bbp\b|blood pressure|spo2|o2 sat|oxygen"
    r"|resp rate|\brr\b|temperature|\btemp\b|\bbgl\b|blood glucose"
    r"|full set|baseline|trending",
    re.IGNORECASE,
)

def clean_for_display(text, show_vitals_inline: bool = True):
    """
    Process [LOG] and [VITAL] tags for chat display.

    Strategy:
    - The LLM always appends a [LOG]+[VITAL] footer. In that footer, [VITAL]
      values are '--' for most responses, but contain real values when the
      student explicitly requested vitals.
    - Strip the footer [LOG] line; if its [VITAL] tags carry real values, save
      them for injection into the narrative (so the student sees them in chat).
    - If the LLM also put [VITAL] tags directly in the narrative body, those
      take priority and the saved footer values are discarded (no duplicates).
    """
    _vital_pat = r"\[VITAL\]\s*(HR|BP|SPO2|RR|BGL|TEMP)[:\-]\s*([^\[\n\r]+)"

    # --- Pass 1: capture footer vitals then strip the [LOG]+[VITAL] block ---
    footer_real_vitals = []  # list of (KEY, "value") tuples from the footer

    def _strip_footer(m):
        pairs = re.findall(_vital_pat, m.group(0), re.IGNORECASE)
        for key, val in pairs:
            clean_val = val.strip()
            if clean_val.strip("-").strip():          # skip '--' placeholders
                footer_real_vitals.append((key.upper(), clean_val))
        return ""

    text = re.sub(
        r"\[LOG\][^\n\r]*(\n[ \t]*\[VITAL\][^\n\r]*)*",
        _strip_footer,
        text, flags=re.IGNORECASE,
    )

    # --- Pass 2: handle any [VITAL] tags still in the narrative body ---
    narrative_raw = re.findall(_vital_pat, text, re.IGNORECASE)
    has_narrative_vitals = any(v.strip().strip("-").strip() for _, v in narrative_raw)

    if has_narrative_vitals:
        # Format narrative [VITAL] tags inline as code spans; footer values
        # are discarded (narrative always wins to avoid duplicates).
        def _fmt(m):
            return f"`{m.group(1).upper()}: {m.group(2).strip()}` "
        text = re.sub(_vital_pat, _fmt, text, flags=re.IGNORECASE)
    else:
        # No narrative vitals — strip any stray [VITAL] tags that slipped through
        text = re.sub(r"\[VITAL\][^\n\r]*", "", text, flags=re.IGNORECASE)

        # Only surface footer vitals in the chat when the student asked for them.
        if footer_real_vitals and show_vitals_inline:
            vitals_line = "  ".join(f"`{k}: {v}`" for k, v in footer_real_vitals)
            text = text.rstrip() + "\n\n" + vitals_line

    # Strip media tags — the chat loop handles image display separately
    text = re.sub(r"\[ECG\]\s*\S+",   "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[CAPNO\]\s*\S+(?:\s+\d+)?", "", text, flags=re.IGNORECASE)

    # Strip [NARRATIVE] block — the chat loop renders it in an expander separately
    text = re.sub(r"\[NARRATIVE\].*", "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


# ── ECG image helpers ─────────────────────────────────────────────────────────
ECG_SLUGS = {
    "normal_sinus":    "Normal Sinus Rhythm",
    "sinus_tach":      "Sinus Tachycardia",
    "sinus_brady":     "Sinus Bradycardia",
    "afib":            "Atrial Fibrillation",
    "aflutter":        "Atrial Flutter",
    "svt":             "SVT",
    "inferior_stemi":  "Inferior STEMI / MI",
    "anterior_stemi":  "Anterior STEMI / MI",
    "lateral_stemi":   "Lateral STEMI / MI",
    "lbbb":            "LBBB",
    "rbbb":            "RBBB",
    "avb1":            "1st Degree AV Block",
    "avb2":            "2nd Degree AV Block",
    "avb3":            "3rd Degree AV Block",
    "wpw":             "WPW",
    "st_depression":   "ST Depression (Ischemia)",
    "hyperacute_t":    "Hyperacute T / Subendo Injury",
    "pvc":             "Ventricular Premature Complex",
}

def _ecg_image_path(slug):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "ecg_images", f"{slug}.png")

def _extract_ecg_slug(text):
    m = re.search(r"\[ECG\]\s*([a-z0-9_]+)", text, re.IGNORECASE)
    return m.group(1).lower().strip() if m else None

# ── Capnography helpers ────────────────────────────────────────────────────────
CAPNO_SLUGS = {
    "normal":           "Normal Capnography",
    "bronchospasm":     "Bronchospasm / COPD (Shark-Fin)",
    "hyperventilation": "Hyperventilation",
    "hypoventilation":  "Hypoventilation",
    "esophageal":       "Esophageal Intubation",
    "apnea":            "Apnea",
    "cardiac_arrest":   "Cardiac Arrest — No Perfusion",
    "rosc":             "ROSC — Spontaneous Circulation Restored",
    "rebreathing":      "CO₂ Rebreathing",
    "rosc":             "ROSC — EtCO₂ Rising",
    "rebreathing":      "Rebreathing / Elevated Baseline",
    "cpr":              "CPR in Progress",
}

def _extract_capno_data(text):
    """Return (slug, etco2_int_or_None) from a [CAPNO] slug optional_etco2 tag."""
    m = re.search(r"\[CAPNO\]\s*([a-z0-9_]+)(?:\s+(\d+))?", text, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).lower().strip(), (int(m.group(2)) if m.group(2) else None)


def generate_capno_plot(slug, etco2=None, rr=None):
    """Generate a dynamic capnography waveform and return it as a PNG BytesIO buffer."""
    _DEFS = {
        "normal":           {"etco2": 38, "rr": 14, "shape": "standard"},
        "hypoventilation":  {"etco2": 58, "rr": 8,  "shape": "standard"},
        "hyperventilation": {"etco2": 28, "rr": 28, "shape": "standard"},
        "bronchospasm":     {"etco2": 50, "rr": 16, "shape": "sharkfin"},
        "esophageal":       {"etco2": 30, "rr": 14, "shape": "esophageal"},
        "apnea":            {"etco2": 0,  "rr": 0,  "shape": "flat"},
        "cardiac_arrest":   {"etco2": 10, "rr": 4,  "shape": "standard"},
        "rosc":             {"etco2": 38, "rr": 10, "shape": "standard"},
        "rebreathing":      {"etco2": 52, "rr": 14, "shape": "rebreathing"},
        "cpr":              {"etco2": 18, "rr": 100, "shape": "cpr"},
    }
    d = _DEFS.get(slug, _DEFS["normal"])
    if etco2 is None: etco2 = d["etco2"]
    if rr    is None: rr    = d["rr"]
    shape = d["shape"]

    duration = 15.0
    n = 3000  # 200 samples/sec
    t = np.linspace(0, duration, n)
    co2 = np.zeros(n)

    if shape != "flat" and rr > 0:
        period = 60.0 / max(rr, 1)
        phase  = (t % period) / period
        breath = np.floor(t / period).astype(int)

        if shape == "cpr":
            co2 = np.where(phase < 0.35,
                           etco2 * np.sin(np.pi * phase / 0.35),
                  np.where(phase < 0.55, etco2 * 0.35, 0.0))

        elif shape == "sharkfin":
            co2 = np.where(phase < 0.18, 0.0,
                  np.where(phase < 0.68, etco2 * (phase - 0.18) / 0.50,
                  np.where(phase < 0.80, etco2 * (1.0 - (phase - 0.68) / 0.12),
                  0.0)))

        elif shape == "esophageal":
            amp = np.maximum(0.0, etco2 * (1.0 - breath * 0.28))
            co2 = np.where(phase < 0.18, 0.0,
                  np.where(phase < 0.30, amp * (phase - 0.18) / 0.12,
                  np.where(phase < 0.68, amp,
                  np.where(phase < 0.80, amp * (1.0 - (phase - 0.68) / 0.12),
                  0.0))))

        elif shape == "rebreathing":
            baseline = etco2 * 0.20
            co2 = np.where(phase < 0.18, baseline,
                  np.where(phase < 0.30,
                           baseline + (etco2 - baseline) * (phase - 0.18) / 0.12,
                  np.where(phase < 0.68, etco2,
                  np.where(phase < 0.80,
                           baseline + (etco2 - baseline) * (1.0 - (phase - 0.68) / 0.12),
                  baseline))))

        else:  # standard — normal, hypo, hyper, cardiac_arrest, rosc
            slope = 0.03  # characteristic α-angle upslope on alveolar plateau
            co2 = np.where(phase < 0.18, 0.0,
                  np.where(phase < 0.30, etco2 * (phase - 0.18) / 0.12,
                  np.where(phase < 0.68, etco2 * (1.0 + slope * (phase - 0.30) / 0.38),
                  np.where(phase < 0.80,
                           etco2 * (1.0 + slope) * (1.0 - (phase - 0.68) / 0.12),
                  0.0))))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.plot(t, co2, color="#00FF00", linewidth=2.2)
    for sp in ax.spines.values():
        sp.set_color("#008800")
        sp.set_linewidth(0.8)
    ax.tick_params(colors="#00FF00", labelsize=8)
    ax.set_xlabel("Time (s)", color="#00FF00", fontsize=9)
    ax.set_ylabel("CO₂ (mmHg)", color="#00FF00", fontsize=9)
    ax.set_xlim(0, duration)
    ax.set_ylim(-4, max(float(etco2) * 1.40, 20))
    ax.grid(True, color="#003300", linewidth=0.5, alpha=0.7)

    etco2_str = str(etco2) if etco2 > 0 else "---"
    rr_str    = str(rr)    if rr    > 0 else "---"
    ax.text(0.98, 0.97, f"EtCO₂  {etco2_str} mmHg",
            transform=ax.transAxes, color="#00FF00", fontsize=13,
            ha="right", va="top", fontfamily="monospace", fontweight="bold")
    ax.text(0.98, 0.74, f"RR  {rr_str} /min",
            transform=ax.transAxes, color="#00FF00", fontsize=13,
            ha="right", va="top", fontfamily="monospace", fontweight="bold")

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#000000", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def _get_db_connection():
    """
    Return (conn, is_postgres).
    Uses PostgreSQL when DATABASE_URL secret is present (Streamlit Cloud),
    falls back to local SQLite otherwise.
    Always ensures the call_metrics table exists.
    """
    db_url = _secret("DATABASE_URL", "")

    if db_url:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        # Each DDL committed separately — ALTER TABLE failure aborts the transaction
        # in PostgreSQL; committing between statements keeps each one independent.
        cur.execute("""CREATE TABLE IF NOT EXISTS call_metrics (
            id SERIAL PRIMARY KEY, timestamp TEXT NOT NULL,
            mode TEXT, acuity TEXT, category TEXT, complaint TEXT,
            had_hazard INTEGER, used_refusal INTEGER,
            narrative TEXT,
            score INTEGER, pass_fail TEXT, transcript TEXT,
            student TEXT DEFAULT 'Anonymous')""")
        conn.commit()
        try:
            cur.execute("ALTER TABLE call_metrics ADD COLUMN student TEXT DEFAULT 'Anonymous'")
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            cur.execute("ALTER TABLE call_metrics ADD COLUMN narrative TEXT")
            conn.commit()
        except Exception:
            conn.rollback()   # column already exists
        cur.execute("""CREATE TABLE IF NOT EXISTS live_sessions (
            session_id TEXT PRIMARY KEY, student TEXT, mode TEXT,
            messages TEXT, vitals TEXT, timeline TEXT,
            acuity TEXT, category TEXT, complaint TEXT,
            active_hazard TEXT, scene_cleared INTEGER, hazard_warned INTEGER,
            updated_at TEXT)""")
        conn.commit()
        cur.close()
        return conn, True
    else:
        conn = sqlite3.connect("simulation_data.db")
        conn.execute("""CREATE TABLE IF NOT EXISTS call_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            mode TEXT, acuity TEXT, category TEXT, complaint TEXT,
            had_hazard INTEGER, used_refusal INTEGER,
            score INTEGER, pass_fail TEXT, transcript TEXT,
            student TEXT DEFAULT 'Anonymous')""")
        try:
            conn.execute("ALTER TABLE call_metrics ADD COLUMN student TEXT DEFAULT 'Anonymous'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE call_metrics ADD COLUMN narrative TEXT")
        except Exception:
            pass
        conn.execute("""CREATE TABLE IF NOT EXISTS live_sessions (
            session_id TEXT PRIMARY KEY, student TEXT, mode TEXT,
            messages TEXT, vitals TEXT, timeline TEXT,
            acuity TEXT, category TEXT, complaint TEXT,
            active_hazard TEXT, scene_cleared INTEGER, hazard_warned INTEGER,
            updated_at TEXT)""")
        conn.commit()
        return conn, False


def log_call_metrics(mode, acuity, category, complaint, response_text, had_hazard, used_refusal, student="Anonymous"):
    """Write one completed simulation record to the analytics database."""
    try:
        # --- Extract score and pass/fail from the debrief text ---
        score_m = re.search(r'(?:\[SCORE\]|SCORE\b)[^0-9\n]{0,30}(\d{1,3})\s*/\s*100', response_text, re.IGNORECASE)
        score   = int(score_m.group(1)) if score_m else None

        pf_m    = re.search(r'(?:\[RESULT\]|RESULT\b)[^A-Za-z\n]{0,20}(PASS|FAIL)', response_text, re.IGNORECASE)
        pass_fail = pf_m.group(1).upper() if pf_m else None

        # --- Extract model PCR narrative ---
        narr_m  = re.search(r'\[NARRATIVE\]\s*(.*?)(?=\[(?:SCORE|RESULT|LOG|VITAL|DEBRIEF)\]|$)',
                            response_text, re.DOTALL | re.IGNORECASE)
        narrative = narr_m.group(1).strip() if narr_m else None

        # --- Build readable transcript from message history ---
        lines = []
        for msg in st.session_state.get("messages", []):
            if isinstance(msg, HumanMessage):
                lines.append(f"STUDENT: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"MEDSIM:  {msg.content}")
        transcript = "\n\n".join(lines)

        timestamp = datetime.now().isoformat(timespec="seconds")

        conn, is_postgres = _get_db_connection()
        ph = "%s" if is_postgres else "?"
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO call_metrics
                (timestamp, mode, acuity, category, complaint,
                 had_hazard, used_refusal, score, pass_fail, transcript, student, narrative)
               VALUES ({",".join([ph]*12)})""",
            (timestamp, mode, acuity, category, complaint,
             int(had_hazard), int(used_refusal), score, pass_fail, transcript, student, narrative),
        )
        conn.commit()
        cur.close()
        conn.close()
        # Store success so it survives the st.rerun()
        st.session_state["_db_log_error"] = None

    except Exception as e:
        # Store error in session state — st.warning() before st.rerun() is invisible
        st.session_state["_db_log_error"] = str(e)

# --- 5. SESSION PERSISTENCE ---
def _serialize_messages(messages):
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):  out.append({"t": "h", "c": m.content})
        elif isinstance(m, AIMessage):   out.append({"t": "a", "c": m.content})
        elif isinstance(m, SystemMessage): out.append({"t": "s", "c": m.content})
    return json.dumps(out)

def _deserialize_messages(s):
    msgs = []
    for item in json.loads(s):
        if item["t"] == "h": msgs.append(HumanMessage(content=item["c"]))
        elif item["t"] == "a": msgs.append(AIMessage(content=item["c"]))
        elif item["t"] == "s": msgs.append(SystemMessage(content=item["c"]))
    return msgs

def _save_live_session():
    """Checkpoint active call state to DB after every LLM response."""
    sid = st.session_state.get("session_id")
    if not sid or not st.session_state.get("started") or st.session_state.get("sim_finished"):
        return
    try:
        conn, is_pg = _get_db_connection()
        ph = "%s" if is_pg else "?"
        hazard = json.dumps(st.session_state.get("active_hazard")) if st.session_state.get("active_hazard") else None
        vals = (
            sid,
            st.session_state.get("student_name", "Anonymous"),
            st.session_state.get("mode", ""),
            _serialize_messages(st.session_state.get("messages", [])),
            json.dumps(st.session_state.get("vitals", {})),
            json.dumps(st.session_state.get("timeline", [])),
            st.session_state.get("current_acuity", ""),
            st.session_state.get("current_category", ""),
            st.session_state.get("current_complaint", ""),
            hazard,
            int(st.session_state.get("scene_cleared", True)),
            int(st.session_state.get("hazard_warned", False)),
            datetime.now().isoformat(timespec="seconds"),
        )
        cur = conn.cursor()
        if is_pg:
            cur.execute(f"""INSERT INTO live_sessions
                (session_id,student,mode,messages,vitals,timeline,acuity,category,complaint,
                 active_hazard,scene_cleared,hazard_warned,updated_at)
                VALUES ({",".join([ph]*13)})
                ON CONFLICT (session_id) DO UPDATE SET
                messages=EXCLUDED.messages, vitals=EXCLUDED.vitals, timeline=EXCLUDED.timeline,
                active_hazard=EXCLUDED.active_hazard, scene_cleared=EXCLUDED.scene_cleared,
                hazard_warned=EXCLUDED.hazard_warned, updated_at=EXCLUDED.updated_at""", vals)
        else:
            cur.execute(f"""INSERT OR REPLACE INTO live_sessions
                (session_id,student,mode,messages,vitals,timeline,acuity,category,complaint,
                 active_hazard,scene_cleared,hazard_warned,updated_at)
                VALUES ({",".join([ph]*13)})""", vals)
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass  # Never let checkpointing crash the sim

def _load_live_session(sid):
    """Restore a checkpointed session. Returns dict or None."""
    try:
        conn, is_pg = _get_db_connection()
        ph = "%s" if is_pg else "?"
        cur = conn.cursor()
        cur.execute(f"""SELECT student,mode,messages,vitals,timeline,acuity,category,complaint,
            active_hazard,scene_cleared,hazard_warned FROM live_sessions WHERE session_id={ph}""", (sid,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return None
        return {
            "student": row[0], "mode": row[1],
            "messages": _deserialize_messages(row[2]),
            "vitals": json.loads(row[3]),
            "timeline": json.loads(row[4]),
            "acuity": row[5], "category": row[6], "complaint": row[7],
            "active_hazard": json.loads(row[8]) if row[8] else None,
            "scene_cleared": bool(row[9]),
            "hazard_warned": bool(row[10]),
        }
    except Exception:
        return None

def _clear_live_session(sid):
    """Delete checkpoint when call ends cleanly."""
    try:
        conn, is_pg = _get_db_connection()
        ph = "%s" if is_pg else "?"
        cur = conn.cursor()
        cur.execute(f"DELETE FROM live_sessions WHERE session_id={ph}", (sid,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass

# --- 6. INITIAL SESSION STATE + AUTO-RESTORE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vitals" not in st.session_state: st.session_state.vitals = {"HR": "--", "BP": "--", "RR": "--", "SPO2": "--", "BGL": "--", "TEMP": "--"}
if "timeline" not in st.session_state: st.session_state.timeline = []
if "started" not in st.session_state: st.session_state.started = False
if "sim_finished" not in st.session_state: st.session_state.sim_finished = False

# Auto-restore a dropped session from the URL session ID
if not st.session_state.started:
    _sid = st.query_params.get("sid")
    if _sid:
        _saved = _load_live_session(_sid)
        if _saved:
            st.session_state.session_id     = _sid
            st.session_state.started        = True
            st.session_state.sim_finished   = False
            st.session_state.mode           = _saved["mode"]
            st.session_state.student_name   = _saved["student"]
            st.session_state.messages       = _saved["messages"]
            st.session_state.vitals         = _saved["vitals"]
            st.session_state.timeline       = _saved["timeline"]
            st.session_state.current_acuity    = _saved["acuity"]
            st.session_state.current_category  = _saved["category"]
            st.session_state.current_complaint = _saved["complaint"]
            st.session_state.active_hazard  = _saved["active_hazard"]
            st.session_state.scene_cleared  = _saved["scene_cleared"]
            st.session_state.hazard_warned  = _saved["hazard_warned"]
            st.session_state.start_time     = datetime.now()  # reconnect — real origin lost, reset clock

# --- SIDEBAR UPDATES ---
with st.sidebar:
    st.title("📟 Control Center")
    standard = st.radio("Standard", ["PWC Protocols", "NREMT"], horizontal=True,
                        disabled=st.session_state.started,
                        help="PWC: evaluated against local Prince William County protocols. NREMT: evaluated against National EMS Education Standards.")
    st.session_state.standard = "NREMT" if standard == "NREMT" else "PWC"

    cat = st.selectbox("Category", ["Medical", "Trauma", "Pediatric", "Cardiac"], disabled=st.session_state.started)
    acuity = st.select_slider("Acuity", options=["Easy", "Moderate", "Hard", "Critical"], disabled=st.session_state.started)
    mode = st.radio("Protocol Level", ["BLS", "ALS"], disabled=st.session_state.started)
    
    st.divider()
    st.subheader("👤 Student")
    student_name = st.text_input("Your Name:", placeholder="Leave blank to stay anonymous", disabled=st.session_state.started)

    st.divider()
    # THE CUSTOM DISPATCH OVERRIDE
    st.subheader("🎯 Custom Dispatch")
    custom_scenario = st.text_input("Override Scenario:", placeholder="e.g., Snake bite to the hand", disabled=st.session_state.started)
    st.caption("Leave blank to use the random pool.")



# --- 7. MAIN UI ---
if not st.session_state.started:

    st.info("⚠️ MedSim Academy uses AI to help you practice clinical decision‑making. The scenarios are fictional, and the system scores you using the protocols installed at the time—these might not match your agency's latest updates, and the AI isn't perfect. When treating real patients, always rely on your medical director and your local protocols.")

    with st.expander("📖 READ FIRST: Standard Operating Procedure", expanded=False):
        st.markdown("""
        ### **1. Starting the Call**
        Select the **Category**, **Acuity**, and **Protocol Level** in the sidebar, then click **🚀 START CALL**. You can also enter your own dispatch in the **Override Scenario** field to practice a specific call.
        ### **2. Initial Response**
        Your first response must be **"Scene safe, BSI"**. This will trigger the scene and environment description. If the scene is not safe, call for the appropriate resources before proceeding.
        ### **3. Interaction**
        Treat the chat as your voice and hands. You can **Assess** (*"I check skin"*, *"obtain vitals"*, *"obtain 12-lead"*, *"get capno"*), **Intervene** (*"I start an IV"*), or **Communicate** (*"I talk to family"*).
        ### **4. Monitor & Timeline**
        The monitor updates vitals and the timeline logs your actions in real-time.
        ### **5. Ending the Call**
        The simulation concludes with a **[DEBRIEF]** when care is transferred, the patient expires, a refusal is obtained, or the provider is injured.
        ### **6. The Debrief**
        A clinical performance review will analyze your actions against the standard you selected.
        ### **7. Special Situations**
        - **Language barrier:** If the patient speaks only Spanish, type your assessment in Spanish or say *"contact Language Line"* to get an interpreter on the line.
        - **Patient refusing care:** If the patient refuses transport but you believe they lack decision-making capacity, say *"I'm requesting an ECO"* to contact the magistrate for an Emergency Custody Order.
        """)

    if st.button("🚀 START CALL", type="primary", use_container_width=True):
        # 1. EXPANDED CALIBRATED MASTER POOL
        pool = {
            "Medical": {
                "Easy": [
                    "Minor constipation for 3 days", "Seasonal flu symptoms with low-grade fever", 
                    "Mild nausea without vomiting", "Stomach pain secondary to mild indigestion",
                    "Isolated productive cough, query bronchitis", "General malaise and fatigue", 
                    "Minor heat cramps after working in the yard", "Mild toothache with stable vitals"
                ],
                "Moderate": [
                    "Diabetic Emergency (hypoglycemic, alert but confused)", "Asthma attack with mild wheezing", 
                    "Sepsis warning signs (fever, altered mental status, high heart rate)", "Suspected Carbon Monoxide exposure",
                    "Allergic reaction (hives and itching, no respiratory distress)", "Severe migraine with photophobia",
                    "Dehydration from gastroenteritis (frequent vomiting)", "Acute localized abdominal pain (suspected appendicitis)"
                ],
                "Hard": [
                    "Status Epilepticus (Active, continuous grand mal seizure)", "Anaphylaxis with severe airway swelling and stridor", 
                    "Active Stroke (CVA) with unilateral facial droop and expressive aphasia", "Opioid Overdose (completely unresponsive, apneic)",
                    "Severe heat stroke with altered mental status and hot, dry skin", "Suspected Meningitis with high fever and nuchal rigidity",
                    "Severe respiratory distress secondary to acute pulmonary edema", "Tricyclic antidepressant overdose with widening QRS complex"
                ],
                "Critical": [
                    "Status Epilepticus (Active, continuous grand mal seizure)", "Anaphylaxis with severe airway swelling and stridor", 
                    "Active Stroke (CVA) with unilateral facial droop and expressive aphasia", "Opioid Overdose (completely unresponsive, apneic)",
                    "Severe heat stroke with altered mental status and hot, dry skin", "Suspected Meningitis with high fever and nuchal rigidity",
                    "Severe respiratory distress secondary to acute pulmonary edema", "Tricyclic antidepressant overdose with widening QRS complex"
                ]
            },
            "Trauma": {
                "Easy": [
                    "Stubbed toe with minor swelling and no deformity", "Minor road rash from a slow-speed bicycle fall", 
                    "Epistaxis (isolated, controlled nosebleed)", "Isolated finger dislocation without skin tear",
                    "Minor laceration to the forearm with controlled bleeding", "Superficial chemical burn from household bleach",
                    "Ankle sprain after stepping off a curb", "Contusion to the shin from kicking a coffee table"
                ],
                "Moderate": [
                    "Fall from standing with a shortened, externally rotated leg (suspected hip fracture)", 
                    "Stabbing (isolated, superficial abdominal laceration with slow bleeding)", 
                    "Isolated closed mid-shaft radius/ulna fracture with intact distal pulses", 
                    "Large scalp laceration with moderate active bleeding from a fall",
                    "Blunt chest trauma from a steering wheel, query rib fractures", "Second-degree burns to the lower leg from boiling water",
                    "Dog bite to the forearm with puncture wounds", "High-velocity paint gun injury to the index finger"
                ],
                "Hard": [
                    "Fall from a ladder (approx. 20 feet) onto concrete", "Gunshot wound (GSW) to the right upper quadrant of the abdomen", 
                    "Motor Vehicle Collision (MVC) with heavy mechanical entrapment and altered mental status", 
                    "Partial traumatic amputation of the right hand at the wrist",
                    "Tension Pneumothorax with absent breath sounds and tracheal deviation", 
                    "Open Pneumothorax (Sucking chest wound) from a penetrating chest trauma",
                    "Unstable Pelvic Fracture following a motorcycle accident", "Flail Chest with paradoxical chest wall movement"
                ],
                "Critical": [
                    "Fall from a ladder (approx. 20 feet) onto concrete", "Gunshot wound (GSW) to the right upper quadrant of the abdomen", 
                    "Motor Vehicle Collision (MVC) with heavy mechanical entrapment and altered mental status", 
                    "Partial traumatic amputation of the right hand at the wrist",
                    "Tension Pneumothorax with absent breath sounds and tracheal deviation", 
                    "Open Pneumothorax (Sucking chest wound) from a penetrating chest trauma",
                    "Unstable Pelvic Fracture following a motorcycle accident", "Flail Chest with paradoxical chest wall movement"
                ]
            },
            "Pediatric": {
                "Easy": [
                    "7-year-old with a minor hot water scald (1st-degree burn on forearm)", 
                    "18-month-old with a mild croup cough, alert and playful",
                    "5-year-old with a superficial laceration from a playground slide",
                    "3-year-old with isolated ear pain and low-grade fever",
                    "12-year-old with mild heat exhaustion following a soccer match"
                ],
                "Moderate": [
                    "6-month-old post-ictal following a brief febrile seizure prior to arrival", 
                    "3-year-old who accidentally ingested a small amount of household cleaning spray",
                    "8-year-old with moderate asthma exacerbation, speaking in short sentences",
                    "2-year-old with a foreign body obstruction (bead lodged in the right nostril)",
                    "14-year-old with a closed femur fracture following a bicycle accident"
                ],
                "Hard": [
                    "2-year-old actively choking on a grape with a completely silent airway", 
                    "4-year-old pulled from a swimming pool, unresponsive with shallow agonal respirations", 
                    "10-year-old pedestrian vs car, thrown 15 feet onto asphalt, altered mental status",
                    "5-year-old in severe anaphylaxis from a peanut allergy with marked facial swelling and stridor",
                    "9-month-old lethargic with a bulging fontanelle and high fever",
                    "6-year-old with full-thickness burns to the chest and abdomen from a campfire"
                ],
                "Critical": [
                    "2-year-old actively choking on a grape with a completely silent airway", 
                    "4-year-old pulled from a swimming pool, unresponsive with shallow agonal respirations", 
                    "10-year-old pedestrian vs car, thrown 15 feet onto asphalt, altered mental status",
                    "5-year-old in severe anaphylaxis from a peanut allergy with marked facial swelling and stridor",
                    "9-month-old lethargic with a bulging fontanelle and high fever",
                    "6-year-old with full-thickness burns to the chest and abdomen from a campfire"
                ]
            },
            "Cardiac": {
                "Easy": [
                    "Palpitations (anxiety-induced, normal sinus rhythm with stable vitals)", 
                    "Asymptomatic mild hypertension (BP 160/95, no headache or vision changes)",
                    "Mild dizziness upon standing, resolved with a glass of water",
                    "Isolated episode of lightheadedness, history of benign vertigo"
                ],
                "Moderate": [
                    "Ventricular Tachycardia (stable with a palpable, strong pulse)", 
                    "Symptomatic Bradycardia (heart rate 42, feeling lightheaded and fatigued)", 
                    "Congestive Heart Failure (CHF) with mild respiratory distress and bilateral crackles",
                    "Atrial Fibrillation with rapid ventricular response (RVR), heart rate 140, mild chest flutters"
                ],
                "Hard": [
                    "Chest Pain indicative of an acute STEMI (crushing, radiating substernal pain)", 
                    "Cardiac Arrest (V-Fib witnessed by family, bystander CPR in progress)", 
                    "Cardiogenic Shock (profound hypotension, cool, clammy skin post-myocardial infarction)", 
                    "Aortic Dissection (sudden, severe, tearing pain radiating to the shoulder blades)",
                    "Unstable Supraventricular Tachycardia (SVT) with poor perfusion, heart rate 210",
                    "Third-Degree Heart Block with a ventricular escape rate of 28, altered mental status"
                ],
                "Critical": [
                    "Chest Pain indicative of an acute STEMI (crushing, radiating substernal pain)", 
                    "Cardiac Arrest (V-Fib witnessed by family, bystander CPR in progress)", 
                    "Cardiogenic Shock (profound hypotension, cool, clammy skin post-myocardial infarction)", 
                    "Aortic Dissection (sudden, severe, tearing pain radiating to the shoulder blades)",
                    "Unstable Supraventricular Tachycardia (SVT) with poor perfusion, heart rate 210",
                    "Third-Degree Heart Block with a ventricular escape rate of 28, altered mental status"
                ]
            }
        }

        HAZARD_POOL = [
            {"type": "electrical", "fix": "power company"},
            {"type": "animal", "fix": "animal control"},
            {"type": "Chemical", "fix": "hazmat"},
            {"type": "violence", "fix": "police"}
        ]
        
        # 2. CALIBRATED SELECTION LOGIC
        if custom_scenario.strip():
            complaint = custom_scenario
            st.session_state.current_complaint = custom_scenario
            dispatch_instruction = f"a {acuity} {cat} emergency: {complaint}"
        else:
            category_dict = pool.get(cat, pool["Medical"])
            difficulty_list = category_dict.get(acuity, category_dict["Moderate"])
            choice = random.choice(difficulty_list)
            dispatch_instruction = f"a {acuity}-acuity case: {choice}"
            st.session_state.current_complaint = choice
        
        st.session_state.current_acuity = acuity
        st.session_state.current_category = cat

        if random.random() < 0.10:
            st.session_state.active_hazard = random.choice(HAZARD_POOL)
            st.session_state.scene_cleared = False
            st.session_state.hazard_warned = False
            st.session_state.hazard_phase = 1
        else:
            st.session_state.active_hazard = None
            st.session_state.scene_cleared = True

        st.session_state.started = True
        st.session_state.student_name = student_name.strip() or "Anonymous"
        st.session_state.mode = mode
        st.session_state.start_time = datetime.now()
        st.session_state.sim_finished = False
        st.session_state.timeline = ["Simulation started."]
        st.session_state.session_id = uuid.uuid4().hex[:10]
        st.query_params["sid"] = st.session_state.session_id
        
        # 3. DISPATCH
        active_standard = st.session_state.get("standard", "PWC")
        scope_ref = load_scope_reference(active_standard)
        scope_label = "NATIONAL EMS EDUCATION STANDARDS (NREMT)" if active_standard == "NREMT" else "LOCAL PROTOCOL REFERENCE — PWC"
        sys_p = f"!!! {mode} MODE !!!\nACUITY: {acuity}\nCATEGORY: {cat}\nEVALUATION STANDARD: {active_standard}\n{_secret('SYSTEM_PROMPT_CONTENT', '').replace('{mode}', mode)}"
        if scope_ref:
            sys_p += f"\n\n[{scope_label} — AUTHORITATIVE FOR ALL SCOPE DETERMINATIONS]\n{scope_ref}"
        if custom_scenario.strip():
            sys_p += (
                "\n\n[CUSTOM SCENARIO OVERRIDE — PARAMETER LOCK ACTIVE]\n"
                "The instructor has specified exact dispatch parameters. "
                "SUSPEND all Procedural Variation rules. "
                "You MUST preserve the patient's exact age, sex, and chief complaint as dispatched — do NOT change any of these. "
                "Only the street address, apartment details, and incidental scene environment details may vary to make the dispatch realistic. "
                "Example: if dispatched as '75 YOM chest pain', the patient must be a 75-year-old male with chest pain."
            )
        dispatch_msg = f"Dispatch {dispatch_instruction}."
        st.session_state.messages = [
            SystemMessage(content=sys_p),
            HumanMessage(content=dispatch_msg),
        ]
        
        with st.spinner("Dispatching..."):
            dispatch_resp = None
            try:
                dispatch_resp = _invoke_with_retry(st.session_state.messages)
            except Exception:
                st.error("Could not connect to simulation engine. Please try again.")
                for key in ["started", "mode", "start_time", "sim_finished", "timeline",
                            "active_hazard", "scene_cleared", "hazard_warned", "hazard_phase",
                            "current_complaint", "current_acuity", "current_category"]:
                    st.session_state.pop(key, None)
                st.rerun()
            if dispatch_resp:
                process_medsim_turn(dispatch_resp.content)
                st.session_state.messages.append(AIMessage(content=dispatch_resp.content))
                _save_live_session()
        st.rerun()

else:
    col_chat, _, col_data = st.columns([2, 0.1, 1.2])

    # col_chat rendered first so it sits above col_data when columns stack on mobile
    with col_chat:
        # Hint pills — top of chat column, above message history
        if not st.session_state.sim_finished:
            st.markdown(
                '<div style="display:flex;gap:10px;margin-bottom:10px;">'
                '<span style="background:#1a472a;color:#a3d9a5;padding:4px 12px;border-radius:20px;font-size:0.78rem;">🟢 Begin: "Scene safe, BSI"</span>'
                '<span style="background:#2c2c54;color:#a0a0d0;padding:4px 12px;border-radius:20px;font-size:0.78rem;">🏁 End: Transfer care</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        prev_vitals_requested = False
        for msg in st.session_state.messages:
            raw_content = msg.content if isinstance(msg.content, str) else ""
            if isinstance(msg, HumanMessage):
                prev_vitals_requested = bool(_VITALS_REQUEST_KW.search(raw_content))
            if isinstance(msg, (AIMessage, HumanMessage)) and "Dispatch" not in raw_content:
                clean_text  = re.sub(r"--- LOCAL PROTOCOL REFERENCE ---.*--- STUDENT ACTION ---", "", raw_content, flags=re.DOTALL)
                ecg_slug              = _extract_ecg_slug(clean_text)
                capno_slug, capno_etco2 = _extract_capno_data(clean_text)
                show_vitals = prev_vitals_requested if isinstance(msg, AIMessage) else True
                clean_text  = clean_for_display(clean_text, show_vitals_inline=show_vitals)
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                with st.chat_message(role, avatar="🚑" if role=="assistant" else "🩺"):
                    st.markdown(clean_text)
                    # Model PCR narrative — shown at end of debrief message
                    if isinstance(msg, AIMessage) and "[DEBRIEF]" in raw_content:
                        narr_m = re.search(r'\[NARRATIVE\]\s*(.*?)(?=\[(?:SCORE|RESULT|LOG|VITAL)\]|$)',
                                           raw_content, re.DOTALL | re.IGNORECASE)
                        if narr_m:
                            with st.expander("📋 Model PCR Narrative", expanded=False):
                                st.caption("⚠️ This is a model narrative reflecting ideal care — not the student's actual actions.")
                                st.markdown(narr_m.group(1).strip())
                    if ecg_slug and ecg_slug in ECG_SLUGS:
                        img_path = _ecg_image_path(ecg_slug)
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"12-Lead ECG — {ECG_SLUGS[ecg_slug]}", use_container_width=True)
                        else:
                            st.caption(f"⚠ ECG image not found: {img_path}")
                    if capno_slug and capno_slug in CAPNO_SLUGS:
                        buf = generate_capno_plot(capno_slug, etco2=capno_etco2)
                        st.image(buf, caption=f"Capnography — {CAPNO_SLUGS[capno_slug]}", use_container_width=True)

        # Show any database logging error that survived the rerun
        if st.session_state.get("_db_log_error"):
            st.error(f"⚠️ Analytics logging error: {st.session_state['_db_log_error']}")

        # Student Input
        u_input = None
        if not st.session_state.sim_finished:
            typed = st.chat_input("Enter action...")
            if typed:
                u_input = typed

        if u_input and not st.session_state.sim_finished:
            if True:
                with st.chat_message("user"): st.markdown(u_input)
                

                # --- TWO-STEP SCENE SAFETY FIREWALL ---
                hazard_active = (
                    hasattr(st.session_state, 'active_hazard')
                    and st.session_state.active_hazard
                    and not st.session_state.scene_cleared
                )
                if hazard_active:
                    u_input_lower = u_input.lower()
                    hazard = st.session_state.active_hazard

                    staging_keywords = ["stage", "staging", "back up", "pull away", "retreat", "wait", "stand by", "distance"]
                    resource_keywords = [
                        hazard["fix"],
                        "pd", "police", "cop", "law enforcement", "sheriff",
                        "911", "dispatch", "resources", "assistance",
                        "animal control", "animal",
                        "power company", "utility company", "utilities", "electric company", "power", "electric", "lineman",
                        "hazmat", "haz-mat", "decon", "chemical",
                        "fire dept", "fire department", "fire", "rescue", "engine", "ladder",
                    ]

                    user_staged = any(word in u_input_lower for word in staging_keywords)
                    user_called_help = any(word in u_input_lower for word in resource_keywords)

                    if not st.session_state.hazard_warned:
                        st.session_state.hazard_warned = True
                        scene_directive = (
                            f"--- CRITICAL SIMULATION INSTRUCTION ---\n"
                            f"The student just declared Scene Safe/BSI. You must now generate the [SCENE] "
                            f"presentation. An active environmental threat is present. Use your TOML rules to "
                            f"organically synthesize a hazard based on this raw parameter token: {hazard['type']}. "
                            f"It MUST logically harmonize with the dispatch complaint in the history above. "
                            f"CRITICAL: The patient's age, sex, and chief complaint MUST exactly match the dispatch — do NOT change them. "
                            f"Do not print the token name. Conclude by asking how they wish to proceed."
                        )
                        scene_history = st.session_state.messages + [HumanMessage(content=scene_directive)]
                        feedback = "[SCENE] Scene safety assessment in progress. Stand by."
                        try:
                            resp = _invoke_with_retry(scene_history)
                            feedback = resp.content
                        except Exception:
                            pass  # keep fallback feedback; student can re-send
                        st.session_state.messages.append(HumanMessage(content=u_input))
                        st.session_state.messages.append(AIMessage(content=feedback))
                        st.session_state.timeline.append("Arrived on scene; hazard encountered.")
                        st.rerun()
                    else:
                        if user_staged or user_called_help:
                            st.session_state.scene_cleared = True
                            feedback = "[SCENE] Copy that. Initializing staging protocols and routing specialized resources. The threat has been mitigated and the area is declared safe. You may now approach the patient."
                            clear_directive = (
                                "--- CRITICAL SIMULATION UPDATE ---\n"
                                "The active environmental hazard has been completely resolved. The scene is now "
                                "100% safe. You must completely drop the hazard from the narrative. Do not mention "
                                "any scene threats again. The student is now safely at the patient's side. "
                                "Transition immediately to STATE 4: ASSESSMENT & TREATMENT."
                            )
                            st.session_state.messages.append(HumanMessage(content=u_input))
                            st.session_state.messages.append(AIMessage(content=feedback))
                            st.session_state.messages.append(SystemMessage(content=clear_directive))
                            st.session_state.timeline.append(f"Scene safely mitigated via: '{u_input}'")
                            st.rerun()
                        else:
                            if st.session_state.get("hazard_phase", 1) == 1:
                                # PHASE 1 — in-world block, one chance remaining
                                phase1_directive = (
                                    f"--- SCENE SAFETY PHASE 1 ---\n"
                                    f"The student attempted to approach the patient without addressing the active {hazard['type']} hazard. "
                                    f"Output ONLY vivid in-world narrative of the hazard physically blocking their approach. "
                                    f"NO system error messages, NO meta-commentary, NO labels like 'CRITICAL SAFETY ERROR'. "
                                    f"Use the PHASE 1 blocking template from your HAZARD CONSEQUENCE PROTOCOL. "
                                    f"End with the hazard still active and the student unable to reach the patient. Do NOT debrief."
                                )
                                temp_history = st.session_state.messages + [
                                    HumanMessage(content=u_input),
                                    SystemMessage(content=phase1_directive),
                                ]
                                try:
                                    resp = _invoke_with_retry(temp_history)
                                    feedback = resp.content
                                except Exception:
                                    feedback = f"[SCENE] The {hazard['type']} hazard surges forward as you step toward the patient, cutting off your path. You are forced back. How do you wish to proceed?"
                                st.session_state.hazard_phase = 2
                                st.session_state.messages.append(HumanMessage(content=u_input))
                                st.session_state.messages.append(AIMessage(content=feedback))
                                st.session_state.timeline.append("Scene safety warning: patient contact attempted with active hazard.")
                                st.rerun()
                            else:
                                # PHASE 2 — consequence + DEBRIEF
                                phase2_directive = (
                                    f"--- SCENE SAFETY PHASE 2 — MANDATORY CONSEQUENCE ---\n"
                                    f"The student has now attempted to reach the patient a second time without resolving the {hazard['type']} hazard. "
                                    f"Execute the PHASE 2 consequence immediately using the appropriate CONSEQUENCE TEMPLATE from your HAZARD CONSEQUENCE PROTOCOL. "
                                    f"Narrate the consequence in 2-3 vivid sentences, then output the full [DEBRIEF] block with SCORE: 0/100 and RESULT: FAIL. "
                                    f"Do NOT allow patient contact. Do NOT continue the simulation."
                                )
                                temp_history = st.session_state.messages + [
                                    HumanMessage(content=u_input),
                                    SystemMessage(content=phase2_directive),
                                ]
                                try:
                                    resp = _invoke_with_retry(temp_history)
                                    feedback = resp.content
                                except Exception:
                                    feedback = (
                                        f"[SCENE] The {hazard['type']} hazard reaches you before you can get to the patient. You are incapacitated and cannot continue care.\n"
                                        f"[DEBRIEF]\n1. CLINICAL OUTCOME: Provider incapacitated due to scene safety failure.\n"
                                        f"2. PROTOCOL ADHERENCE: CRITICAL VIOLATION — Scene safety was not established before patient contact.\n"
                                        f"3. SCORE: 0/100\n4. RESULT: FAIL\n[SCORE] 0/100\n[RESULT] FAIL"
                                    )
                                st.session_state.messages.append(HumanMessage(content=u_input))
                                st.session_state.messages.append(AIMessage(content=feedback))
                                st.session_state.timeline.append("CRITICAL VIOLATION: Scene safety failure — provider incapacitated.")
                                process_medsim_turn(feedback)
                                if "[DEBRIEF]" in feedback:
                                    st.session_state.sim_finished = True
                                    _clear_live_session(st.session_state.get("session_id", ""))
                                    log_call_metrics(
                                        mode=st.session_state.mode,
                                        acuity=st.session_state.get('current_acuity', 'Moderate'),
                                        category=st.session_state.get('current_category', 'Medical'),
                                        complaint=st.session_state.get('current_complaint', 'Random Pool Run'),
                                        response_text=feedback,
                                        had_hazard=True,
                                        used_refusal=False,
                                        student=st.session_state.get('student_name', 'Anonymous'),
                                    )
                                st.rerun()
                # --- END OF SCENE SAFETY FIREWALL ---
                else:
                    with st.spinner("🔍 Consulting Protocols..."):
                        context = get_protocol_context(u_input)

                    _HANDOFF_PHRASES = [
                        "transferring care", "transfer care", "transfer of care",
                        "handing off", "hand off", "handoff",
                        "we're at the hospital", "we are at the hospital", "arrived at the hospital",
                        "turning patient over", "turning over care", "turning over the patient",
                        "i'm handing", "i am handing",
                        "als is on scene", "als on scene", "als has arrived",
                        "patient refused", "patient is refusing", "obtain a refusal",
                        "signing the refusal", "sign the refusal", "signed the refusal",
                        "patient signing", "refusal form",
                    ]
                    _u_lower = u_input.lower()
                    _is_handoff = any(p in _u_lower for p in _HANDOFF_PHRASES)

                    firewall = f"--- MANDATORY: STUDENT IS A {st.session_state.mode} PROVIDER ---\n"
                    if _is_handoff:
                        firewall += "--- MANDATORY DEBRIEF TRIGGER: The student has completed patient hand-off or transport. You MUST output [DEBRIEF] in this response. Do not ask follow-up questions. ---\n"
                    enriched = f"{firewall}--- LOCAL PROTOCOL REFERENCE ---\n{context}\n\n--- STUDENT ACTION ---\n{u_input}"
                    temp_history = st.session_state.messages + [HumanMessage(content=enriched)]

                    with st.spinner("Responding..."):
                        resp = None
                        try:
                            resp = _invoke_with_retry(temp_history)
                        except Exception:
                            st.error("⚠️ The AI engine is temporarily unavailable (Google API overloaded). Your progress is saved — please re-send your last message in a moment.")
                            st.stop()

                    st.session_state.messages.append(HumanMessage(content=u_input))
                    st.session_state.messages.append(AIMessage(content=resp.content))
                    process_medsim_turn(resp.content)

                    if "[DEBRIEF]" in resp.content:
                        st.session_state.sim_finished = True
                        _clear_live_session(st.session_state.get("session_id", ""))
                        had_hz = hasattr(st.session_state, 'active_hazard') and st.session_state.active_hazard is not None
                        used_ref = "[REFUSAL]" in resp.content or "signed refusal" in u_input.lower()
                        log_call_metrics(
                            mode=st.session_state.mode,
                            acuity=st.session_state.get('current_acuity', 'Moderate'),
                            category=st.session_state.get('current_category', 'Medical'),
                            complaint=st.session_state.get('current_complaint', 'Random Pool Run'),
                            response_text=resp.content,
                            had_hazard=had_hz,
                            used_refusal=used_ref,
                            student=st.session_state.get('student_name', 'Anonymous'),
                        )
                    else:
                        _save_live_session()
                    st.rerun()

    # Monitor panel — rendered after col_chat so it stacks below chat on mobile
    with col_data:
        st.subheader("💓 Patient Monitor")
        c1, c2 = st.columns(2)
        v_items = list(st.session_state.vitals.items())
        for i, (k, v) in enumerate(v_items):
            target_col = c1 if i % 2 == 0 else c2
            target_col.markdown(
                f'<div class="vital-card"><span class="vital-label">{k}</span><br>'
                f'<span class="vital-value">{v}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        with st.expander("🕒 Timeline", expanded=True):
            for entry in reversed(st.session_state.timeline):
                st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)