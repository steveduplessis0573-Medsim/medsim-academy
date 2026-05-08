import streamlit as st
import os
import re
from datetime import datetime
from dotenv import load_dotenv

# RAG Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown('<div class="ready-room"><h1>MedSim Academy</h1><p>Restricted Access</p></div>', unsafe_allow_html=True)
        st.text_input("Enter Academy Access Code", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Academy Access Code", type="password", on_change=password_entered, key="password")
        st.error("😕 Access Code denied.")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- 1. SETTINGS & CSS ---
st.set_page_config(page_title="MedSim Academy", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #333; }
    .vital-card { background-color: #000; border: 1px solid #444; border-radius: 5px; padding: 10px; margin-bottom: 8px; font-family: 'Courier New', monospace; }
    .vital-label { color: #888; font-size: 0.7rem; text-transform: uppercase; }
    .vital-value { color: #00FF00; font-size: 1.1rem; font-weight: bold; }
    .log-entry { font-size: 0.85rem; color: #00FF00; font-family: monospace; border-bottom: 1px solid #222; padding: 4px 0; }
    .ready-room { background-color: #1a1a1a; padding: 40px; border-radius: 15px; border: 1px solid #333; text-align: center; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINES ---
load_dotenv()

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

@st.cache_resource
def load_protocol_db():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local("protocol_db", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Protocol DB Error: {e}")
        return None

def get_protocol_context(user_query):
    db = load_protocol_db()
    if db:
        docs = db.similarity_search(user_query, k=3)
        return "\n---\n".join([d.page_content for d in docs])
    return "No protocol context available."

def process_medsim_turn(raw_text):
    diff = datetime.now() - st.session_state.start_time
    minutes, seconds = divmod(diff.seconds, 60)
    timestamp = f"T+{minutes:02d}:{seconds:02d}"

    # Vitals Normalizer
    v_matches = re.findall(r"\[VITAL\]\s*([^:\[\n\r]+?)\s*[:\-]\s*([^\[\n\r]+?)(?=\s*\[|$)", raw_text, re.IGNORECASE)
    vital_map = {
        "HEART RATE": "HR", "PULSE": "HR", "HR": "HR",
        "BLOOD PRESSURE": "BP", "BP": "BP",
        "RESPIRATORY RATE": "RR", "RESPIRATIONS": "RR", "RR": "RR",
        "BLOOD GLUCOSE": "BGL", "GLUCOSE": "BGL", "BGL": "BGL",
        "SPO2": "SPO2", "OXYGEN SATURATION": "SPO2", "O2": "SPO2", "SAT": "SPO2",
        "TEMPERATURE": "TEMP", "TEMP": "TEMP"
    }

    for label, val in v_matches:
        clean_key = re.sub(r'[^A-Z0-9]', '', label.strip().upper())
        final_key = vital_map.get(clean_key, label.strip().upper())
        st.session_state.vitals[final_key] = val.strip().rstrip(',')

    # Log Entries
    l_matches = re.findall(r"\[LOG\]\s*(?:[\d:]+)?\s*([^\[\n\r]+)", raw_text)
    for entry in l_matches:
        clean_entry = f"{timestamp} | {entry.strip()}"
        if clean_entry not in st.session_state.timeline:
            st.session_state.timeline.append(clean_entry)

    clean_text = re.sub(r"\[VITAL\].*?(?=\[|$)", "", raw_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\[LOG\].*?(?=\[|$)", "", clean_text, flags=re.IGNORECASE)
    return re.sub(r"\[.*?\]", "", clean_text).strip()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vitals" not in st.session_state: st.session_state.vitals = {}
if "timeline" not in st.session_state: st.session_state.timeline = []
if "sim_finished" not in st.session_state: st.session_state.sim_finished = False
if "start_time" not in st.session_state: st.session_state.start_time = datetime.now()
if "started" not in st.session_state: st.session_state.started = False

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("📟 Control Center")
    cat = st.selectbox("Category", ["Random", "Medical", "Trauma", "Pediatric", "Cardiac"], disabled=st.session_state.started)
    diff = st.select_slider("Acuity", options=["Easy", "Moderate", "Hard", "Critical"], disabled=st.session_state.started)
    mode = st.radio("Protocol Level", ["BLS", "ALS"], disabled=st.session_state.started)
    
    st.divider()
    st.subheader("📚 Protocol Search")
    search_q = st.text_input("Manual Lookup...", placeholder="e.g., Nitro contraindications")
    if st.button("Search PDF"):
        if search_q:
            results = get_protocol_context(search_q)
            st.info(results)
    
    if st.session_state.started:
        if st.button("🔄 Reset Academy"):
            for key in ["messages", "vitals", "timeline", "sim_finished", "started"]:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

# --- 5. MAIN UI ---
if not st.session_state.started:
    st.markdown('<div class="ready-room"><h1>MedSim Academy</h1><p>Protocol-Driven Clinical Simulations</p><br><br></div>', unsafe_allow_html=True)
    if st.button("🚀 START CALL", type="primary", use_container_width=True):
        st.session_state.started = True
        st.session_state.target_cat = cat
        st.session_state.target_diff = diff
        st.session_state.mode = mode
        st.session_state.start_time = datetime.now()
        st.rerun()
else:
    col_chat, col_data = st.columns([2, 1])
    
    with col_data:
        st.subheader("💓 Monitor")
        v = st.session_state.vitals
        c1, c2 = st.columns(2)
        def render(l, k):
            st.markdown(f'<div class="vital-card"><span class="vital-label">{l}</span><br><span class="vital-value">{v.get(k, "--")}</span></div>', unsafe_allow_html=True)
        with c1: render("HR", "HR"); render("RR", "RR"); render("BGL", "BGL")
        with c2: render("BP", "BP"); render("SpO2", "SPO2"); render("Temp", "TEMP")
        st.divider()
        st.subheader("🕒 Log")
        for entry in reversed(st.session_state.timeline):
            st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)

    with col_chat:
        if not st.session_state.messages:
            full_p = f"!!! {st.session_state.mode} MODE !!!\nCHALLENGE: {st.session_state.target_cat} | {st.session_state.target_diff}\n{st.secrets['SYSTEM_PROMPT_CONTENT']}"
            st.session_state.messages = [SystemMessage(content=full_p), HumanMessage(content=f"Dispatch a {st.session_state.target_diff} {st.session_state.target_cat} call.")]
            resp = get_llm().invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=process_medsim_turn(resp.content)))
            st.rerun()

        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage): continue
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            if role == "user" and "Dispatch a" in msg.content: continue
            
            # Stealth RAG Shield
            display_content = re.sub(r"--- LOCAL PROTOCOL REFERENCE ---.*?--- STUDENT ACTION ---", "", msg.content, flags=re.DOTALL).strip()
            with st.chat_message(role, avatar="🚑" if role=="assistant" else "🩺"):
                st.markdown(display_content)

        if not st.session_state.sim_finished:
            u_input = st.chat_input("Enter action...")
            if u_input:
                with st.spinner("🔍 Consulting Protocols..."):
                    context = get_protocol_context(u_input)
                
                enriched_input = f"--- LOCAL PROTOCOL REFERENCE ---\n{context}\n\n--- STUDENT ACTION ---\n{u_input}"
                st.session_state.messages.append(HumanMessage(content=enriched_input))
                
                with st.chat_message("assistant"):
                    resp = get_llm().invoke(st.session_state.messages)
                    st.session_state.messages.append(AIMessage(content=process_medsim_turn(resp.content)))
                    if "[DEBRIEF]" in resp.content: st.session_state.sim_finished = True
                    st.rerun()