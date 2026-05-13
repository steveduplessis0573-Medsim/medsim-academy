import streamlit as st
import os
import re
from datetime import datetime

# --- 1. CONFIG & CSS (Must be first) ---
st.set_page_config(page_title="MedSim Academy", page_icon="🩺", layout="wide")

# Inject CSS to make the third column (Monitor) "Sticky"
st.markdown(
    """
    <style>
        /* 1. Force the main container to allow sticky positioning */
        [data-testid="stAppViewBlockContainer"] {
            overflow: visible !important;
        }

        /* 2. Target the inner container of the 3rd column specifically */
        [data-testid="column"]:nth-of-type(3) [data-testid="stVerticalBlock"] {
            position: sticky !important;
            top: 2rem !important;
            z-index: 100;
            height: auto !important;
        }

        /* Styling for visual clarity */
        .vital-card { background-color: #000; border: 1px solid #444; border-radius: 5px; padding: 10px; margin-bottom: 8px; }
        .vital-label { color: #888; font-size: 0.7rem; text-transform: uppercase; }
        .vital-value { color: #00FF00; font-size: 1.1rem; font-weight: bold; }
        .log-entry { font-size: 0.85rem; color: #00FF00; font-family: monospace; border-bottom: 1px solid #222; padding: 4px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. ACCESS CONTROL ---
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown('<div class="ready-room"><h1>🚑 MedSim Academy</h1><p>Restricted Access - Enter Credentials</p></div>', unsafe_allow_html=True)
        pwd = st.text_input("Access Code", type="password")
        if pwd:
            if pwd == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Access Denied.")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. IMPORTS & ENGINES ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def get_llm():
    # Using the confirmed 2.5 Pro model for deep clinical reasoning
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

# --- 4. DATA PROCESSING ---
def render(l, k):
    v = st.session_state.vitals
    st.markdown(f'<div class="vital-card"><span class="vital-label">{l}</span><br><span class="vital-value">{v.get(k, "--")}</span></div>', unsafe_allow_html=True)

def process_medsim_turn(raw_text):
    diff = datetime.now() - st.session_state.start_time
    minutes, seconds = divmod(diff.seconds, 60)
    timestamp = f"T+{minutes:02d}:{seconds:02d}"

    v_pattern = r"(?:\[VITAL\]\s*)?(HR|BP|SPO2|RR|BGL|BG|GLUCOSE|TEMP|TEMPERATURE|PULSE)[:\-]\s*([^\[\n\r]+)"
    v_matches = re.findall(v_pattern, raw_text, re.IGNORECASE)
    
    vital_map = {
        "HEART RATE": "HR", "PULSE": "HR", "HR": "HR",
        "BLOOD PRESSURE": "BP", "BP": "BP",
        "RESPIRATORY RATE": "RR", "RESPIRATIONS": "RR", "RR": "RR",
        "BLOOD GLUCOSE": "BGL", "GLUCOSE": "BGL", "BGL": "BGL", "BG": "BGL",
        "SPO2": "SPO2", "O2": "SPO2", "SAT": "SPO2",
        "TEMPERATURE": "TEMP", "TEMP": "TEMP"
    }

    for label, val in v_matches:
        clean_label = label.strip().upper()
        final_key = vital_map.get(clean_label, clean_label)
        clean_val = val.strip().split(' ')[0].rstrip(',.')
        st.session_state.vitals[final_key] = clean_val

    l_matches = re.findall(r"\[LOG\]\s*([^\[\n\r]+)", raw_text, re.IGNORECASE)
    for entry in l_matches:
        clean_entry = f"{timestamp} | {entry.strip()}"
        if clean_entry not in st.session_state.timeline:
            st.session_state.timeline.append(clean_entry)

    clean_text = re.sub(r"\[VITAL\].*?(\n|$)", "", raw_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\[LOG\].*?(\n|$)", "", clean_text, flags=re.IGNORECASE)
    return clean_text.strip()

# --- 5. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vitals" not in st.session_state: st.session_state.vitals = {}
if "timeline" not in st.session_state: st.session_state.timeline = []
if "sim_finished" not in st.session_state: st.session_state.sim_finished = False
if "start_time" not in st.session_state: st.session_state.start_time = datetime.now()
if "started" not in st.session_state: st.session_state.started = False

# --- 6. SIDEBAR ---
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

# --- 7. MAIN UI ---
if not st.session_state.started:
    st.markdown('<div class="ready-room"><h1>MedSim Academy</h1><p>Protocol-Driven Clinical Simulations</p></div>', unsafe_allow_html=True)
    
    # --- INSTRUCTION EXPANDER ---
    with st.expander("📖 READ FIRST: Standard Operating Procedure", expanded=True):
        st.markdown("""
        ### **1. Starting the Call**
        Select the **Category**, **Acuity**, and **Protocol Level** (BLS/ALS) in the sidebar, then click **🚀 START CALL**.

        ### **2. Initial Response**
        Once you receive the dispatch, your first response must be **"Scene safe, BSI"**. This will trigger MedSim to describe the scene and environment.

        ### **3. Interacting with the Patient/Bystanders**
        Treat the chat box as your voice and hands. You can:
        *   **Assess:** *"I check the patient's skin condition"*
        *   **Intervene:** *"I am starting a large-bore IV with 0.9% Normal Saline."*
        *   **Communicate:** *"I ask the family for a list of medications."*
        
        MedSim will respond with realistic physiological changes. It will not tell you what to do, and it will not stop you from making mistakes. **You can kill your patient.**

        ### **4. Monitor & Timeline**
        *   **Monitor:** Updates vitals as you explicitly request or assess them.
        *   **Timeline:** Tracks your actions with realistic timing.

        ### **5. Ending the Call**
        The simulation concludes and triggers a **[DEBRIEF]** when:
        *   Care is transferred to ALS or the hospital.
        *   The patient refuses transport.
        *   The patient expires.

        ### **6. The Debrief**
        The debrief will analyze your actions against local protocols and provide a clinical performance review. Protocol citations will only appear here if a violation occurred.

        ---
        * **Printing:** Select the three dots (⋮) in the top right of the screen to print your call record.
        * **Manual Lookup:** Use the **Protocol Search** in the sidebar at any time to reference local guidelines.
        """)

    if st.button("🚀 START CALL", type="primary", use_container_width=True):
        st.session_state.started = True
        st.session_state.target_cat = cat
        st.session_state.target_diff = diff
        st.session_state.mode = mode
        st.session_state.start_time = datetime.now()
        st.rerun()
else:
    # 3-Column Layout: Chat (Left) | Spacer | Monitor (Right)
    col_chat, col_gap, col_data = st.columns([2, 0.1, 1.2])
    
    # --- RIGHT COLUMN: THE STATIC MONITOR ---
    with col_data:
        st.subheader("💓 Patient Monitor")
        # Vitals Grid
        c1, c2 = st.columns(2)
        with c1: 
            render("HR", "HR"); render("RR", "RR"); render("BGL", "BGL")
        with c2: 
            render("BP", "BP"); render("SpO2", "SPO2"); render("Temp", "TEMP")
        
        st.divider()
        st.subheader("🕒 Timeline")
        # Display the logs already stored in session state
        for entry in reversed(st.session_state.timeline):
            st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)

    # --- LEFT COLUMN: THE SCROLLABLE CHAT ---
    with col_chat:
        # Line 186: This must be indented under 'with col_chat'
        chat_container = st.container(height=700, border=False)

        with chat_container:
            # Line 188: This must be indented under 'with chat_container'
            if not st.session_state.messages:
                full_p = f"!!! {st.session_state.mode} MODE !!!\nCHALLENGE: {st.session_state.target_cat} | {st.session_state.target_diff}\n{st.secrets['SYSTEM_PROMPT_CONTENT']}"
                st.session_state.messages = [SystemMessage(content=full_p), HumanMessage(content=f"Dispatch a {st.session_state.target_diff} {st.session_state.target_cat} call.")]
                
                with st.spinner("Dispatching..."):
                    resp = get_llm().invoke(st.session_state.messages)
                    # This captures the first [LOG] for the timeline
                    process_medsim_turn(resp.content) 
                    # This saves the raw response (with hidden tags) to history
                    st.session_state.messages.append(AIMessage(content=resp.content))
                st.rerun()

            # 2. DISPLAY CHAT HISTORY (Also indented inside the container)
            for msg in st.session_state.messages:
                if isinstance(msg, SystemMessage): continue
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                if role == "user" and "Dispatch a" in msg.content: continue
                
                # Cleanup for the student's eyes
                d_text = re.sub(r"--- LOCAL PROTOCOL REFERENCE ---.*?--- STUDENT ACTION ---", "", msg.content, flags=re.DOTALL).strip()
                d_text = re.sub(r"\[VITAL\].*?(\n|$)", "", d_text, flags=re.IGNORECASE)
                d_text = re.sub(r"\[LOG\].*?(\n|$)", "", d_text, flags=re.IGNORECASE)

                with st.chat_message(role, avatar="🚑" if role=="assistant" else "🩺"):
                    st.markdown(d_text)
                    
        # 3. USER INPUT (Stays at the bottom of the chat column)
        if not st.session_state.sim_finished:
            u_input = st.chat_input("Enter action...")
            if u_input:
                # Add user input to history
                st.session_state.messages.append(HumanMessage(content=u_input))
                
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Consulting Protocols..."):
                            context = get_protocol_context(u_input)
                        
                        # Prepare the enriched prompt for the AI
                        enriched_input = f"--- LOCAL PROTOCOL REFERENCE ---\n{context}\n\n--- STUDENT ACTION ---\n{u_input}"
                        
                        # Replace the last user message with the enriched version for the LLM
                        st.session_state.messages[-1] = HumanMessage(content=enriched_input)
                        
                        resp = get_llm().invoke(st.session_state.messages)
                        process_medsim_turn(resp.content) # Updates timeline/vitals
                        st.session_state.messages.append(AIMessage(content=resp.content))
                        
                        if "[DEBRIEF]" in resp.content: 
                            st.session_state.sim_finished = True
                st.rerun()