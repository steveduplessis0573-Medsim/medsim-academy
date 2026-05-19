import streamlit as st
import os
import re
import random
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 1. PERFORMANCE CACHING ---
st.set_page_config(page_title="MedSim Academy", page_icon="🩺", layout="wide")

@st.cache_resource
def load_resources():
    # Cache the 'Brain' so it stays in RAM and never reloads during the session
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("protocol_db", embedder, allow_dangerous_deserialization=True)
    # Using 2.0-Flash as the stable fallback to bypass the 404 errors on '3' and '1.5'
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, timeout=60)
    return embedder, db, llm

embeddings, vector_db, llm_engine = load_resources()

# --- 2. ACCESS CONTROL ---
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown('<div style="text-align:center; padding: 50px;"><h1>🚑 MedSim Academy</h1><p>Restricted Access - Enter Credentials</p></div>', unsafe_allow_html=True)
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

# --- CSS: THE "STAY PUT" MONITOR ---
st.markdown("""
    <style>
        /* 1. Force all parent containers to allow sticky children to 'float' */
        [data-testid="stAppViewBlockContainer"], 
        [data-testid="stVerticalBlock"], 
        [data-testid="stHorizontalBlock"] {
            overflow: visible !important;
        }

        /* 2. Target the 3rd column's internal container specifically */
        div[data-testid="stColumn"]:nth-of-type(3) > div {
            position: -webkit-sticky !important;
            position: sticky !important;
            top: 2rem !important;
            z-index: 1000 !important;
            height: fit-content !important;
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
    </style>
""", unsafe_allow_html=True)

# --- 4. ENGINE FUNCTIONS ---
def get_protocol_context(query):
    # Retrieve top 2 chunks to minimize prompt size and maximize speed
    docs = vector_db.similarity_search(query, k=2)
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
            ts = datetime.now() - st.session_state.start_time
            timestamp = f"T+{ts.seconds//60:02d}:{ts.seconds%60:02d}"
            st.session_state.timeline.append(f"{timestamp} | {clean_entry}")
    
    # 3. Clean the text for display
    return re.sub(r"\[VITAL\].*?(\n|$)|\[LOG\].*?(\n|$)", "", text, flags=re.IGNORECASE).strip()

# --- 5. INITIAL SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vitals" not in st.session_state: st.session_state.vitals = {"HR": "--", "BP": "--", "RR": "--", "SPO2": "--", "BGL": "--", "TEMP": "--"}
if "timeline" not in st.session_state: st.session_state.timeline = []
if "started" not in st.session_state: st.session_state.started = False
if "sim_finished" not in st.session_state: st.session_state.sim_finished = False

# --- SIDEBAR UPDATES ---
with st.sidebar:
    st.title("📟 Control Center")
    cat = st.selectbox("Category", ["Medical", "Trauma", "Pediatric", "Cardiac"], disabled=st.session_state.started)
    acuity = st.select_slider("Acuity", options=["Easy", "Moderate", "Hard", "Critical"], disabled=st.session_state.started)
    mode = st.radio("Protocol Level", ["BLS", "ALS"], disabled=st.session_state.started)
    
    st.divider()
    # THE CUSTOM DISPATCH OVERRIDE
    st.subheader("🎯 Custom Dispatch")
    custom_scenario = st.text_input("Override Scenario:", placeholder="e.g., Snake bite to the hand", disabled=st.session_state.started)
    st.caption("Leave blank to use the random pool.")

    st.divider()
    st.subheader("📚 Protocol Lookup")
    search_q = st.text_input("Manual Search...", placeholder="e.g. CPAP indications")
    if st.button("Search PDF"):
        if search_q: st.info(get_protocol_context(search_q))
    
    if st.session_state.started:
        if st.button("🔄 Reset Academy"):
            for key in ["messages", "vitals", "timeline", "started", "sim_finished", "start_time"]:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

# --- 7. MAIN UI ---
if not st.session_state.started:
    st.markdown('<h1 style="text-align:center;">MedSim Academy</h1>', unsafe_allow_html=True)
    
    with st.expander("📖 READ FIRST: Standard Operating Procedure", expanded=True):
        st.markdown("""
        ### **1. Starting the Call**
        Select the **Category**, **Acuity**, and **Protocol Level** in the sidebar, then click **🚀 START CALL**.
        ### **2. Initial Response**
        Your first response must be **"Scene safe, BSI"**. This will trigger the scene and environment description.
        ### **3. Interaction**
        Treat the chat as your voice and hands. You can **Assess** (*"I check skin"*), **Intervene** (*"I start an IV"*), or **Communicate** (*"I talk to family"*).
        ### **4. Monitor & Timeline**
        The monitor updates vitals and the timeline logs your actions in real-time.
        ### **5. Ending the Call**
        The simulation concludes with a **[DEBRIEF]** when care is transferred or the patient expires.
        ### **6. The Debrief**
        A clinical performance review will analyze your actions against local protocols.
        """)

    if st.button("🚀 START CALL", type="primary", use_container_width=True):
        # 1. THE EXPANDED MASTER POOL (5 more added per category)
        pool = {
            "Medical": [
                "Opioid Overdose", "Hypoglycemic Emergency", "Active Stroke", "Sepsis", "Diabetic Emergency",
                "Status Epilepticus", "Anaphylaxis (Airway involvement)", "Heat Stroke (Environmental)", 
                "Carbon Monoxide Poisoning", "Suspected Meningitis (Fever/Neck stiffness)"
            ],
            "Trauma": [
                "Fall from ladder (20ft)", "GSW to abdomen", "MVC with entrapment", "Partial Amputation", "Tension Pneumothorax",
                "Open Pneumothorax (Sucking chest wound)", "Unstable Pelvic Fracture", "Flail Chest", 
                "Diving Accident (Suspected C-spine)", "Industrial Crush Injury (>4 hours)"
            ],
            "Pediatric": [
                "6-month-old Febrile Seizure", "2-year-old Choking", "8-year-old Severe Asthma", "4-year-old Near-drowning", "10-year-old Ped vs Car",
                "18-month-old Croup (Stridor)", "12-year-old DKA (Fruity breath)", "3-year-old Poisoning (Cleaning product)", 
                "5-year-old Anaphylaxis (Peanut allergy)", "7-year-old Burn (Hot water scald)"
            ],
            "Cardiac": [
                "Chest Pain (STEMI)", "Symptomatic Bradycardia", "Cardiac Arrest (V-Fib)", "Supraventricular Tachycardia (SVT)", "CHF (Pulmonary Edema)",
                "Aortic Dissection (Tearing pain)", "Third-Degree Heart Block", "Ventricular Tachycardia (Stable)", 
                "Hypertensive Crisis (Neuro symptoms)", "Cardiogenic Shock (Post-MI)"
            ]
        }
        
        # 2. SELECTION LOGIC
        # If the user typed in the Custom box, use that. Otherwise, pick random.
        if custom_scenario.strip():
            complaint = custom_scenario
            dispatch_instruction = f"a {acuity} {cat} emergency: {complaint}"
        else:
            choice = random.choice(pool.get(cat, ["General Illness"]))
            dispatch_instruction = f"a {acuity} {choice}"
        
        st.session_state.started = True
        st.session_state.mode = mode
        st.session_state.start_time = datetime.now()
        
        # 3. DISPATCH
        sys_p = f"!!! {mode} MODE !!!\nACUITY: {acuity}\nCATEGORY: {cat}\n{st.secrets['SYSTEM_PROMPT_CONTENT']}"
        st.session_state.messages = [
            SystemMessage(content=sys_p), 
            HumanMessage(content=f"Dispatch {dispatch_instruction}.")
        ]
        
        with st.spinner("Dispatching..."):
            resp = llm_engine.invoke(st.session_state.messages)
            process_medsim_turn(resp.content)
            st.session_state.messages.append(AIMessage(content=resp.content))
        st.rerun()

else:
    col_chat, _, col_data = st.columns([2, 0.1, 1.2])
    
    with col_data:
        st.subheader("💓 Patient Monitor")
        c1, c2 = st.columns(2)
        v_items = list(st.session_state.vitals.items())
        for i, (k, v) in enumerate(v_items):
            target_col = c1 if i % 2 == 0 else c2
            target_col.markdown(f'<div class="vital-card"><span class="vital-label">{k}</span><br><span class="vital-value">{v}</span></div>', unsafe_allow_html=True)
        
        st.divider()
        st.subheader("🕒 Timeline")
        for entry in reversed(st.session_state.timeline):
            st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)

    with col_chat:
        for msg in st.session_state.messages:
            if isinstance(msg, (AIMessage, HumanMessage)) and "Dispatch" not in msg.content:
                # Cleanup: Hide the 'Enriched' RAG context from the student's view
                clean_text = re.sub(r"--- LOCAL PROTOCOL REFERENCE ---.*--- STUDENT ACTION ---", "", msg.content, flags=re.DOTALL)
                clean_text = process_medsim_turn(clean_text)
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                with st.chat_message(role, avatar="🚑" if role=="assistant" else "🩺"):
                    st.markdown(clean_text)

        # Student Input
        if not st.session_state.sim_finished:
            if u_input := st.chat_input("Enter action..."):
                with st.chat_message("user"): st.markdown(u_input)
                
                with st.spinner("🔍 Consulting Protocols..."):
                    context = get_protocol_context(u_input)
                
                # Scope Firewall & Context Injection
                firewall = f"--- MANDATORY: STUDENT IS A {st.session_state.mode} PROVIDER ---\n"
                enriched = f"{firewall}--- LOCAL PROTOCOL REFERENCE ---\n{context}\n\n--- STUDENT ACTION ---\n{u_input}"
                
                # Create the temporary history loop for the LLM
                temp_history = st.session_state.messages + [HumanMessage(content=enriched)]
                
                with st.spinner("Responding..."):
                    resp = llm_engine.invoke(temp_history)
                    
                    # 1. Save clean actions to history
                    st.session_state.messages.append(HumanMessage(content=u_input)) 
                    st.session_state.messages.append(AIMessage(content=resp.content))
                    
                    # 2. THE CRITICAL REWIRE: Process vitals immediately *before* checking debrief
                    # This extracts the [VITAL] tags and puts them into st.session_state.vitals right now
                    process_medsim_turn(resp.content) 
                    
                    if "[DEBRIEF]" in resp.content:
                        st.session_state.sim_finished = True
                
                # 3. FORCE RE-RENDER: This jumps back to the top of the script
                # The next thing the user sees is the monitor painted with the brand new vitals
                st.rerun()