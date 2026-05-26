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
            ts = datetime.now() - st.session_state.start_time
            timestamp = f"T+{ts.seconds//60:02d}:{ts.seconds%60:02d}"
            st.session_state.timeline.append(f"{timestamp} | {clean_entry}")
    
    # 3. Clean the text for display
    return re.sub(r"\[VITAL\].*?(\n|$)|\[LOG\].*?(\n|$)", "", text, flags=re.IGNORECASE).strip()

def clean_for_display(text):
    """
    Strip the [LOG]+[VITAL] documentation footer block as a unit.
    Render any remaining (narrative) [VITAL] tags inline when they carry real
    values; strip silently when all are placeholders ('--').
    """
    # Strip [LOG] lines AND any [VITAL] lines that immediately follow them.
    # The system prompt mandates a [LOG] then [VITAL] footer on every response;
    # removing them together prevents footer vitals from appearing twice when
    # the LLM also reported vitals inline in the narrative body.
    text = re.sub(
        r"\[LOG\][^\n\r]*(\n[ \t]*\[VITAL\][^\n\r]*)*",
        "", text, flags=re.IGNORECASE,
    )

    # Check remaining (narrative) vitals for real values
    raw_values = re.findall(
        r"\[VITAL\]\s*(?:HR|BP|SPO2|RR|BGL|TEMP)[:\-]\s*([^\[\n\r]+)",
        text, re.IGNORECASE,
    )
    has_real_vitals = any(v.strip().strip("-").strip() for v in raw_values)

    if has_real_vitals:
        def _fmt(m):
            # Trailing space keeps adjacent code spans from merging in Markdown
            return f"`{m.group(1).upper()}: {m.group(2).strip()}` "
        text = re.sub(
            r"\[VITAL\]\s*(HR|BP|SPO2|RR|BGL|TEMP)[:\-]\s*([^\[\n\r]+)",
            _fmt, text, flags=re.IGNORECASE,
        )
    else:
        text = re.sub(r"\[VITAL\][^\n\r]*", "", text, flags=re.IGNORECASE)

    return text.strip()

def _get_db_connection():
    """
    Return (conn, is_postgres).
    Uses PostgreSQL when DATABASE_URL secret is present (Streamlit Cloud),
    falls back to local SQLite otherwise.
    """
    db_url = st.secrets.get("DATABASE_URL", "")
    if db_url:
        import psycopg2
        return psycopg2.connect(db_url), True
    else:
        import sqlite3
        conn = sqlite3.connect("simulation_data.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_metrics (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                mode      TEXT, acuity TEXT, category TEXT, complaint TEXT,
                had_hazard INTEGER, used_refusal INTEGER,
                score     INTEGER, pass_fail TEXT, transcript TEXT
            )
        """)
        conn.commit()
        return conn, False


def log_call_metrics(mode, acuity, category, complaint, response_text, had_hazard, used_refusal):
    """Write one completed simulation record to the analytics database."""
    try:
        # --- Extract score and pass/fail from the debrief text ---
        score_m = re.search(r'\[SCORE\]\s*(\d+)/100|SCORE[:\s]+(\d+)/100', response_text, re.IGNORECASE)
        score   = int(score_m.group(1) or score_m.group(2)) if score_m else None

        pf_m    = re.search(r'\[RESULT\]\s*(PASS|FAIL)|RESULT[:\s]+(PASS|FAIL)', response_text, re.IGNORECASE)
        pass_fail = (pf_m.group(1) or pf_m.group(2)).upper() if pf_m else None

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
        ph = "%s" if is_postgres else "?"   # placeholder differs between drivers
        conn.execute(
            f"""INSERT INTO call_metrics
                (timestamp, mode, acuity, category, complaint,
                 had_hazard, used_refusal, score, pass_fail, transcript)
               VALUES ({",".join([ph]*10)})""",
            (timestamp, mode, acuity, category, complaint,
             int(had_hazard), int(used_refusal), score, pass_fail, transcript),
        )
        conn.commit()
        conn.close()

    except Exception as e:
        # Never let a logging failure crash the simulation
        st.warning(f"⚠️ Analytics logging failed silently: {e}")

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
            for key in ["messages", "vitals", "timeline", "started", "sim_finished", "start_time", "current_complaint", "current_acuity", "current_category"]:
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
        The simulation concludes with a **[DEBRIEF]** when care is transferred, the patient expires or a refusal is obtained.
        ### **6. The Debrief**
        A clinical performance review will analyze your actions against local protocols.
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
        else:
            st.session_state.active_hazard = None
            st.session_state.scene_cleared = True

        st.session_state.started = True
        st.session_state.mode = mode
        st.session_state.start_time = datetime.now()
        st.session_state.sim_finished = False
        st.session_state.timeline = ["Simulation started."]
        
        # 3. DISPATCH
        sys_p = f"!!! {mode} MODE !!!\nACUITY: {acuity}\nCATEGORY: {cat}\n{st.secrets['SYSTEM_PROMPT_CONTENT'].replace('{mode}', mode)}"
        if custom_scenario.strip():
            sys_p += (
                "\n\n[CUSTOM SCENARIO OVERRIDE — PARAMETER LOCK ACTIVE]\n"
                "The instructor has specified exact dispatch parameters. "
                "SUSPEND all Procedural Variation rules. "
                "You MUST preserve the patient's exact age, sex, and chief complaint as dispatched — do NOT change any of these. "
                "Only the street address, apartment details, and incidental scene environment details may vary to make the dispatch realistic. "
                "Example: if dispatched as '75 YOM chest pain', the patient must be a 75-year-old male with chest pain."
            )
        st.session_state.messages = [
            SystemMessage(content=sys_p), 
            HumanMessage(content=f"Dispatch {dispatch_instruction}.")
        ]
        
        import time, httpx
        with st.spinner("Dispatching..."):
            dispatch_resp = None
            for attempt in range(3):
                try:
                    dispatch_resp = llm_engine.invoke(st.session_state.messages)
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        st.error("Could not connect to simulation engine. Please try again.")
                        for key in ["started", "mode", "start_time", "sim_finished", "timeline",
                                    "active_hazard", "scene_cleared", "hazard_warned",
                                    "current_complaint", "current_acuity", "current_category"]:
                            st.session_state.pop(key, None)
                        st.rerun()
            if dispatch_resp:
                process_medsim_turn(dispatch_resp.content)
                st.session_state.messages.append(AIMessage(content=dispatch_resp.content))
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
                clean_text = re.sub(r"--- LOCAL PROTOCOL REFERENCE ---.*--- STUDENT ACTION ---", "", msg.content, flags=re.DOTALL)
                clean_text = clean_for_display(clean_text)
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                with st.chat_message(role, avatar="🚑" if role=="assistant" else "🩺"):
                    st.markdown(clean_text)

        # Student Input
        if not st.session_state.sim_finished:
            if u_input := st.chat_input("Enter action..."):
                with st.chat_message("user"): st.markdown(u_input)
                
                import time, httpx

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
                    resource_keywords = [hazard["fix"], "pd", "police", "cop", "911", "resources", "animal control", "power company", "hazmat", "fire dept"]

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
                            f"Do not print the token name. Conclude by asking how they wish to proceed."
                        )
                        scene_history = st.session_state.messages + [HumanMessage(content=scene_directive)]
                        feedback = "[SCENE] Scene safety assessment in progress. Stand by."
                        for attempt in range(3):
                            try:
                                resp = llm_engine.invoke(scene_history)
                                feedback = resp.content
                                break
                            except (httpx.RemoteProtocolError, httpx.HTTPError):
                                time.sleep(2)
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
                            feedback = (
                                f"[SCENE] \U0001f6a8 CRITICAL SAFETY ERROR! You were explicitly warned about an "
                                f"active threat requiring {hazard['fix'].upper()} intervention.\n\n"
                                f"You cannot proceed with clinical patient care until you STAGE your ambulance "
                                f"or request the appropriate resources!"
                            )
                            st.session_state.messages.append(HumanMessage(content=u_input))
                            st.session_state.messages.append(AIMessage(content=feedback))
                            st.session_state.timeline.append("CRITICAL VIOLATION: Ignored scene safety warning.")
                            st.rerun()
                # --- END OF SCENE SAFETY FIREWALL ---
                else:
                    with st.spinner("🔍 Consulting Protocols..."):
                        context = get_protocol_context(u_input)

                    firewall = f"--- MANDATORY: STUDENT IS A {st.session_state.mode} PROVIDER ---\n"
                    enriched = f"{firewall}--- LOCAL PROTOCOL REFERENCE ---\n{context}\n\n--- STUDENT ACTION ---\n{u_input}"
                    temp_history = st.session_state.messages + [HumanMessage(content=enriched)]

                    with st.spinner("Responding..."):
                        max_retries = 3
                        resp = None
                        for attempt in range(max_retries):
                            try:
                                resp = llm_engine.invoke(temp_history)
                                break
                            except (httpx.RemoteProtocolError, httpx.HTTPError) as net_err:
                                if attempt < max_retries - 1:
                                    st.warning(f"Network hiccup. Reconnecting... (Attempt {attempt + 1}/{max_retries})")
                                    time.sleep(2)
                                else:
                                    st.error("The Google API server is dropping requests. Please check your connection or try again.")
                                    raise net_err

                    st.session_state.messages.append(HumanMessage(content=u_input))
                    st.session_state.messages.append(AIMessage(content=resp.content))
                    process_medsim_turn(resp.content)

                    if "[DEBRIEF]" in resp.content:
                        st.session_state.sim_finished = True
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
                        )
                    st.rerun()