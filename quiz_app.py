import os, json, re, hashlib, time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedSim Quiz", page_icon="🧠", layout="centered")

# ── Secrets ────────────────────────────────────────────────────────────────────
def _secret(key, default=""):
    val = os.environ.get(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

# ── Auth ───────────────────────────────────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.markdown("### 🧠 MedSim Protocol Quiz")
    pwd = st.text_input("Access code", type="password", label_visibility="collapsed",
                        placeholder="Enter access code")
    if st.button("Enter", use_container_width=True, type="primary"):
        token = hashlib.sha256(_secret("APP_PASSWORD", "").encode()).hexdigest()[:20]
        given = hashlib.sha256(pwd.encode()).hexdigest()[:20]
        if token == given:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect code")
    return False

if not check_password():
    st.stop()

# ── Resources ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db    = FAISS.load_local("protocol_db", embed, allow_dangerous_deserialization=True)
    llm   = ChatGoogleGenerativeAI(
        model=_secret("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.4,
        timeout=60,
    )
    return db, llm

vector_db, llm_engine = load_resources()

# ── Scope reference ────────────────────────────────────────────────────────────
@st.cache_data
def load_scope() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scope", "PWC_scope.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

SCOPE_REF = load_scope()

# ── Retry wrapper ──────────────────────────────────────────────────────────────
def _invoke(messages, max_attempts=4):
    delay, last_err = 3, None
    for attempt in range(max_attempts):
        try:
            return llm_engine.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["503","502","529","unavailable","overloaded","429","quota","try again"]) \
               and attempt < max_attempts - 1:
                last_err = e
                time.sleep(delay * (2 ** attempt))
            else:
                raise
    raise last_err or RuntimeError("No attempts made")

TOPICS = ["Random", "Cardiac", "Respiratory / Airway", "Trauma", "Toxicology",
          "Medications & Doses", "Pediatric", "OB / GYN", "Shock / Resuscitation"]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    .block-container { max-width: 680px; padding-top: 1.5rem; }

    .question-box {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 18px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e0e0f0;
        line-height: 1.55;
    }
    .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-bls { background:#1e3a5f; color:#90caf9; }
    .badge-als { background:#3b1f2b; color:#f48fb1; }
    .badge-mc  { background:#1e2d1e; color:#a5d6a7; }
    .badge-tf  { background:#2d2a1e; color:#ffe082; }

    .result-correct   { color:#4ade80; font-size:1.1rem; font-weight:700; margin:8px 0; }
    .result-incorrect { color:#f87171; font-size:1.1rem; font-weight:700; margin:8px 0; }

    .explanation-box {
        background:#162616;
        border-left: 4px solid #4ade80;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color:#c8e6c9;
        margin-top:10px;
        line-height:1.5;
    }

    .score-number { font-size:3.5rem; font-weight:800; text-align:center; }
    .score-label  { font-size:1rem; text-align:center; color:#aaa; margin-top:-10px; }

    /* Bigger radio tap targets on mobile */
    div[data-testid="stRadio"] > label { display:none; }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] { display:none; }

    @media (max-width: 768px) {
        .question-box { font-size: 0.98rem; padding: 16px; }
        .score-number { font-size: 2.8rem; }
    }
</style>
""", unsafe_allow_html=True)

# ── Question generation ────────────────────────────────────────────────────────
BLS_FORMULARY = """BLS FORMULARY — COMPLETE AND EXHAUSTIVE (no other medications exist for BLS, regardless of topic):
• Aspirin 324 mg PO (chewed) — suspected ACS / cardiac chest pain
• Epinephrine auto-injector: 0.3 ml IM if ≥65 lbs, 0.15 ml IM if <65 lbs — anaphylaxis; may repeat once in 5 min
• Albuterol 2.5 mg + Ipratropium Bromide 0.5 mg via nebulizer — respiratory distress (BLS max 2 doses; ALS may continue beyond)
• Ondansetron 4 mg ODT — nausea/vomiting
• Naloxone 2 mg IN — opioid overdose
• Oral glucose gel PO — symptomatic hypoglycemia (BGL <60)
• Nitroglycerin 0.4 mg SL (max 1.2 mg cumulative incl. patient's own doses) — ACS/pulmonary edema; hold if SBP <100 or PDE5 inhibitor within 48 hr
• CPAP 5 cm H2O — respiratory distress with adequate tidal volume
• Hemostatic gauze / wound packing — hemorrhage control
• DuoDote auto-injector (atropine + pralidoxime) — nerve agent/organophosphate with ≥2 mild or any severe symptoms

Do NOT generate any question that involves a medication not listed above.
Diphenhydramine, fentanyl, morphine, midazolam, amiodarone, adenosine, dextrose IV, ketamine,
ketorolac, glucagon, methylprednisolone, magnesium sulfate, metoprolol, epinephrine IV/IO, and
all other IV/IO medications are ALS-only — never ask a BLS provider about them."""


def generate_questions(mode: str, topic: str, n: int) -> list[dict]:
    query = topic if topic != "Random" else "EMS patient care protocol medication procedure assessment"
    docs  = vector_db.similarity_search(query, k=min(n + 2, 8))
    if mode == "BLS":
        docs = [d for d in docs if "[ALS SECTION:]" not in d.page_content]
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    scope_block = BLS_FORMULARY if mode == "BLS" else "For ALS mode: the full medication list in the SCOPE AUTHORITY is available."
    other_level = "ALS" if mode == "BLS" else "BLS"

    prompt = f"""You are an EMS training quiz generator for Prince William County (PWC) protocols.
You are creating questions for a {mode} provider.

SCOPE AUTHORITY — THIS IS THE AUTHORITATIVE SOURCE FOR ALL SCOPE DECISIONS:
{SCOPE_REF}

PROTOCOL TEXT (retrieved context for this topic):
{context}

SCOPE RULES:
- Only ask about medications, procedures, and decisions that are explicitly authorized for {mode} providers in the SCOPE AUTHORITY above.
- Do NOT ask {mode} providers about medications or procedures listed only under the {other_level} section.
- If a medication appears in both BLS and ALS sections, the question is valid for {mode}.

{scope_block}

Generate exactly {n} quiz questions grounded in the protocol text and scope above.
Mix multiple-choice and true/false. At least half should be multiple-choice.
Make questions practical and clinically relevant — test dosing, indications, contraindications, and scope decisions.

Return ONLY a valid JSON array with no surrounding text or markdown fences:
[
  {{
    "type": "mc",
    "question": "Question text here?",
    "options": ["A. First option", "B. Second option", "C. Third option", "D. Fourth option"],
    "correct": "B",
    "explanation": "One sentence citing the specific protocol rule or value."
  }},
  {{
    "type": "tf",
    "question": "True or False: Statement here.",
    "options": ["True", "False"],
    "correct": "True",
    "explanation": "One sentence citing the specific protocol rule or value."
  }}
]"""

    resp = _invoke([HumanMessage(content=prompt)])
    raw  = resp.content.strip()
    raw  = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    questions = json.loads(raw)
    # Sanitise — ensure every entry has required keys
    valid = []
    for q in questions:
        if all(k in q for k in ("type", "question", "options", "correct", "explanation")):
            valid.append(q)
    return valid[:n]

# ── Session state defaults ─────────────────────────────────────────────────────
def _reset_quiz():
    for k in ["questions", "answers", "q_index", "quiz_active",
              "quiz_done", "answered_current", "quiz_mode"]:
        st.session_state.pop(k, None)

for k, v in [("q_index", 0), ("questions", []), ("answers", []),
             ("quiz_active", False), ("quiz_done", False), ("answered_current", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── SCREEN 1: Setup ────────────────────────────────────────────────────────────
if not st.session_state.quiz_active and not st.session_state.quiz_done:
    st.title("🧠 Protocol Quiz")
    st.caption("Questions generated from PWC EMS protocols")
    st.divider()

    mode       = st.radio("Provider level", ["BLS", "ALS"], horizontal=True)
    topic      = st.selectbox("Topic", TOPICS)
    n_q        = st.select_slider("Questions", options=[5, 10, 15], value=10)

    st.write("")
    if st.button("Start Quiz →", use_container_width=True, type="primary"):
        with st.spinner("Generating questions from protocol library…"):
            try:
                qs = generate_questions(mode, topic, n_q)
                if not qs:
                    st.error("No questions generated — try a different topic.")
                else:
                    st.session_state.questions        = qs
                    st.session_state.quiz_mode        = mode
                    st.session_state.q_index          = 0
                    st.session_state.answers          = []
                    st.session_state.answered_current = False
                    st.session_state.quiz_active      = True
                    st.rerun()
            except json.JSONDecodeError:
                st.error("Couldn't parse questions — please try again.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── SCREEN 2: Quiz ─────────────────────────────────────────────────────────────
elif st.session_state.quiz_active and not st.session_state.quiz_done:
    questions = st.session_state.questions
    idx       = st.session_state.q_index
    total     = len(questions)

    if idx >= total:
        st.session_state.quiz_done   = True
        st.session_state.quiz_active = False
        st.rerun()

    q    = questions[idx]
    mode = st.session_state.get("quiz_mode", "BLS")

    # Header
    col_a, col_b = st.columns([3, 1])
    with col_a:
        badge_mode = f'<span class="badge badge-{mode.lower()}">{mode}</span>'
        badge_type = f'<span class="badge badge-{"mc" if q["type"]=="mc" else "tf"}">{"Multiple Choice" if q["type"]=="mc" else "True / False"}</span>'
        st.markdown(f"{badge_mode} &nbsp; {badge_type}", unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<p style="text-align:right;color:#888;font-size:0.85rem;padding-top:4px">{idx+1} / {total}</p>',
                    unsafe_allow_html=True)

    st.progress((idx) / total)
    st.markdown(f'<div class="question-box">{q["question"]}</div>', unsafe_allow_html=True)

    if not st.session_state.answered_current:
        answer = st.radio("", q["options"], index=None, key=f"q_{idx}")
        st.write("")
        if answer:
            if st.button("Submit Answer", use_container_width=True, type="primary"):
                # Determine if correct
                letter = answer[0] if q["type"] == "mc" else answer
                is_correct = letter.strip().upper() == q["correct"].strip().upper()
                correct_full = next(
                    (o for o in q["options"] if o.upper().startswith(q["correct"].upper())),
                    q["correct"],
                )
                st.session_state.answers.append({
                    "question":     q["question"],
                    "type":         q["type"],
                    "selected":     answer,
                    "correct":      q["correct"],
                    "correct_full": correct_full,
                    "is_correct":   is_correct,
                    "explanation":  q["explanation"],
                })
                st.session_state.answered_current = True
                st.rerun()
    else:
        result = st.session_state.answers[-1]
        if result["is_correct"]:
            st.markdown('<p class="result-correct">✓ Correct!</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<p class="result-incorrect">✗ Incorrect &mdash; correct answer: {result["correct_full"]}</p>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="explanation-box">📋 {result["explanation"]}</div>', unsafe_allow_html=True)
        st.write("")
        label = "Next Question →" if idx < total - 1 else "See Results →"
        if st.button(label, use_container_width=True, type="primary"):
            st.session_state.q_index          += 1
            st.session_state.answered_current  = False
            st.rerun()

# ── SCREEN 3: Results ──────────────────────────────────────────────────────────
elif st.session_state.quiz_done:
    answers = st.session_state.answers
    score   = sum(1 for a in answers if a["is_correct"])
    total   = len(answers)
    pct     = round(score / total * 100) if total else 0
    mode    = st.session_state.get("quiz_mode", "BLS")

    color = "#4ade80" if pct >= 80 else "#facc15" if pct >= 60 else "#f87171"
    verdict = "Pass ✓" if pct >= 80 else "Needs Review" if pct >= 60 else "Keep Studying"

    st.markdown(f'<p class="score-number" style="color:{color}">{pct}%</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="score-label">{score} / {total} correct &nbsp;·&nbsp; {mode} &nbsp;·&nbsp; {verdict}</p>',
                unsafe_allow_html=True)
    st.divider()

    st.subheader("Review")
    for i, a in enumerate(answers):
        icon = "✅" if a["is_correct"] else "❌"
        label = a["question"][:65] + ("…" if len(a["question"]) > 65 else "")
        with st.expander(f"{icon}  Q{i+1}: {label}"):
            st.write(f"**Your answer:** {a['selected']}")
            if not a["is_correct"]:
                st.write(f"**Correct answer:** {a['correct_full']}")
            st.markdown(f'<div class="explanation-box">📋 {a["explanation"]}</div>',
                        unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("New Quiz", use_container_width=True, type="primary"):
            _reset_quiz()
            st.rerun()
    with c2:
        if st.button("Retry Same Settings", use_container_width=True):
            st.session_state.questions        = []
            st.session_state.answers          = []
            st.session_state.q_index          = 0
            st.session_state.quiz_active      = False
            st.session_state.quiz_done        = False
            st.session_state.answered_current = False
            st.rerun()
