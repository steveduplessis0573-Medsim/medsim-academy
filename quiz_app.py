import os, json, re, hashlib
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
    api_key = _secret("GOOGLE_API_KEY")
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    base  = os.path.dirname(os.path.abspath(__file__))
    db    = Chroma(persist_directory=os.path.join(base, "chroma_db"), embedding_function=embed)
    llm   = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        temperature=0.4,
    )
    return db, llm

vector_db, llm_engine = load_resources()

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
def generate_questions(mode: str, topic: str, n: int) -> list[dict]:
    query = topic if topic != "Random" else "EMS patient care protocol medication procedure assessment"
    docs  = vector_db.similarity_search(query, k=min(n + 2, 8))
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    other = "ALS" if mode == "BLS" else "BLS"
    prompt = f"""You are an EMS training quiz generator for Prince William County protocols.
You are creating questions for a {mode} provider. Do NOT ask about {other}-only medications or procedures.

PROTOCOL TEXT:
{context}

Generate exactly {n} quiz questions grounded only in the protocol text above.
Mix multiple-choice and true/false. At least half should be multiple-choice.
Make questions practical and clinically relevant — test dosing, indications,
contraindications, and scope decisions.

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

    resp = llm_engine.invoke([HumanMessage(content=prompt)])
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
