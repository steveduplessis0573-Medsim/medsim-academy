import streamlit as st
import pandas as pd

st.set_page_config(page_title="EMS Analytics Dashboard", layout="wide")

# --- Password gate (same credential as the main app) ---
if "password_correct" not in st.session_state:
    st.markdown('<div style="text-align:center;padding:50px;"><h2>🔒 Analytics Dashboard</h2></div>', unsafe_allow_html=True)
    pwd = st.text_input("Access Code", type="password")
    if pwd:
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("😕 Access Denied.")
    st.stop()

# --- Database connection (mirrors app.py logic) ---
def _load_data():
    db_url = st.secrets.get("DATABASE_URL", "")
    try:
        if db_url:
            import psycopg2
            conn = psycopg2.connect(db_url)
            df = pd.read_sql("SELECT * FROM call_metrics ORDER BY id DESC", conn)
            conn.close()
        else:
            import sqlite3
            conn = sqlite3.connect("simulation_data.db")
            df = pd.read_sql_query("SELECT * FROM call_metrics ORDER BY id DESC", conn)
            conn.close()
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

df = _load_data()

st.title("📊 EMS Clinical Simulation Analytics")
st.caption("Real-time view of student performance across all simulation runs.")
st.markdown("---")

if df.empty:
    st.info("No simulation data yet. Complete a call in MedSim Academy to populate this dashboard.")
    st.stop()

# ── 1. TOP-LINE METRICS ──────────────────────────────────────────────────────
total    = len(df)
avg_score = int(df["score"].dropna().mean()) if not df["score"].dropna().empty else 0
pass_rate = int((df["pass_fail"].str.upper() == "PASS").sum() / total * 100) if total else 0

bls_df   = df[df["mode"].str.upper() == "BLS"]
als_df   = df[df["mode"].str.upper() == "ALS"]
bls_avg  = int(bls_df["score"].dropna().mean()) if not bls_df.empty else 0
als_avg  = int(als_df["score"].dropna().mean()) if not als_df.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Runs",       total)
c2.metric("Avg Score",        f"{avg_score}/100")
c3.metric("Pass Rate",        f"{pass_rate}%")
c4.metric("BLS Avg Score",    f"{bls_avg}/100")
c5.metric("ALS Avg Score",    f"{als_avg}/100")

st.markdown("---")

# ── 2. SCORE TREND ───────────────────────────────────────────────────────────
st.subheader("📈 Score Trend (chronological)")
chart_df = df.dropna(subset=["score"]).iloc[::-1].reset_index(drop=True)
st.line_chart(chart_df["score"])

st.markdown("---")

# ── 3. BREAKDOWN CHARTS ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏷️ Runs by Category")
    cat_counts = df["category"].value_counts()
    st.bar_chart(cat_counts)

with col_b:
    st.subheader("⚡ Runs by Acuity")
    acuity_order = ["Easy", "Moderate", "Hard", "Critical"]
    acuity_counts = df["acuity"].value_counts().reindex(acuity_order).dropna()
    st.bar_chart(acuity_counts)

st.markdown("---")

# ── 4. FULL RECORDS TABLE ────────────────────────────────────────────────────
st.subheader("📁 All Simulation Records")
st.dataframe(
    df,
    column_config={
        "id":           "Run #",
        "timestamp":    "Date/Time",
        "mode":         "Level",
        "acuity":       "Acuity",
        "category":     "Type",
        "complaint":    "Chief Complaint",
        "had_hazard":   "Hazard?",
        "used_refusal": "Refusal?",
        "score":        "Score",
        "pass_fail":    "Result",
        "transcript":   None,   # hidden — too long for the table view
    },
    use_container_width=True,
)

st.markdown("---")

# ── 5. TRANSCRIPT INSPECTOR ──────────────────────────────────────────────────
st.subheader("🔍 Full Debrief Transcript")
run_options = {f"Run #{row['id']} — {row['complaint']} ({row['mode']}, {row['timestamp'][:10]})": row['id']
               for _, row in df.iterrows()}
selected_label = st.selectbox("Select a run:", list(run_options.keys()))

if selected_label:
    selected_id = run_options[selected_label]
    row = df[df["id"] == selected_id].iloc[0]
    r1, r2, r3 = st.columns(3)
    r1.metric("Score",    f"{row['score']}/100")
    r2.metric("Result",   row["pass_fail"] or "—")
    r3.metric("Category", f"{row['category']} / {row['acuity']}")
    st.text_area("Full transcript:", value=row["transcript"] or "", height=450)
