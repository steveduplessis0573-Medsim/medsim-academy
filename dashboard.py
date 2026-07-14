import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="MedSim Academy — Instructor Analytics", layout="wide")

# ── DB connection (Neon on Railway, SQLite locally) ─────────────────────────
def _secret(key, default=None):
    val = os.environ.get(key)
    if val is not None:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

@st.cache_data(ttl=60)
def load_data():
    db_url = _secret("DATABASE_URL", "")
    if db_url:
        import psycopg2
        conn = psycopg2.connect(db_url)
        df = pd.read_sql("SELECT * FROM call_metrics ORDER BY id DESC", conn)
        conn.close()
    else:
        import sqlite3
        conn = sqlite3.connect("simulation_data.db")
        df = pd.read_sql("SELECT * FROM call_metrics ORDER BY id DESC", conn)
        conn.close()
    # Normalise columns that may not exist on older rows
    if "standard" not in df.columns:
        df["standard"] = "PWC"
    df["standard"] = df["standard"].fillna("PWC")
    df["standard_label"] = df["standard"].map({"PWC": "Local Protocol", "NREMT": "National Registry"}).fillna("Local Protocol")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    return df

st.title("MedSim Academy — Instructor Analytics")
st.caption("Historical student performance metrics, trends, and full evaluation transcripts.")

if st.button("Refresh"):
    load_data.clear()
    st.rerun()

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.stop()

if df.empty:
    st.info("No simulation data yet — complete a run in the main app first.")
    st.stop()

# ── 1. METRIC CARDS ─────────────────────────────────────────────────────────
total_runs  = len(df)
last_score  = int(df["score"].dropna().iloc[0]) if df["score"].notna().any() else 0
avg_score   = int(df["score"].mean()) if df["score"].notna().any() else 0
pass_rate   = int(len(df[df["pass_fail"] == "PASS"]) / total_runs * 100) if total_runs else 0
als_avg     = int(df[df["mode"] == "ALS"]["score"].mean()) if len(df[df["mode"] == "ALS"]) else 0
bls_avg     = int(df[df["mode"] == "BLS"]["score"].mean()) if len(df[df["mode"] == "BLS"]) else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last Run", f"{last_score}/100")
c2.metric("Avg Score", f"{avg_score}/100")
c3.metric("Pass Rate", f"{pass_rate}%")
c4.metric("ALS Avg", f"{als_avg}/100")
c5.metric("BLS Avg", f"{bls_avg}/100")

st.markdown("---")

# ── 2. CHARTS ROW ───────────────────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

# Left: stacked bar — Runs by Category, coloured by Acuity
with chart_left:
    st.subheader("Runs by Category & Acuity")
    acuity_order = ["Easy", "Moderate", "Hard", "Critical"]
    cat_pivot = (
        df.groupby(["category", "acuity"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[c for c in acuity_order if c in df["acuity"].unique()], fill_value=0)
    )
    st.bar_chart(cat_pivot, color=["#4ade80", "#facc15", "#fb923c", "#f87171"])

# Right: Call volume by date
with chart_right:
    st.subheader("Call Volume by Date")
    daily = (
        df.dropna(subset=["date"])
        .groupby("date")
        .size()
        .rename("Calls")
        .sort_index()
    )
    st.bar_chart(daily)

st.markdown("---")

# ── 3. RECORDS TABLE ────────────────────────────────────────────────────────
st.subheader("All Simulation Records")

display_cols = {
    "id":             st.column_config.NumberColumn("Run ID", width="small"),
    "timestamp":      st.column_config.DatetimeColumn("Date/Time", format="MM/DD/YY HH:mm"),
    "student":        st.column_config.TextColumn("Student"),
    "mode":           st.column_config.TextColumn("Level", width="small"),
    "acuity":         st.column_config.TextColumn("Acuity", width="small"),
    "category":       st.column_config.TextColumn("Type"),
    "complaint":      st.column_config.TextColumn("Chief Complaint"),
    "standard_label": st.column_config.TextColumn("Standard"),
    "had_hazard":     st.column_config.CheckboxColumn("Hazard?", width="small"),
    "used_refusal":   st.column_config.CheckboxColumn("Refusal?", width="small"),
    "score":          st.column_config.NumberColumn("Score", width="small"),
    "pass_fail":      st.column_config.TextColumn("Status", width="small"),
    "transcript":     None,
    "narrative":      None,
    "standard":       None,
    "date":           None,
}

show_cols = [c for c in display_cols if c in df.columns and display_cols[c] is not None]
st.dataframe(
    df[show_cols],
    column_config={k: v for k, v in display_cols.items() if k in show_cols},
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ── 4. TRANSCRIPT INSPECTOR ─────────────────────────────────────────────────
st.subheader("Full Debrief Transcript")

run_options = df["id"].tolist()
selected_id = st.selectbox(
    "Select Run ID:",
    run_options,
    format_func=lambda i: f"#{i} — {df[df['id']==i]['complaint'].values[0]} ({df[df['id']==i]['timestamp'].values[0]})",
)

if selected_id:
    row = df[df["id"] == selected_id].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{row.get('score', '—')}/100")
    c2.metric("Status", row.get("pass_fail", "—"))
    c3.metric("Category", f"{row.get('category', '—')} / {row.get('acuity', '—')}")

    if row.get("transcript"):
        st.text_area("Chat & Debrief Log:", value=row["transcript"], height=400)

    if row.get("narrative"):
        with st.expander("Model PCR Narrative"):
            st.caption("Reflects ideal care — not the student's actual actions.")
            st.markdown(row["narrative"])
