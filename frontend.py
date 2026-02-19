from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import compiled LangGraph app
# -----------------------------
from backend import app


# =============================
# Helpers
# =============================

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], payload: Any):
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current_state.update(next(iter(payload.values())))
        else:
            current_state.update(payload)
    return current_state


# =============================
# Markdown Renderer (FIXED)
# =============================

def render_md(md: str):
    """
    Proper markdown rendering with full LaTeX support.
    DO NOT fragment markdown.
    """

    st.markdown(
        md,
        unsafe_allow_html=True,
    )


# =============================
# Past Blogs Loader
# =============================

def list_blogs() -> List[Path]:
    files = [
        p for p in Path(".").glob("*.md")
        if p.is_file() and p.name.lower() != "readme.md"
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")
st.title("📝 LangGraph Blog Writing Agent")

# Sidebar
with st.sidebar:
    st.header("Generate Blog")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate", type="primary")

    st.divider()
    st.subheader("📂 Past Blogs")

    blogs = list_blogs()

    if blogs:
        labels = {}
        for p in blogs:
            md = read_md(p)
            title = extract_title(md, p.stem)
            label = f"{title} · {p.name}"
            labels[label] = p

        selected = st.radio(
            "Select blog",
            list(labels.keys()),
            label_visibility="collapsed"
        )

        if st.button("Load Blog"):
            md = read_md(labels[selected])
            st.session_state["last_out"] = {
                "plan": None,
                "evidence": [],
                "final": md,
            }
    else:
        st.caption("No saved blogs yet.")


# Tabs
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🧾 Logs"]
)

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


# =============================
# Run Graph
# =============================

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "final": "",
        "as_of": as_of.isoformat(),
        "recency_days": 7,
    }

    status = st.status("Running graph...", expanded=True)
    current_state: Dict[str, Any] = {}

    for kind, payload in try_stream(app, inputs):
        if kind == "updates":
            current_state = extract_latest_state(current_state, payload)
            log(str(payload))

        elif kind == "final":
            st.session_state["last_out"] = payload
            status.update(label="✅ Done", state="complete", expanded=False)


# =============================
# Render Results
# =============================

out = st.session_state.get("last_out")

if out:

    # -------- Plan Tab --------
    with tab_plan:
        plan = out.get("plan")
        if not plan:
            st.info("No plan available.")
        else:
            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()

            st.write("###", plan.get("blog_title"))
            st.write("**Audience:**", plan.get("audience"))
            st.write("**Tone:**", plan.get("tone"))
            st.write("**Blog kind:**", plan.get("blog_kind"))

            tasks = plan.get("tasks", [])
            if tasks:
                df = pd.DataFrame(tasks)
                st.dataframe(df, use_container_width=True)

    # -------- Evidence Tab --------
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned.")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(e)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # -------- Preview Tab --------
    with tab_preview:
        final_md = out.get("final") or ""

        if not final_md:
            st.warning("No markdown output.")
        else:
            render_md(final_md)

            title = extract_title(final_md, "blog")
            filename = f"{safe_slug(title)}.md"

            st.download_button(
                "⬇️ Download Markdown",
                final_md.encode("utf-8"),
                filename,
                mime="text/markdown",
            )

            bundle = bundle_zip(final_md, filename, Path("images"))
            st.download_button(
                "📦 Download Bundle",
                bundle,
                f"{safe_slug(title)}_bundle.zip",
                mime="application/zip",
            )

    # -------- Logs Tab --------
    with tab_logs:
        st.text_area("Logs", "\n\n".join(logs[-50:]), height=500)

else:
    st.info("Enter a topic and click **Generate**.")
