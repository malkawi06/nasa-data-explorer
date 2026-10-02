"""Streamlit drag-and-drop app. Run with: nasa-explore --app"""

import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process

st.set_page_config(page_title="NASA Data Explorer", layout="wide")
st.title("NASA Data Explorer")
lang = st.sidebar.radio("Language", ["en", "ar"], horizontal=True)
use_ai = st.sidebar.checkbox("AI summary (needs Ollama or an API key)")
var = st.sidebar.text_input("Variable / column (optional)") or None
files = st.file_uploader("Drop data files, papers or archives", accept_multiple_files=True)

for up in files or []:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / up.name
        path.write_bytes(up.getbuffer())
        with st.spinner(f"Analysing {up.name}..."):
            reports = process(
                path, ReadOptions(var=var, lang=lang), Path(tmp) / "reports", ai=use_ai
            )
        for r in reports:
            with st.expander(f"{r.file} - {r.reader}", expanded=True):
                html = r.html_path.read_text(encoding="utf-8")
                components.html(html, height=900, scrolling=True)
                st.download_button(
                    "Download report",
                    html,
                    file_name=r.html_path.name,
                    mime="text/html",
                    key=str(r.html_path),
                )
