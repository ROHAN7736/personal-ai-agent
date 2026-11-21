"""Streamlit chat UI for the personal AI agent. Run: streamlit run app.py"""
import streamlit as st

import agent

st.set_page_config(page_title="Personal AI Agent", page_icon="🤖")
st.title("🤖 Personal AI Agent")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.caption(f"Provider: {agent.PROVIDER} · Model: {agent.MODEL}")
    st.metric("Tokens this session", agent.total_tokens)
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()
    st.subheader("Saved notes")
    st.text(agent.recall_notes())

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent.run(prompt, st.session_state.history)
            except Exception as e:
                st.error(f"Error: {e}")
    st.rerun()
