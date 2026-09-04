"""
Streamlit UI for the AMR Stewardship Assistant.

Wraps app.py's retrieval/prompt logic (SYSTEM_PROMPTS, build_prompt,
MODEL_PATH, CACHE_DIR) without duplicating it. app.py itself is untouched
and remains usable as a standalone CLI.

Usage:
    streamlit run streamlit_app.py
"""

import pickle
import sys

import streamlit as st
from llama_cpp import Llama

from app import SYSTEM_PROMPTS, build_prompt, MODEL_PATH, CACHE_DIR

st.set_page_config(page_title="AMR Stewardship Assistant")


@st.cache_resource(show_spinner="Loading model...")
def load_llm(persona: str, n_ctx: int = 4096, n_threads: int = 4, n_batch: int = 512) -> Llama:
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run `bash download_model.sh` first.")
        st.stop()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        verbose=False,
    )

    cache_path = CACHE_DIR / f"{persona}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            llm.load_state(state)
            print(f"[cache] loaded warmed prefix for persona '{persona}'", file=sys.stderr)
        except Exception as e:
            print(f"[cache] could not load {cache_path.name}, continuing without it: {e}", file=sys.stderr)
    else:
        print(f"[cache] no warmed cache for '{persona}' -- run cache_warmup.py to speed up cold starts.", file=sys.stderr)

    return llm


st.title("AMR Stewardship Assistant")

persona = st.selectbox(
    "Persona",
    options=list(SYSTEM_PROMPTS.keys()),
    index=list(SYSTEM_PROMPTS.keys()).index("health_worker"),
)

if "history" not in st.session_state or st.session_state.get("persona") != persona:
    st.session_state.history = []
    st.session_state.persona = persona

llm = load_llm(persona)

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", state="running")
        placeholder = st.empty()

        messages = build_prompt(persona, user_query, history=st.session_state.history)
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=True,
        )

        answer = ""
        first_token = True
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content")
            if token:
                if first_token:
                    status.update(label="Answering...", state="running")
                    first_token = False
                answer += token
                placeholder.markdown(answer)

        status.update(label="Answering...", state="complete")

    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": answer})
