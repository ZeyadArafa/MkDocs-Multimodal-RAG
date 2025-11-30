import streamlit as st
import os
from rag import load_rag_resources, ask_gemini 

st.set_page_config(page_title="MkDocs Multimodal RAG", layout="wide")
st.title("📚 MkDocs AI (Multimodal)")
st.caption("I can read text AND see the documentation images!")

# --- Load Resources ---
@st.cache_resource
def get_resources():
    return load_rag_resources()

resources = get_resources()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this message has an associated image, show it
        if "image" in message and message["image"]:
            st.image(message["image"], caption="Relevant Documentation Image", width=500)

if prompt := st.chat_input("Ask about config, themes, or deployment..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "image": None})

    with st.spinner("Searching text & scanning images..."):
        # Call the new function
        answer, docs, best_image = ask_gemini(prompt, resources)

    with st.chat_message("assistant"):
        st.markdown(answer)
        
        if best_image:
            st.image(best_image, caption="I found this relevant image in the docs!", width=500)
            
        with st.expander("View Source Context"):
            for doc in docs:
                st.write(f"Source: {doc.metadata['source']}")
                st.text(doc.page_content[:150] + "...")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "image": best_image
    })