import os
import requests
import streamlit as st


BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Smart Librarian • RAG + Tool", page_icon="📚")
st.title("Smart Librarian - RAG + Tool")
st.caption("Recomandări de cărți bazate pe interese")


query = st.text_input("Ce vrei să citești?", placeholder="Ex: prietenie și magie, povești de război, distopie...")


c1, c2, c3, c4 = st.columns(4)
with c1:
    top_k = st.slider("Rezultate RAG", 1, 12, 6)
with c2:
    num_recs = st.slider("Recomandări", 1, 5, 2)
with c3:
    tts = st.toggle("Text to Speech", value=False)
with c4:
    gen_img = st.toggle("Imagine (cover)", value=False)


if query:
    with st.spinner("Generez recomandări..."):
        r = requests.post(f"{BACKEND}/recommend", json={
            "query": query,
            "top_k": top_k,
            "num_recommendations": num_recs,
            "language_filter": True,
            "generate_image": gen_img,
            "tts": tts
        })

    if not r.ok:
        st.error("Backend error!!!")
    else:
        data = r.json()
        items = data.get("items", [])
        if not items:
            st.warning("Nu am găsit potriviri.")
        for it in items:
            st.subheader(it.get("title") or "Fără titlu")
            if it.get("rationale"):
                st.write(it["rationale"])
            if it.get("detailed_summary"):
                st.markdown("**Rezumat (tool):**")
                st.write(it["detailed_summary"])
            cimg, caud = st.columns(2)
            with cimg:
                if it.get("image_path"):
                    st.image(it["image_path"], caption="Copertă generată")
            with caud:
                if it.get("audio_path"):
                    st.audio(it["audio_path"])
            st.markdown("---")
