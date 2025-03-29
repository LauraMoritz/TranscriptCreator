import streamlit as st
import requests

st.title("🎙️ Transcript Creator")

st.write("Lade eine Audio- oder Videodatei hoch, um sie transkribieren zu lassen.")

uploaded_file = st.file_uploader("Wähle eine Datei aus", type=["mp3", "mp4", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Transkription starten"):
        with st.spinner("Transkription läuft..."):
            files = {"audio": uploaded_file.getvalue()}
            response = requests.post(
                "https://transcriptcreator.onrender.com/transkript",  # <== HIER korrigiert!
                files=files
            )

            if response.status_code == 200:
                st.success("✅ Transkription erfolgreich!")
                st.download_button("📄 Transkript herunterladen", response.content, file_name="transkript.txt")
            else:
                st.error(f"❌ Fehler: {response.status_code}")
