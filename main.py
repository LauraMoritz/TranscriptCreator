from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
import whisper
import subprocess

app = FastAPI()

# Whisper-Modell laden (tiny)
model = whisper.load_model("tiny")

@app.post("/transkript")
async def create_transkript(audio: UploadFile = File(...)):
    input_path = f"input_files/{audio.filename}"
    audio_path = "input_files/temp_audio.wav"
    output_txt = f"output_files/{audio.filename}_transkript.txt"

    # Dateien speichern
    os.makedirs("input_files", exist_ok=True)
    os.makedirs("output_files", exist_ok=True)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Audio mit ffmpeg in 16kHz WAV umwandeln
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        audio_path
    ], check=True)

    # Transkription mit Whisper
    result = model.transcribe(audio_path, language="de")

    # Nur den Text speichern
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Temporäre Dateien löschen
    os.remove(input_path)
    os.remove(audio_path)

    return FileResponse(path=output_txt, filename="transkript.txt", media_type="text/plain")
