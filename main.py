from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
import whisper
import subprocess
from pyannote.audio.pipelines import SpeakerDiarization

app = FastAPI()

# Hugging Face API Key (aus Umgebungsvariable)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Optional: Pfad zu ffmpeg setzen (nur nötig für lokalen Mac, nicht auf Render)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin/"

# Whisper-Modell laden
model = whisper.load_model("tiny")

# Diarization-Pipeline laden
pipeline = SpeakerDiarization.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=HUGGINGFACE_TOKEN
)

@app.post("/transkript")
async def create_transkript(audio: UploadFile = File(...)):
    input_path = f"input_files/{audio.filename}"
    audio_path = "input_files/temp_audio.wav"
    output_txt = f"output_files/{audio.filename}_transkript.txt"

    # Hochgeladene Datei speichern
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Konvertiere Audio zu WAV (16kHz, mono) via ffmpeg
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ], check=True)

    # Transkription mit Whisper
    result = model.transcribe(audio_path, language="de")

    # Sprecheranalyse
    diarization = pipeline({"uri": "audio", "audio": audio_path})

    # Sprecher-Zeiten sammeln
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append((turn.start, turn.end, speaker))

    # Transkript mit Sprechern zusammenbauen
    transcript_with_speakers = ""
    current_speaker = None
    for segment in result["segments"]:
        start_time = segment["start"]
        text = segment["text"].strip()

        assigned_speaker = "Unbekannt"
        for s_start, s_end, speaker in speakers:
            if s_start <= start_time <= s_end:
                assigned_speaker = f"Sprecher {int(speaker[-1]) + 1}"
                break

        if assigned_speaker != current_speaker:
            transcript_with_speakers += f"\n{assigned_speaker}: {text}\n"
            current_speaker = assigned_speaker
        else:
            transcript_with_speakers += f"{text}\n"

    # Speichern in Datei
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript_with_speakers)

    # Temporäre Dateien löschen
    os.remove(input_path)
    os.remove(audio_path)

    # Transkript zurückgeben
    return FileResponse(path=output_txt, filename="transkript.txt", media_type="text/plain")
