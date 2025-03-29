from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
import whisper
from moviepy.editor import AudioFileClip
from pyannote.audio.pipelines import SpeakerDiarization

app = FastAPI()

# Hugging Face API Key (ersetzen mit deinem Token!)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Pfad zu ffmpeg setzen (wichtig für Mac, ggf. anpassen)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin/"

# Whisper-Modell laden (einmalig beim Start)
model = whisper.load_model("small")

# Diarization-Pipeline einmalig beim Start laden
pipeline = SpeakerDiarization.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=HUGGINGFACE_TOKEN
)

@app.post("/transkript")
async def create_transkript(audio: UploadFile = File(...)):
    input_path = f"input_files/{audio.filename}"
    audio_path = "input_files/temp_audio.wav"
    output_txt = f"output_files/{audio.filename}_transkript.txt"

    # Datei abspeichern
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Audio konvertieren zu WAV
    audio_clip = AudioFileClip(input_path)
    audio_clip.write_audiofile(audio_path, codec="pcm_s16le", fps=16000)

    # Transkription mit Whisper
    result = model.transcribe(audio_path, language="de")

    # Speaker Diarization
    diarization = pipeline({"uri": "audio", "audio": audio_path})

    # Sprecher-Zeiten auslesen
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append((turn.start, turn.end, speaker))

    # Sprecher im Transkript kennzeichnen
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

    # Ergebnis in Datei speichern
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(transcript_with_speakers)

    # Temporäre Dateien löschen
    os.remove(input_path)
    os.remove(audio_path)

    # Ergebnis-Datei zurückgeben
    return FileResponse(path=output_txt, filename="transkript.txt", media_type="text/plain")
