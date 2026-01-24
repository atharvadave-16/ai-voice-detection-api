from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from detector import detect_voice_from_base64

API_KEY = "sk_test_123456789"  # change later

app = FastAPI(title="AI Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/api/voice-detection")
def voice_detection(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    classification, confidence, explanation = detect_voice_from_base64(
        request.audioBase64
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
