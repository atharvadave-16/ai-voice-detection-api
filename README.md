# AI-Generated Voice Detection API

This project is a REST API that detects whether a given voice recording is **AI-generated or Human** using language-agnostic acoustic analysis.

The system supports voice samples in:
- Tamil
- English
- Hindi
- Malayalam
- Telugu

## 🚀 Features
- Accepts Base64-encoded MP3 audio
- Classifies voice as `AI_GENERATED` or `HUMAN`
- Provides a confidence score (0.0 – 1.0)
- Returns a human-readable explanation
- API key–protected endpoint
- Publicly deployed and evaluation-ready

---

## 🧠 How It Works
The system extracts acoustic features such as:
- Pitch variability
- Spectral consistency
- Loudness regularity

These features are combined into a **confidence-based decision**, acknowledging that modern AI voices can closely resemble natural human speech.

---

## 📡 API Endpoint
https://ai-voice-detection-api.onrender.com/api/voice-detection

### Request Headers
x-api-key: 
Content-Type: application/json


### Request Body
 ```
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_MP3_STRING>"
}
Success Response
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.2,
  "explanation": "Voice shows natural pitch variation and irregular rhythm consistent with human speech."
}

Error Response
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```
 🛠 Tech Stack
 Python

FastAPI

Librosa

NumPy

Render (deployment)


⚠️ Notes

The model is intentionally explainable and confidence-based

Detection is probabilistic, not absolute

Designed for robustness and ethical AI usage

📌 Status

✅ Live


