import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
import logging
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
import json
import base64
import subprocess
from werkzeug.utils import secure_filename
from gtts import gTTS

from config import Config
from agents.agent_decision import process_query
from agents.offline_fallback_agent import OfflineFallbackAgent

# Load configuration
config = Config()
logger = logging.getLogger(__name__)
offline_fallback_agent = OfflineFallbackAgent()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def get_baidu_access_token():
    """Get Access Token for Baidu API"""
    url = f"https://aip.baidubce.com/oauth/2.0/token?client_id={config.speech.baidu_api_key}&client_secret={config.speech.baidu_secret_key}&grant_type=client_credentials"
    payload = json.dumps("", ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload.encode("utf-8"))
    return response.json().get("access_token")


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes


# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()


class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []
    language: str = "en"
    preferred_agent: str = "AUTO"


class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID
    language: str = "en"


def _extract_response_text(response_data: Dict) -> str:
    messages = response_data.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content
        if isinstance(last_message, str):
            return last_message
    return ""


def _build_offline_response(query: Union[str, Dict], language: str, reason: str = "") -> Dict:
    response_text = offline_fallback_agent.generate(query=query, language=language)
    payload = {
        "status": "success",
        "response": response_text,
        "agent": "OFFLINE_FALLBACK_AGENT"
    }
    if reason:
        payload["fallback_reason"] = reason
    return payload


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}


@app.post("/chat")
def chat(
        request: QueryRequest,
        response: Response,
        session_id: Optional[str] = Cookie(None)
):
    """Process user text query through the multi-agent system."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    # Set session cookie
    response.set_cookie(key="session_id", value=session_id)

    # Force offline mode for fully disconnected deployments
    if config.api.force_offline_mode:
        return _build_offline_response(
            query=request.query,
            language=request.language,
            reason="force_offline_mode_enabled"
        )

    try:
        response_data = process_query(request.query, language=request.language, preferred_agent=request.preferred_agent)
        response_text = _extract_response_text(response_data)

        # If online pipeline returns empty output, still fallback gracefully.
        if not response_text.strip():
            return _build_offline_response(
                query=request.query,
                language=request.language,
                reason="empty_online_response"
            )

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data.get("agent_name", "UNKNOWN_AGENT")
        }

        # If it's the skin lesion segmentation agent, check for output image
        if response_data.get("agent_name") == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        return result
    except Exception as e:
        logger.exception("Online chat pipeline failed")
        if config.api.enable_offline_fallback:
            return _build_offline_response(
                query=request.query,
                language=request.language,
                reason=f"online_pipeline_error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_image(
        response: Response,
        image: UploadFile = File(...),
        text: str = Form(""),
        language: str = Form("en"),
        preferred_agent: str = Form("AUTO"),
        session_id: Optional[str] = Cookie(None)
):
    """Process medical image uploads with optional text input."""
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )

    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )

    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Set session cookie
    response.set_cookie(key="session_id", value=session_id)

    try:
        query = {"text": text, "image": file_path}

        if config.api.force_offline_mode:
            return _build_offline_response(
                query=query,
                language=language,
                reason="force_offline_mode_enabled"
            )

        response_data = process_query(query, language=language, preferred_agent=preferred_agent)
        response_text = _extract_response_text(response_data)

        if not response_text.strip():
            return _build_offline_response(
                query=query,
                language=language,
                reason="empty_online_response"
            )

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data.get("agent_name", "UNKNOWN_AGENT")
        }

        # If it's the skin lesion segmentation agent, check for output image
        if response_data.get("agent_name") == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        return result
    except Exception as e:
        logger.exception("Online upload pipeline failed")
        if config.api.enable_offline_fallback:
            query = {"text": text, "image": file_path}
            return _build_offline_response(
                query=query,
                language=language,
                reason=f"online_pipeline_error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove temporary file after response is prepared
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as remove_error:
            print(f"Failed to remove temporary file: {str(remove_error)}")


@app.post("/validate")
def validate_medical_output(
        response: Response,
        validation_result: str = Form(...),
        comments: Optional[str] = Form(None),
        language: str = Form("en"),
        session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for medical AI outputs."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"

        response_data = process_query(validation_query, language=language)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data['messages'][-1].content
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_data['messages'][-1].content
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Endpoint to transcribe speech using Baidu ASR API"""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )

    try:
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"

        # Read and save the file
        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)

        file_size = os.path.getsize(temp_audio)
        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )

        # Convert to PCM (Baidu requires pcm, wav, amr, m4a, 16000Hz, 16bit, mono)
        wav_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.wav"

        try:
            # Use subprocess to call ffmpeg to convert to 16kHz, mono, 16-bit wav
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_audio,
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                wav_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(wav_path, "rb") as f:
                speech_data = f.read()

            length = len(speech_data)
            if length == 0:
                raise Exception("Converted audio is empty")

            speech_base64 = base64.b64encode(speech_data).decode('utf-8')

            # Call Baidu API
            token = get_baidu_access_token()
            if not token:
                raise Exception("Failed to get Baidu access token")

            url = f"https://vop.baidu.com/server_api?dev_pid=1537&cuid=medical_agent_user&token={token}"

            payload = json.dumps({
                "format": "wav",
                "rate": 16000,
                "channel": 1,
                "cuid": "medical_agent_user",
                "token": token,
                "speech": speech_base64,
                "len": length
            })

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload)
            result = response.json()

            # Clean up temp files
            try:
                os.remove(temp_audio)
                os.remove(wav_path)
            except Exception as e:
                print(f"Could not delete temp files: {e}")

            if result.get("err_no") == 0 and result.get("result"):
                transcription = result["result"][0]
                return {"transcript": transcription}
            else:
                error_msg = result.get("err_msg", "Unknown API error")
                print(f"Baidu API Error: {error_msg}, code: {result.get('err_no')}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API error: {error_msg}"}
                )

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing audio: {str(e)}"}
            )

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using Google TTS (gTTS)"""
    try:
        text = request.text
        language = request.language

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )

        # Determine language for gTTS
        lang = 'zh-CN' if language == 'zh' else 'en'

        # Generate speech
        tts = gTTS(text=text, lang=lang)

        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        tts.save(temp_audio_path)

        # Return the generated audio file
        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)