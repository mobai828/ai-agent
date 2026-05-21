import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List, Literal
import glob
import threading
import time
import logging
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import uvicorn
import requests
import json
import base64
import subprocess
from werkzeug.utils import secure_filename
from gtts import gTTS

from config import Config
from agents.agent_decision import cleanup_expired_checkpoints, process_query
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
BRAIN_TUMOR_OUTPUT = "uploads/brain_tumor_output"
BRAIN_STROKE_OUTPUT = "uploads/brain_stroke_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [
    UPLOAD_FOLDER,
    FRONTEND_UPLOAD_FOLDER,
    BRAIN_TUMOR_OUTPUT,
    BRAIN_STROKE_OUTPUT,
    SPEECH_DIR,
]:
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


def cleanup_langgraph_checkpoints():
    """Periodically clean expired LangGraph checkpoint threads."""
    while True:
        try:
            cleaned = cleanup_expired_checkpoints()
            if cleaned:
                logger.info("Cleaned %s expired LangGraph checkpoint threads.", cleaned)
        except Exception as e:
            logger.warning("Error during LangGraph checkpoint cleanup: %s", e)
        time.sleep(config.checkpoint.cleanup_interval_seconds)


if getattr(config.checkpoint, "cleanup_enabled", True):
    checkpoint_cleanup_thread = threading.Thread(target=cleanup_langgraph_checkpoints, daemon=True)
    checkpoint_cleanup_thread.start()


class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []
    language: str = "en"
    preferred_agent: str = "AUTO"


class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID
    language: str = "en"


class BrainStrokeStages(BaseModel):
    segmentation: bool = Field(
        ...,
        description="阶段1：图像分割是否成功完成。"
    )
    lesion_marking: bool = Field(
        ...,
        description="阶段2：病灶标记图是否成功生成。"
    )
    ai_diagnosis: bool = Field(
        ...,
        description="阶段3：AI 初步结论文本是否成功生成。"
    )


class BrainStrokeSegmentResponse(BaseModel):
    status: Literal["success", "error", "not_implemented"] = Field(
        ...,
        description="本次处理状态。"
    )
    task: Literal["auto", "hemorrhage", "ischemia"] = Field(
        ...,
        description="实际执行的卒中子任务。"
    )
    source: Optional[Literal["remote", "local"]] = Field(
        default=None,
        description="分割来源：remote=远程 Heyi 服务；local=本地 fallback。"
    )
    stages: BrainStrokeStages = Field(
        ...,
        description="三阶段流程执行状态。"
    )
    diagnosis: str = Field(
        ...,
        description="AI 初步结论（markdown 文本）。"
    )
    message: str = Field(
        ...,
        description="简要状态消息，便于日志和上游展示。"
    )
    result_image: Optional[str] = Field(
        default=None,
        description="分割结果图 URL（通常带 ?v=时间戳 防缓存）。"
    )


class APIErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str
    task: Optional[Literal["auto", "hemorrhage", "ischemia"]] = None


def _extract_response_text(response_data: Dict) -> str:
    messages = response_data.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content
        if isinstance(last_message, str):
            return last_message
    return ""


def _attach_brain_cv_result_image(result: Dict, response_data: Dict) -> None:
    """If the reserved-interface brain CV agents produced a result image, expose it to the frontend.

    结果图固定写到 ``brain_*_plot.png``，文件名不变，浏览器会按 URL 缓存导致
    "第二次上传仍然显示第一次的结果"。这里在 URL 后追加 ``?v=<mtime>``，
    让每次新写入的文件对应一个新的 URL，强制前端重新拉取。
    """
    agent_name = response_data.get("agent_name", "") or ""
    if "BRAIN_TUMOR_AGENT" in agent_name:
        plot_path = os.path.join(BRAIN_TUMOR_OUTPUT, "brain_tumor_plot.png")
        if os.path.exists(plot_path):
            result["result_image"] = _cache_busted_url(
                "/uploads/brain_tumor_output/brain_tumor_plot.png", plot_path
            )
    elif "BRAIN_STROKE_AGENT" in agent_name:
        plot_path = os.path.join(BRAIN_STROKE_OUTPUT, "brain_stroke_plot.png")
        if os.path.exists(plot_path):
            result["result_image"] = _cache_busted_url(
                "/uploads/brain_stroke_output/brain_stroke_plot.png", plot_path
            )


def _cache_busted_url(public_url: str, file_path: str) -> str:
    """给静态资源 URL 追加基于 mtime 的版本戳，避免浏览器缓存旧结果。"""
    try:
        version = int(os.path.getmtime(file_path) * 1000)
    except OSError:
        version = int(time.time() * 1000)
    sep = "&" if "?" in public_url else "?"
    return f"{public_url}{sep}v={version}"


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
    if config.api.force_offline_mode:
        return _build_offline_response(
            query=request.query,
            language=request.language,
            reason="force_offline_mode_enabled"
        )

    try:
        response_data = process_query(
            request.query,
            language=request.language,
            preferred_agent=request.preferred_agent,
            session_id=session_id,
        )
        response_text = _extract_response_text(response_data)

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data.get("agent_name", "UNKNOWN_AGENT")
        }

        # Reserved-interface brain CV agents: surface result image if generated
        _attach_brain_cv_result_image(result, response_data)

        if not response_text.strip() and config.api.enable_offline_fallback:
            return _build_offline_response(
                query=request.query,
                language=request.language,
                reason="empty_online_response"
            )

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


SUPPORTED_STROKE_TASKS = {"auto", "hemorrhage", "ischemia"}


def _normalize_stroke_task(raw: Optional[str]) -> Optional[str]:
    """归一化前端 / 直连接口传入的 stroke_task 字段。

    支持中英文别名（出血/缺血等），统一映射为远程服务能识别的小写 token。
    输入为空 / None 返回 None，由下游决定是否走默认值。
    """
    if not raw:
        return None
    cleaned = str(raw).strip().lower()
    aliases = {
        "出血": "hemorrhage",
        "出血灶": "hemorrhage",
        "hemorrhagic": "hemorrhage",
        "haemorrhage": "hemorrhage",
        "缺血": "ischemia",
        "缺血灶": "ischemia",
        "ischemic": "ischemia",
        "ischaemia": "ischemia",
        "infarction": "ischemia",
        "梗死": "ischemia",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in SUPPORTED_STROKE_TASKS:
        return None
    return cleaned


@app.post("/upload")
async def upload_image(
        response: Response,
        image: UploadFile = File(...),
        text: str = Form(""),
        language: str = Form("en"),
        preferred_agent: str = Form("AUTO"),
        stroke_task: Optional[str] = Form(None),
        session_id: Optional[str] = Cookie(None)
):
    """Process medical image uploads with optional text input.

    Args:
        stroke_task: 仅对 BRAIN_STROKE_AGENT 生效，``auto`` / ``hemorrhage`` /
            ``ischemia``。推荐由上游分类模块（"判断缺血或出血"的板块）
            显式给定，避免远程服务在 ``auto`` 下做次优判断。
    """
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

    normalized_stroke_task = _normalize_stroke_task(stroke_task)

    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    if config.api.force_offline_mode:
        return _build_offline_response(
            query={"text": text, "image": file_path, "stroke_task": normalized_stroke_task},
            language=language,
            reason="force_offline_mode_enabled"
        )

    try:
        query = {"text": text, "image": file_path, "stroke_task": normalized_stroke_task}
        response_data = process_query(
            query,
            language=language,
            preferred_agent=preferred_agent,
            session_id=session_id,
        )
        response_text = _extract_response_text(response_data)

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data.get("agent_name", "UNKNOWN_AGENT"),
            "stroke_task": normalized_stroke_task,
        }

        # Reserved-interface brain CV agents: surface result image if generated
        _attach_brain_cv_result_image(result, response_data)

        if not response_text.strip() and config.api.enable_offline_fallback:
            return _build_offline_response(
                query=query,
                language=language,
                reason="empty_online_response"
            )

        return result
    except Exception as e:
        logger.exception("Online upload pipeline failed")
        if config.api.enable_offline_fallback:
            query = {"text": text, "image": file_path, "stroke_task": normalized_stroke_task}
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


@app.post(
    "/api/brain_stroke/segment",
    response_model=BrainStrokeSegmentResponse,
    responses={
        200: {
            "description": "脑卒中三阶段流水线执行成功（直连接口）。",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "task": "ischemia",
                        "source": "remote",
                        "stages": {
                            "segmentation": True,
                            "lesion_marking": True,
                            "ai_diagnosis": True
                        },
                        "diagnosis": "**AI 初步结论（脑卒中检测）**\n\n- 任务类型：**缺血灶（ischemia）**\n- 共检测到 **1** 个可疑区域；主病灶位于 **右脑上部**。",
                        "message": "脑卒中检测完成（远程 Heyi 分割服务，task=ischemia）。",
                        "result_image": "/uploads/brain_stroke_output/brain_stroke_plot.png?v=1714032123456"
                    }
                }
            }
        },
        400: {
            "description": "请求参数错误（例如文件格式不支持）。",
            "model": APIErrorResponse,
        },
        413: {
            "description": "文件体积超过限制。",
            "model": APIErrorResponse,
        },
        500: {
            "description": "服务端处理异常。",
            "model": APIErrorResponse,
        },
    },
)
async def api_brain_stroke_segment(
    image: UploadFile = File(...),
    task: str = Form("auto"),
):
    """脑卒中分割直连接口（**给负责缺血/出血判断的外部板块调用**）。

    与 ``/upload`` 不同，本接口绕过对话路由，直接返回脑卒中三阶段流水线
    的结构化结果，便于其他系统/微服务集成。

    Form 参数：
        image (file): 待分析的医学影像（PNG / JPG / JPEG）。
        task  (str):  ``auto`` / ``hemorrhage`` / ``ischemia``。
                      上游建议**显式**给值；缺省 ``auto`` 仅作为兜底。

    返回值：
        ```json
        {
          "status": "success" | "error" | "not_implemented",
          "task":   "auto" | "hemorrhage" | "ischemia",
          "source": "remote" | "local",
          "diagnosis": "**AI 初步结论…**",
          "stages":   {"segmentation": true, "lesion_marking": true, "ai_diagnosis": true},
          "result_image": "/uploads/brain_stroke_output/brain_stroke_plot.png?v=...",
          "message": "脑卒中检测完成（远程 Heyi 分割服务，task=ischemia）。"
        }
        ```
    """
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Unsupported file type. Allowed formats: PNG, JPG, JPEG",
            },
        )

    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "message": (
                    f"File too large. Maximum size allowed: "
                    f"{config.api.max_image_upload_size}MB"
                ),
            },
        )

    normalized_task = _normalize_stroke_task(task) or "auto"

    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        from agents.agent_decision import AgentConfig as _AC

        pipeline_result = _AC.image_analyzer.detect_brain_stroke(
            file_path, task=normalized_task
        )

        stages = pipeline_result.get("stages", {}) or {}
        payload: Dict = {
            "status": pipeline_result.get("status", "error"),
            "task": pipeline_result.get("task", normalized_task),
            "source": pipeline_result.get("source"),
            "stages": {
                "segmentation": bool(stages.get("segmentation", False)),
                "lesion_marking": bool(stages.get("lesion_marking", False)),
                "ai_diagnosis": bool(stages.get("ai_diagnosis", False)),
            },
            "diagnosis": pipeline_result.get("diagnosis", ""),
            "message": pipeline_result.get("message", ""),
        }

        plot_path = os.path.join(BRAIN_STROKE_OUTPUT, "brain_stroke_plot.png")
        if os.path.exists(plot_path):
            payload["result_image"] = _cache_busted_url(
                "/uploads/brain_stroke_output/brain_stroke_plot.png", plot_path
            )

        return payload
    except Exception as e:  # noqa: BLE001
        logger.exception("/api/brain_stroke/segment failed")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "task": normalized_task,
                "message": f"脑卒中检测执行异常：{e}",
            },
        )
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as remove_error:  # noqa: BLE001
            logger.warning(
                "[/api/brain_stroke/segment] 临时文件清理失败: %s", remove_error
            )


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

        response_data = process_query(validation_query, language=language, session_id=session_id)

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