"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            "Please set it in your local .env file."
        )
    return value

# Load environment variables from .env file
load_dotenv()

ZHIPU_API_KEY = _required_env("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.1  # Deterministic
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.7  # Creative but factual
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.3  # Slightly creative but factual
        )
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 2048  # embedding-3 default dimension is 2048
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = False  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL", "").strip()
        self.api_key = os.getenv("QDRANT_API_KEY", "").strip()
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Modify based on documents and performance
        self.chunk_overlap = 50  # Modify based on documents and performance
        # Initialize OpenAI Embeddings for Zhipu
        self.embedding_model = OpenAIEmbeddings(
            model="embedding-3",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL
        )
        self.llm = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.3  # Slightly creative but factual
        )
        self.summarizer_model = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.5  # Slightly creative but factual
        )
        self.chunker_model = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.0  # factual
        )
        self.response_generator_model = ChatOpenAI(
            model="glm-4-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.3  # Slightly creative but factual
        )
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.brain_stroke_model_path = "./agents/image_analysis_agent/brain_stroke_agent/models/brain_stroke_segmentation.pth"
        # Reserved-interface CV agents output paths
        self.brain_tumor_output_dir = "./uploads/brain_tumor_output"
        self.brain_tumor_output_path = "./uploads/brain_tumor_output/brain_tumor_plot.png"
        self.brain_stroke_output_dir = "./uploads/brain_stroke_output"
        self.brain_stroke_output_path = "./uploads/brain_stroke_output/brain_stroke_plot.png"

        # ---- heyi-Trans-master 通用视觉模型相关配置 ----
        # 输入分辨率（ViT-B/16 默认 224；patch_size=16，需能被整除）
        self.heyi_image_size = 224
        # 运行设备："cuda" / "cpu" / None（None 时自动检测）
        self.heyi_device = None
        # 骨干网络名称（对应 heyi ViTEncoder 支持的模型）
        self.heyi_backbone = "vit_b_16"
        # 是否允许使用 ImageNet 预训练 backbone（无微调权重时作为演示降级）
        self.heyi_allow_pretrained_fallback = True

        # ---- Heyi 远程分割服务（同学部署的医疗图像分割 API v2.0）----
        # 远程服务根地址；为空字符串则视为未配置，直接走本地 adapter。
        # 推荐值见 .env.example，请通过 .env 配置而非在此硬编码。
        self.heyi_remote_url = os.getenv("HEYI_REMOTE_URL", "").strip()
        # 总开关：关闭后无论远程是否可达，都只走本地 adapter。
        self.heyi_remote_enabled = _to_bool(
            os.getenv("HEYI_REMOTE_ENABLED"), default=True
        )
        # /segment 总超时（秒）；3D NIfTI 推理耗时较长，默认给 120s。
        self.heyi_remote_timeout = float(os.getenv("HEYI_REMOTE_TIMEOUT", "120"))
        # /health 探活超时（秒）；必须短，否则每次请求都会卡 2 分钟。
        self.heyi_remote_health_timeout = float(
            os.getenv("HEYI_REMOTE_HEALTH_TIMEOUT", "3")
        )
        # 默认 task 参数：auto / hemorrhage / ischemia。
        self.heyi_remote_task = os.getenv("HEYI_REMOTE_TASK", "auto").strip() or "auto"

        self.llm = ChatOpenAI(
            model="glm-4v-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.1  # Keep deterministic for classification tasks
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)
        self.baidu_api_key = os.getenv("BAIDU_API_KEY", "").strip()
        self.baidu_secret_key = os.getenv("BAIDU_SECRET_KEY", "").strip()

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "BRAIN_STROKE_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB
        # When enabled, all requests use local offline fallback responder only.
        self.force_offline_mode = _to_bool(os.getenv("FORCE_OFFLINE_MODE"), default=False)
        # When enabled, failures in online pipeline will always fallback locally.
        self.enable_offline_fallback = _to_bool(os.getenv("ENABLE_OFFLINE_FALLBACK"), default=True)

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = _required_env("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()