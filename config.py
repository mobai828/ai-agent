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
# Load environment variables from .env file
load_dotenv()

ZHIPU_API_KEY = "690a8432895d43b29a3c4259556ed696.cX6szzSrapTt5USb"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

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
        self.url = "https://ca42b516-972f-4d16-8ee3-9d9431766498.us-west-2-0.aws.cloud.qdrant.io:6333"
        self.api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjVjNmY0YjYtZDgwOS00OWE5LWJlZmQtZDUzM2ZhNTgyNWI5In0.-vBLt9If2t4F-iL4eagx3gwb7I7YjQPNuzyRDuwHH04"
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
        self.llm = ChatOpenAI(
            model="glm-4v-flash",
            api_key=ZHIPU_API_KEY,
            base_url=ZHIPU_BASE_URL,
            temperature = 0.1  # Keep deterministic for classification tasks
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)
        self.baidu_api_key = "OSP8KQ8n1DN5cGYE0WDOqSTt"
        self.baidu_secret_key = "31sK7hNGN9oZ8zWC739AA6prPuiU2KGH"

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
        self.tavily_api_key = "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"
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()