<div align="center">

<h1 align="center"><strong>⚕️ Multi-Agent Medical Assistant<h6 align="center">AI-powered multi-agentic system for medical diagnosis and assistance</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangGraph - Version](https://img.shields.io/badge/LangGraph-0.3+-teal?style=for-the-badge&logo=langgraph)
![LangChain - Version](https://img.shields.io/badge/LangChain-0.3+-teal?style=for-the-badge&logo=langchain)
![Qdrant Client - Version](https://img.shields.io/badge/Qdrant-1.13+-red?style=for-the-badge&logo=qdrant)
![FastAPI - Version](https://img.shields.io/badge/FastAPI-0.115+-teal?style=for-the-badge&logo=fastapi)
[![Generic badge](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://github.com/mobai828/agentgithub/blob/main/LICENSE) 

</div>

----

## 📌 Overview

The **Multi-Agent Medical Assistant** is an **AI-powered chatbot** designed to assist with **medical diagnosis, research, and patient interactions**. This project has been customized and optimized for lightweight deployment, rapid execution, and enhanced frontend interactions.

🚀 **Powered by Multi-Agent Intelligence**, this system integrates:  
- **🤖 Large Language Models (LLMs)** (Powered by Zhipu AI / GLM)
- **🖼️ Computer Vision Models** for medical imaging analysis (PyTorch)
- **📚 Retrieval-Augmented Generation (RAG)** leveraging Qdrant Cloud Vector Database
- **🌐 Real-time Web Search** for up-to-date medical insights (Powered by Tavily)
- **👨‍⚕️ Human-in-the-Loop Validation** to verify AI-based medical image diagnoses

### **Key Customizations in this Version** 📖  
🔹 **Lightweight PDF Parsing**: Replaced heavy enterprise parsers (Docling) with lightweight `PyPDFLoader` for ultra-fast RAG ingestion.  
🔹 **Dynamic Agent Routing**: Features an intuitive frontend where users can either rely on the LLM-based automatic routing or **manually force-select** specific agents (Plan B approach).  
🔹 **Enhanced Error Handling**: Robust frontend-backend communication that explicitly displays backend errors (like timeouts or validation errors) to the user via toast notifications and in-chat alerts.  
🔹 **Tavily Integration**: Seamless integration of Tavily API via environment variables for stable web search operations.  
🔹 **Bilingual Support**: Fully supports English and Chinese UI, including dynamically updating Agent Capability descriptions.

---

## ✨ Key Features

- 🤖 **Multi-Agent Architecture**: Specialized agents working in harmony:
  - **Medical Conversation Agent**: General health queries and greetings.
  - **Medical RAG Agent**: Deep-dives into uploaded medical literature.
  - **Web Search Agent**: Fetches the latest medical news and research.
  - **Computer Vision Agents**: Analyzes Chest X-rays and Skin Lesions.

- 🔍 **Advanced Agentic RAG Retrieval System**:
  - Lightweight PDF parsing using `PyPDFLoader`.
  - LLM-based semantic chunking with `RecursiveCharacterTextSplitter`.
  - Cloud-based Qdrant Vector Database hybrid search.
  - Input-output guardrails for safe responses.

- 🏥 **Medical Imaging Analysis**:
  - Chest X-ray Disease Classification.
  - Skin Lesion Segmentation.
  - *Note: Brain Tumor Detection is a placeholder for future implementation.*

- 🌐 **Real-time Research Integration**: Web search agent retrieves the latest medical research papers and findings using Tavily.

- 👩‍⚕️ **Expert Oversight System**: Human-in-the-loop verification by medical professionals before finalizing Computer Vision outputs.

- 💻 **Intuitive User Interface**: Designed for healthcare professionals with minimal technical expertise. Features interactive agent selection and toast notifications.

---

## 🛠️ Technology Stack

| Component | Technologies |
|-----------|-------------|
| 🔹 **Backend Framework** | FastAPI |
| 🔹 **Agent Orchestration** | LangGraph, LangChain |
| 🔹 **LLM Provider** | Zhipu AI (GLM-4) |
| 🔹 **Knowledge Storage** | Qdrant Cloud Vector Database |
| 🔹 **Web Search** | Tavily API |
| 🔹 **Medical Imaging** | PyTorch (Torchvision), OpenCV (Headless) |
| 🔹 **Frontend** | HTML, CSS, JavaScript, Bootstrap 5 |

---

## 🚀 Installation & Setup

### Prerequisites:
- Python 3.11+
- API keys for Zhipu AI, Tavily, and Qdrant Cloud.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mobai828/agentgithub.git
cd agentgithub
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure API Keys
Edit `config.py` to ensure your API keys are correctly set, or export them as environment variables:
- `ZHIPU_API_KEY`
- `TAVILY_API_KEY`
- Qdrant Cloud URL & API Key

### 4️⃣ Run the Application
```bash
python app.py
```
The application will start on `http://localhost:8000`.

### 5️⃣ Data Ingestion (Optional)
To ingest your own medical PDFs into the RAG system:
Place your PDF files in the `data/raw` directory and run:
```bash
python ingest_rag_data.py --dir data/raw
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**. 

You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that you meet the conditions of the Apache 2.0 License.

See the [`LICENSE`](LICENSE) file for the full text.

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mobai828/agentgithub/issues).
