import json
import os
import re
import math
from typing import Union, Dict, List, Tuple


class OfflineFallbackAgent:
    """
    Lightweight local fallback responder.
    Guarantees a non-empty response without external network dependencies.
    """

    def __init__(self):
        self.urgent_keywords_zh = ["胸痛", "呼吸困难", "昏迷", "抽搐", "大出血", "中风", "急救", "意识不清"]
        self.urgent_keywords_en = ["chest pain", "shortness of breath", "seizure", "stroke", "heavy bleeding", "unconscious"]
        self.knowledge_entries = self._load_knowledge_entries()
        self.entry_tokens, self.idf = self._build_retrieval_index(self.knowledge_entries)

    def _load_knowledge_entries(self) -> List[Dict]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "data", "offline_knowledge_base.json")
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            if isinstance(entries, list):
                return entries
        except Exception:
            # Keep fallback responder available even if KB loading fails.
            return []
        return []

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower().strip()
        if not text:
            return []

        # English/number words
        word_tokens = re.findall(r"[a-z0-9]+", text)

        # Chinese bigrams (simple but effective for lightweight local retrieval)
        zh_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        zh_bigrams = []
        for i in range(len(zh_chars) - 1):
            zh_bigrams.append(zh_chars[i] + zh_chars[i + 1])

        return word_tokens + zh_bigrams

    def _build_retrieval_index(self, entries: List[Dict]) -> Tuple[List[List[str]], Dict[str, float]]:
        if not entries:
            return [], {}

        doc_tokens: List[List[str]] = []
        doc_freq: Dict[str, int] = {}

        for entry in entries:
            text_parts = []
            text_parts.extend(entry.get("keywords", []))
            text_parts.append(entry.get("zh_answer", ""))
            text_parts.append(entry.get("en_answer", ""))
            combined = " ".join([p for p in text_parts if p])
            tokens = self._tokenize(combined)
            doc_tokens.append(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        n_docs = len(entries)
        idf = {}
        for token, df in doc_freq.items():
            # BM25-style smoothed IDF
            idf[token] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        return doc_tokens, idf

    def _normalize_query(self, query: Union[str, Dict]) -> str:
        if isinstance(query, dict):
            text = query.get("text", "")
            has_image = "image" in query and bool(query.get("image"))
            if text:
                return text
            if has_image:
                return "用户上传了医学图像并请求分析"
            return ""
        return (query or "").strip()

    def _is_urgent(self, text: str) -> bool:
        lowered = text.lower()
        if any(keyword in text for keyword in self.urgent_keywords_zh):
            return True
        if any(keyword in lowered for keyword in self.urgent_keywords_en):
            return True
        return False

    def _detect_topic(self, text: str) -> str:
        lowered = text.lower()
        if any(x in text for x in ["发烧", "咳嗽", "喉咙痛"]) or any(x in lowered for x in ["fever", "cough", "sore throat"]):
            return "respiratory"
        if any(x in text for x in ["头痛", "偏头痛", "头晕"]) or any(x in lowered for x in ["headache", "migraine", "dizziness"]):
            return "headache"
        if any(x in text for x in ["血压", "高血压"]) or "blood pressure" in lowered:
            return "blood_pressure"
        if any(x in text for x in ["糖尿病", "血糖"]) or any(x in lowered for x in ["diabetes", "blood sugar"]):
            return "diabetes"
        if any(x in text for x in ["饮食", "减肥", "运动", "睡眠"]) or any(x in lowered for x in ["diet", "exercise", "sleep", "weight loss"]):
            return "lifestyle"
        return "general"

    def _score_entry(self, text: str, keywords: List[str]) -> int:
        if not keywords:
            return 0
        lowered = text.lower()
        score = 0
        for keyword in keywords:
            if not keyword:
                continue
            if keyword.lower() in lowered or keyword in text:
                score += 1
        return score

    def _semantic_score(self, query_tokens: List[str], entry_tokens: List[str]) -> float:
        if not query_tokens or not entry_tokens:
            return 0.0

        entry_token_set = set(entry_tokens)
        score = 0.0
        for token in query_tokens:
            if token in entry_token_set:
                score += self.idf.get(token, 1.0)
        return score

    def _retrieve_best_entries(self, text: str, top_k: int = 2) -> List[Tuple[int, Dict]]:
        scored: List[Tuple[float, Dict]] = []
        query_tokens = self._tokenize(text)

        for idx, entry in enumerate(self.knowledge_entries):
            keyword_score = float(self._score_entry(text, entry.get("keywords", [])))
            semantic_score = self._semantic_score(query_tokens, self.entry_tokens[idx] if idx < len(self.entry_tokens) else [])
            final_score = keyword_score * 2.0 + semantic_score
            if final_score > 0:
                scored.append((final_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _response_zh(self, text: str) -> str:
        if not text:
            return "我在离线模式下可继续服务。你可以描述症状（何时开始、严重程度、是否伴随发热/疼痛/呼吸困难），我会给出下一步建议。"

        if self._is_urgent(text):
            return (
                "你描述的情况可能涉及急症风险。请立即前往最近急诊或呼叫120，不要仅依赖线上建议。"
                "在等待救援时，尽量保持平躺或侧卧，避免独自行动。"
            )

        matched_entries = self._retrieve_best_entries(text, top_k=2)
        if matched_entries:
            lines = []
            for _, entry in matched_entries:
                answer = entry.get("zh_answer", "").strip()
                if answer:
                    lines.append(f"- {answer}")
            if lines:
                return (
                    "根据你的描述，我从本地离线知识库匹配到以下建议：\n"
                    + "\n".join(lines)
                    + "\n\n如果你愿意，我可以继续帮你按“当前症状、持续时间、危险信号、是否需要就医”做进一步分诊。"
                )

        topic = self._detect_topic(text)
        topic_guidance = {
            "respiratory": "如果是上呼吸道症状，先补液、休息、监测体温；若高热持续超过48小时、气促或血氧下降，请尽快线下就医。",
            "headache": "头痛可先休息、补水、避免强光和熬夜；若出现“最严重一次头痛”、神经功能异常、发热颈硬，需急诊评估。",
            "blood_pressure": "建议固定时间测量血压（静坐5分钟后，连续2-3次取平均）；若持续明显升高并伴不适，请就医调整方案。",
            "diabetes": "请规律监测空腹和餐后血糖，记录饮食与运动；若出现低血糖症状（心慌、出汗、手抖）需立即补糖并复测。",
            "lifestyle": "可从三点开始：规律作息、每周中等强度运动150分钟、减少高糖高盐加工食品，并记录变化便于复盘。",
            "general": "我已收到你的问题。当前离线模式下我会提供通用健康建议：先描述主要症状、持续时间、是否加重、既往病史和用药情况。"
        }
        guidance = topic_guidance.get(topic, topic_guidance["general"])
        return f"{guidance}\n\n如你愿意，我可以继续按“症状-可能原因-观察指标-就医时机”四步帮你细化。"

    def _response_en(self, text: str) -> str:
        if not text:
            return "I can keep helping in offline mode. Describe your symptoms (onset, severity, and associated signs), and I will provide next-step guidance."

        if self._is_urgent(text):
            return (
                "Your description may indicate an emergency. Please seek urgent care immediately (or call your local emergency number) "
                "instead of relying only on online advice."
            )

        matched_entries = self._retrieve_best_entries(text, top_k=2)
        if matched_entries:
            lines = []
            for _, entry in matched_entries:
                answer = entry.get("en_answer", "").strip()
                if answer:
                    lines.append(f"- {answer}")
            if lines:
                return (
                    "Based on your description, I matched the following guidance from the local offline knowledge base:\n"
                    + "\n".join(lines)
                    + "\n\nIf you want, I can further triage your situation by symptoms, duration, red flags, and when to seek in-person care."
                )

        return (
            "I am currently in offline fallback mode. I can still provide practical health guidance.\n\n"
            "Please share: main symptoms, duration, severity, relevant history, and current medications. "
            "I will then give step-by-step next actions and red flags."
        )

    def generate(self, query: Union[str, Dict], language: str = "en") -> str:
        text = self._normalize_query(query)
        lang = (language or "en").lower()
        if lang.startswith("zh"):
            return self._response_zh(text)
        return self._response_en(text)
