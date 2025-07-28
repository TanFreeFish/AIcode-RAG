import requests
from typing import Dict, Any, Optional
from RAG import initialize_rag_system
from config import SERVICE_CONFIG

class AIService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get('model_type', 'ollama')
        self.model_name = config.get('model_name', 'qwen:7b')
        self.ollama_host = SERVICE_CONFIG["ollama_host"]
        self.embedding_timeout = SERVICE_CONFIG["embedding_timeout"]
        self.rag_retriever = initialize_rag_system()
        
    def generate_response(self, prompt: str, use_rag: bool = False, use_rerank: bool = None) -> str:
        """生成AI回复，支持RAG和重排序"""
        rag_context = None
        if use_rag:
            rag_context = self.rag_retriever.retrieve(prompt, use_rerank=use_rerank)
        
        full_prompt = self._build_prompt(prompt, rag_context)
        
        if self.model_type == 'ollama':
            return self._call_ollama(full_prompt)
        elif self.model_type == 'openai':
            return self._call_openai(full_prompt)
        else:
            return f"Unsupported model type: {self.model_type}"
        
    def _build_prompt(self, prompt: str, context: Optional[str]) -> str:
        """构建最终提示词，整合RAG内容"""
        if context:
            return (
                f"<|im_start|>system\n"
                f"你是一个AI助手，请基于以下上下文信息回答问题。每个上下文片段包含摘要和详细内容：\n\n"
                f"{context}\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{prompt}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return (
            f"<|im_start|>user\n"
            f"{prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def _call_ollama(self, prompt: str) -> str:
        """调用本地Ollama服务"""
        try:
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=self.embedding_timeout)
            return response.json().get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API（预留）"""
        # 实际使用时替换为真实API调用
        return f"OpenAI response to: {prompt}"
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新服务配置"""
        self.config.update(new_config)
        self.model_type = self.config.get('model_type', self.model_type)
        self.model_name = self.config.get('model_name', self.model_name)
        # 重新初始化 Retriever
        self.rag_retriever = initialize_rag_system()