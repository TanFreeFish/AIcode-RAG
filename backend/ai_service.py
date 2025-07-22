import json
import requests
from typing import Dict, Any, Optional

class AIService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get('model_type', 'ollama')
        self.model_name = config.get('model_name', 'qwen:7b')
        
    def generate_response(self, prompt: str, rag_context: Optional[str] = None) -> str:
        """生成AI回复，预留RAG接口"""
        full_prompt = self._build_prompt(prompt, rag_context)
        
        if self.model_type == 'ollama':
            return self._call_ollama(full_prompt)
        elif self.model_type == 'openai':
            return self._call_openai(full_prompt)
        # 可扩展其他API
        
    def _build_prompt(self, prompt: str, context: Optional[str]) -> str:
        """构建最终提示词，整合RAG内容"""
        if context:
            return f"<|im_start|>system\n你是一个AI助手，请基于以下上下文回答：\n{context}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        return f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    
    def _call_ollama(self, prompt: str) -> str:
        """调用本地Ollama服务"""
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload)
            return response.json().get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API（预留）"""
        # 实际使用时替换为真实API调用
        return f"OpenAI response to: {prompt}"
    
    def update_config(self, new_config: Dict[str, Any]):
        """动态更新配置"""
        self.config.update(new_config)
        self.model_type = self.config.get('model_type', self.model_type)
        self.model_name = self.config.get('model_name', self.model_name)

# RAG预留模块
class RAGService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def retrieve_context(self, query: str) -> str:
        """检索相关上下文（待实现）"""
        # TODO: 实现向量搜索等RAG功能
        return f"RAG context for: {query}"
