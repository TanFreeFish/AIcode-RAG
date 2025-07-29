import requests
from typing import Dict, Any, Optional
from RAG import initialize_rag_system
from config import SERVICE_CONFIG

class AIService:
    """
    @brief AI服务类，负责处理与AI模型的交互，支持多种模型类型和RAG功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        @brief 初始化AI服务
        @param config 配置字典，包含模型类型和名称等配置
        """
        self.config = config
        self.model_type = config.get('model_type', 'ollama')
        self.model_name = config.get('model_name', 'qwen:7b')
        self.ollama_host = SERVICE_CONFIG["ollama_host"]
        self.embedding_timeout = SERVICE_CONFIG["embedding_timeout"]
        self.rag_retriever = initialize_rag_system()
        
    def generate_response(self, prompt: str, use_rag: bool = False, use_rerank: bool = None) -> str:
        """
        @brief 生成AI回复，支持RAG和重排序
        @param prompt 用户输入的提示词
        @param use_rag 是否使用RAG检索增强生成
        @param use_rerank 是否使用重排序
        @return AI生成的回复内容
        """
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
        """
        @brief 构建最终提示词，整合RAG内容
        @param prompt 原始用户提示词
        @param context RAG检索到的上下文信息
        @return 构建完成的提示词
        """
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
        """
        @brief 调用本地Ollama服务
        @param prompt 完整的提示词
        @return AI模型的回复内容
        """
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
        """
        @brief 调用OpenAI API（预留）
        @param prompt 完整的提示词
        @return AI模型的回复内容
        """
        # 实际使用时替换为真实API调用
        return f"OpenAI response to: {prompt}"
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        @brief 更新服务配置
        @param new_config 新的配置字典
        """
        self.config.update(new_config)
        self.model_type = self.config.get('model_type', self.model_type)
        self.model_name = self.config.get('model_name', self.model_name)
        # 重新初始化 Retriever
        self.rag_retriever = initialize_rag_system()