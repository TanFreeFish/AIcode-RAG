import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAG_CONFIG
from pathlib import Path
import logging
import requests
import time
from tqdm import tqdm  
from typing import List

logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(self, progress_callback=None):
        """
        @brief 初始化文本分割器
        
        @param progress_callback (function, optional): 进度回调函数
        """
        config = RAG_CONFIG["text_splitter"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", "。", "？", "！", "；", " ", ""],
            keep_separator=True
        )
        self.summarizer_config = RAG_CONFIG.get("summarizer", {})
        self.progress_callback = progress_callback or (lambda **kw: None)
    
    def generate_summary(self, text: str) -> str:
        """
        @brief 调用Qwen大语言模型为输入文本生成简洁的短语级摘要
        
        @param text (str): 需要生成摘要的输入文本
        
        @return str: 生成的摘要文本，失败时返回前5个词的组合
        """
        try:
            prompt = f"请用5-10个字的短语总结以下文本的核心内容，不要解释，只输出短语：\n{text}"
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.summarizer_config.get("model_name", "qwen:7b"),
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip().replace('"', '')
            else:
                logger.warning(f"摘要生成失败: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"摘要生成错误: {str(e)}")
        
        
        return " ".join(text.split()[:5])
    
    
    def _smart_split(self, text: str) -> List[str]:
        """
        @brief 将输入文本按配置的块大小进行分割，同时尽量保持句子完整性
        
        @param text (str): 需要分割的输入文本
        
        @return List[str]: 分割后的文本块列表
        """
        
        sentence_endings = {'。', '？', '！', '；', '.', '?', '!', ';'}
        
        chunks = []
        current_chunk = ""
        char_count = 0
        
        
        for char in text:
            current_chunk += char
            char_count += 1
            
            
            if char_count >= self.splitter._chunk_size * 0.7 and char in sentence_endings:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                char_count = 0
                
            
            elif char_count >= self.splitter._chunk_size:
                
                for i in range(len(current_chunk)-1, -1, -1):
                    if current_chunk[i] in sentence_endings:
                        chunks.append(current_chunk[:i+1].strip())
                        current_chunk = current_chunk[i+1:]
                        char_count = len(current_chunk)
                        break
                else:  
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    char_count = 0
        
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_documents(self, documents):
        """
        @brief 将加载的文档列表分割为较小的文本块，并为每个块生成摘要
        
        @param documents (list): 文档列表，每个元素包含文件路径和内容
        
        @return list: 分割后的文本块列表，每个元素包含文本、摘要等信息
        """
        chunks = []
        total_docs = len(documents)
        
        
        self.progress_callback(stage="split", total=total_docs, current=0, message="开始分割文档")
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="分割文档")):
            content = doc["content"]
            
            content = re.sub(r'\s+', ' ', content).strip()
            
            split_texts = self._smart_split(content)
            
            
            self.progress_callback(
                stage="split",
                current=doc_idx + 1,
                total=total_docs,
                message=f"正在处理文档: {Path(doc['file_path']).name}",
                details=f"分割成 {len(split_texts)} 个片段"
            )
            
            for i, text in enumerate(split_texts):
                
                summary = self.generate_summary(text)
                
                chunks.append({
                    "text": text,
                    "summary": summary,
                    "source": doc["file_path"],
                    "chunk_id": f"{Path(doc['file_path']).stem}_{i}"
                })
        
        
        self.progress_callback(
            stage="split",
            current=total_docs,
            total=total_docs,
            message=f"文档分割完成",
            details=f"共生成 {len(chunks)} 个文本块"
        )
        
        return chunks