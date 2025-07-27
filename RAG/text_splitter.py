import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAG_CONFIG
from pathlib import Path
import logging
import requests
import time
from tqdm import tqdm  # 导入进度条库
from typing import List

logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(self, progress_callback=None):
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
        """使用Qwen模型生成短语级摘要"""
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
        
        # 失败时回退：使用前N个词
        return " ".join(text.split()[:5])
    # 在 TextSplitter 类中添加智能分割方法
    def _smart_split(self, text: str) -> List[str]:
        """改进的分割逻辑，确保句子完整性"""
        # 定义中文句子结束符
        sentence_endings = {'。', '？', '！', '；', '.', '?', '!', ';'}
        
        chunks = []
        current_chunk = ""
        char_count = 0
        
        # 按字符迭代，保持句子完整性
        for char in text:
            current_chunk += char
            char_count += 1
            
            # 达到最小分割长度且遇到句子结束符
            if char_count >= self.splitter._chunk_size * 0.7 and char in sentence_endings:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                char_count = 0
                
            # 达到最大长度强制分割（避免过长）
            elif char_count >= self.splitter._chunk_size:
                # 寻找最近的句子边界
                for i in range(len(current_chunk)-1, -1, -1):
                    if current_chunk[i] in sentence_endings:
                        chunks.append(current_chunk[:i+1].strip())
                        current_chunk = current_chunk[i+1:]
                        char_count = len(current_chunk)
                        break
                else:  # 没有找到边界则硬分割
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    char_count = 0
        
        # 添加剩余内容
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    def split_documents(self, documents):
        """分割文档为文本块并生成摘要"""
        chunks = []
        total_docs = len(documents)
        
        # 添加进度回调
        self.progress_callback(stage="split", total=total_docs, current=0, message="开始分割文档")
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="分割文档")):
            content = doc["content"]
            # 清理多余空白
            content = re.sub(r'\s+', ' ', content).strip()
            # 分割文本
            split_texts = self._smart_split(content)
            
            # 更新进度
            self.progress_callback(
                stage="split",
                current=doc_idx + 1,
                total=total_docs,
                message=f"正在处理文档: {Path(doc['file_path']).name}",
                details=f"分割成 {len(split_texts)} 个片段"
            )
            
            for i, text in enumerate(split_texts):
                # 生成语义摘要
                summary = self.generate_summary(text)
                
                chunks.append({
                    "text": text,
                    "summary": summary,
                    "source": doc["file_path"],
                    "chunk_id": f"{Path(doc['file_path']).stem}_{i}"
                })
        
        # 完成进度
        self.progress_callback(
            stage="split",
            current=total_docs,
            total=total_docs,
            message=f"文档分割完成",
            details=f"共生成 {len(chunks)} 个文本块"
        )
        
        return chunks