import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAG_CONFIG

class TextSplitter:
    def __init__(self):
        config = RAG_CONFIG["text_splitter"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", "。", "？", "！", "；", " ", ""],
            keep_separator=True
        )
    
    def split_documents(self, documents):
        """分割文档为文本块"""
        chunks = []
        for doc in documents:
            content = doc["content"]
            # 清理多余空白
            content = re.sub(r'\s+', ' ', content).strip()
            # 分割文本
            split_texts = self.splitter.split_text(content)
            
            for i, text in enumerate(split_texts):
                chunks.append({
                    "text": text,
                    "source": doc["file_path"],
                    "chunk_id": f"{Path(doc['file_path']).stem}_{i}"
                })
        
        return chunks