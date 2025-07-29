import os
from pathlib import Path
from config import DOCUMENTS_DIR, RAG_CONFIG
import PyPDF2
from docx import Document
import markdown
import re
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        """
        @brief 初始化文档加载器
        """
        self.extensions = RAG_CONFIG["document_loader"]["extensions"]
        self.documents_dir = Path(DOCUMENTS_DIR)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
    
    def load_documents(self):
        """
        
        @brief 从配置的文档目录中查找并加载所有支持格式的文档文件
        
        @return list: 文档列表，每个元素包含文件路径和内容的字典
        """
        documents = []
        
        for ext in self.extensions:
            for file_path in self.documents_dir.glob(f"*{ext}"):
                content = self._load_file(file_path)
                if content:
                    documents.append({
                        "file_path": str(file_path),
                        "content": content
                    })
                    logger.info(f"Loaded document: {file_path.name}")
        
        return documents
    
    def _load_file(self, file_path):
        """
        @brief 根据文件扩展名识别文件类型并使用相应方法加载文件内容
        
        @param file_path (Path): 需要加载的文件路径对象
        
        @return str: 文件内容字符串，加载失败时返回None
        """
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif ext == ".pdf":
                content = []
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            content.append(text)
                return "\n".join(content)
            
            elif ext == ".json":  
                with open(file_path, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                    return str(data)  
            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            
            elif ext == ".md":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return markdown.markdown(f.read())
            
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None

    def add_document(self, file_path):
        """
        @brief 将指定的文档文件复制到文档存储目录中
        
        @param file_path (str): 源文件的完整路径
        
        @return str: 目标文件路径字符串，失败时返回None
        """
        dest_path = self.documents_dir / Path(file_path).name
        try:
            with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            logger.info(f"Added document: {dest_path}")
            return str(dest_path)
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            return None