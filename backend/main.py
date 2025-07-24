# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from ai_service import AIService
import os
from RAG.document_loader import DocumentLoader
from fastapi.responses import FileResponse
from RAG import initialize_rag_system, build_vector_store  # 直接导入build_vector_store

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI()
# 添加静态文件服务
    
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
ai_config = {"model_type": "ollama", "model_name": "qwen:7b"}
ai_service = AIService(ai_config)

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = ai_service.generate_response(request.message, request.use_rag)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_config")
async def update_config(new_config: dict):
    ai_service.update_config(new_config)
    return {"status": "config updated"}

# 重建索引端点
@app.post("/rebuild_index")
async def rebuild_index():
    # 直接调用构建函数
    success = build_vector_store()
    if success:
        # 重建完成后更新检索器
        ai_service.rag_retriever = initialize_rag_system()
        return {"status": "index rebuilt successfully"}
    else:
        return {"status": "failed to rebuild index", "error": "no documents or chunks found"}

# 上传文档端点
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        documents_dir = BASE_DIR / "data" / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 自动触发重建索引
        success = build_vector_store()
        if success:
            # 重建完成后更新检索器
            ai_service.rag_retriever = initialize_rag_system()
            return {"status": "success", "file_path": str(file_path)}
        else:
            return {"status": "file uploaded but failed to rebuild index"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 手动构建嵌入端点
@app.post("/build_embeddings")
async def build_embeddings():
    """手动触发向量嵌入过程"""
    success = build_vector_store()
    if success:
        # 重建完成后更新检索器
        ai_service.rag_retriever = initialize_rag_system()
        return {"status": "embeddings built successfully"}
    else:
        return {"status": "failed to build embeddings", "error": "no documents or chunks found"}
    
# 添加前端服务
@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)