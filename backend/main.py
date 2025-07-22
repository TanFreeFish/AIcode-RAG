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
from RAG import initialize_rag_system

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

# main.py中新增
@app.post("/rebuild_index")
async def rebuild_index():
    ai_service.rag_retriever = initialize_rag_system(force_rebuild=True)
    return {"status": "index rebuilt"}
# 修改上传文档端点

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        documents_dir = BASE_DIR / "data" / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        # 重新加载向量存储
        ai_service.rag_retriever.vector_store.load_index()
        return {"status": "success", "file_path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 添加前端服务
@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "frontend" / "index.html")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
     