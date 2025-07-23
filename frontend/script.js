let chatHistory = [];

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    const useRag = document.getElementById('use-rag').checked;
    
    if (!message) return;
    
    // 添加用户消息
    addMessage('user', message);
    input.value = '';
    
    try {
        // 发送到后端
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                use_rag: useRag
            })
        });
        
        const data = await response.json();
        addMessage('ai', data.response);
    } catch (error) {
        addMessage('ai', `Error: ${error.message}`);
    }
}

async function updateConfig() {
    const modelType = document.getElementById('model-type').value;
    const modelName = document.getElementById('model-name').value;
    
    try {
        await fetch('http://localhost:8000/update_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_type: modelType,
                model_name: modelName
            })
        });
        alert('Configuration updated successfully!');
    } catch (error) {
        alert(`Update failed: ${error.message}`);
    }
}
async function buildEmbeddings() {
    const statusDiv = document.getElementById('embedding-status');
    statusDiv.textContent = "正在生成向量嵌入...这可能需要一段时间";
    
    try {
        const response = await fetch('http://localhost:8000/build_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        statusDiv.textContent = data.status;
    } catch (error) {
        statusDiv.textContent = `生成失败: ${error.message}`;
    }
}
async function uploadDocument() {
    const fileInput = document.getElementById('document-file');
    const file = fileInput.files[0];
    const statusDiv = document.getElementById('upload-status');
    
    if (!file) {
        statusDiv.textContent = "请选择文件";
        return;
    }
    
    statusDiv.textContent = "上传中...";
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('http://localhost:8000/upload_document', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            statusDiv.textContent = `上传成功: ${data.file_path}`;
            // 清空文件输入
            fileInput.value = '';
        } else {
            statusDiv.textContent = `上传失败: ${data.detail || '未知错误'}`;
        }
    } catch (error) {
        statusDiv.textContent = `上传错误: ${error.message}`;
    }
}

// 添加页面加载事件
document.addEventListener('DOMContentLoaded', () => {
    // 初始加载完成后滚动到底部
    const chatHistory = document.getElementById('chat-history');
    chatHistory.scrollTop = chatHistory.scrollHeight;
});
function addMessage(role, content) {
    const chatHistoryElement = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'ai-message-content';
    
    // 安全渲染Markdown
    if (role === 'ai') {
        const sanitized = DOMPurify.sanitize(marked.parse(content));
        contentDiv.innerHTML = sanitized;
    } else {
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(contentDiv);
    chatHistoryElement.appendChild(messageDiv);
    
    // 滚动到底部
    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
    
    // 保存历史
    chatHistory.push({ role, content });
}

// 输入框回车发送
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});