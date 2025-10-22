# ===== requirements.txt =====
# Minimal, lightweight dependencies only
fastapi==0.115.0
uvicorn==0.30.0
nltk==3.8.1
# Remove these heavy dependencies:
# spacy  # ~500MB with models
# sentence-transformers  # ~400MB
# en-core-web-sm  # ~13MB but requires spacy

# ===== vercel.json =====
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}

# ===== .vercelignore =====
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.git/
.github/
*.so
*.egg
*.egg-info/
dist/
build/
.DS_Store
node_modules/
.pytest_cache/
.mypy_cache/
nltk_data/en_core_web_sm/
models/

# ===== app.py - Lightweight Version =====
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformer.app_lite import AcademicTextHumanizer, download_nltk_resources
from nltk.tokenize import word_tokenize
import os

download_nltk_resources()

app = FastAPI(title="Linguify")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    text: str
    use_passive: bool = False
    use_synonyms: bool = False

class ProcessResponse(BaseModel):
    result: str
    error: str = None

def humanize_text_safe(text, use_passive, use_synonyms):
    """Safe text processing with limits"""
    if not text or not text.strip():
        return "⚠️ Please enter text to begin refinement."
    
    text = text.strip()
    
    if len(text) > 5000:
        return f"⚠️ Text too long! Maximum 5,000 characters allowed."
    
    try:
        input_word_count = len(word_tokenize(text, language='english', preserve_line=True))
        
        if input_word_count > 1000:
            return f"⚠️ Text too large! Maximum 1,000 words allowed."
        
        humanizer = AcademicTextHumanizer(
            p_passive=0.2,
            p_synonym_replacement=0.2,
            p_academic_transition=0.3
        )
        
        transformed = humanizer.humanize_text(text, use_passive, use_synonyms)
        
        if not transformed:
            return "❌ Processing returned empty result."
        
        if len(transformed) > 10000:
            transformed = transformed[:10000] + "\n\n⚠️ [Output truncated]"
        
        output_word_count = len(word_tokenize(transformed, language='english', preserve_line=True))
        
        return f"""📊 **Text Statistics:**
- **Input:** {input_word_count} words
- **Output:** {output_word_count} words

🎓 **Refined Text:**

{transformed}
"""
    except Exception as e:
        return f"❌ Error: {str(e)[:200]}"

@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    try:
        result = humanize_text_safe(request.text, request.use_passive, request.use_synonyms)
        return ProcessResponse(result=result)
    except Exception as e:
        return ProcessResponse(result="", error=str(e)[:200])

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Linguify 🪶</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        .header h1 { font-size: 3rem; margin-bottom: 0.5rem; }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
        }
        .options {
            display: flex;
            gap: 2rem;
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            font-weight: 500;
        }
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        textarea {
            width: 100%;
            min-height: 200px;
            padding: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1rem;
            resize: vertical;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .char-count {
            text-align: right;
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        button {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 1rem;
            transition: transform 0.2s;
        }
        button:hover:not(:disabled) { transform: translateY(-2px); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .output {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            min-height: 150px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 1rem;
            color: #667eea;
            font-weight: 600;
        }
        .note {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.9rem;
        }
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .options { flex-direction: column; gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🪶 Linguify</h1>
            <p>Transform AI text into polished academic writing</p>
        </div>

        <div class="card">
            <h2>⚙️ Transformation Options</h2>
            <div class="options">
                <label>
                    <input type="checkbox" id="passive">
                    Convert to Passive Voice
                </label>
                <label>
                    <input type="checkbox" id="synonyms">
                    Use Formal Synonyms
                </label>
            </div>

            <h3>📝 Input Text</h3>
            <textarea 
                id="inputText" 
                placeholder="Paste your text here (max 5,000 characters, 1,000 words)..."
                maxlength="5000"
            ></textarea>
            <div class="char-count">
                <span id="charCount">0</span> / 5,000 characters
            </div>

            <button id="processBtn" onclick="processText()">
                ✨ Transform with Linguify
            </button>
            <div id="loading" class="loading">⏳ Processing...</div>

            <div class="note">
                ⚡ <strong>Lightweight Mode:</strong> Running without heavy ML models for faster serverless deployment
            </div>
        </div>

        <div class="card">
            <h3>🎓 Refined Output</h3>
            <div id="output" class="output">Results will appear here...</div>
        </div>
    </div>

    <script>
        const inputText = document.getElementById('inputText');
        const charCount = document.getElementById('charCount');
        const processBtn = document.getElementById('processBtn');
        
        inputText.addEventListener('input', () => {
            const length = inputText.value.length;
            charCount.textContent = length.toLocaleString();
            charCount.style.color = length > 4500 ? '#dc3545' : '#666';
        });
        
        async function processText() {
            const text = inputText.value.trim();
            const passive = document.getElementById('passive').checked;
            const synonyms = document.getElementById('synonyms').checked;
            const loading = document.getElementById('loading');
            const output = document.getElementById('output');
            
            if (!text) {
                output.innerHTML = '⚠️ Please enter some text first.';
                return;
            }
            
            processBtn.disabled = true;
            loading.style.display = 'block';
            output.innerHTML = '⏳ Processing...';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text, 
                        use_passive: passive, 
                        use_synonyms: synonyms 
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                output.innerHTML = data.error || data.result;
                
            } catch (error) {
                output.innerHTML = `❌ Error: ${error.message}`;
            } finally {
                processBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        inputText.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                processText();
            }
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
