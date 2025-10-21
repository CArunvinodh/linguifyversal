# app.py - FastAPI version
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformer.app import AcademicTextHumanizer, NLP_GLOBAL, download_nltk_resources
from nltk.tokenize import word_tokenize
import os

download_nltk_resources()

app = FastAPI(title="Linguify")

# CORS middleware
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
    if not text.strip():
        return "⚠️ Please enter text to begin refinement."
    
    # Strict limits for serverless
    if len(text) > 10000:
        return "⚠️ Text too long! Maximum 10,000 characters allowed."
    
    try:
        input_word_count = len(word_tokenize(text, language='english', preserve_line=True))
        if input_word_count > 2000:
            return "⚠️ Text too large! Maximum 2,000 words allowed."
            
        humanizer = AcademicTextHumanizer(
            p_passive=0.3,
            p_synonym_replacement=0.3,
            p_academic_transition=0.4
        )
        
        transformed = humanizer.humanize_text(text, use_passive, use_synonyms)
        output_word_count = len(word_tokenize(transformed, language='english', preserve_line=True))
        
        return f"""
📊 **Text Statistics:**
- **Input:** {input_word_count} words
- **Output:** {output_word_count} words

🎓 **Refined Text:**
{transformed}
"""
    except Exception as e:
        return f"❌ Error processing text: {str(e)}"

@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    try:
        result = humanize_text_safe(request.text, request.use_passive, request.use_synonyms)
        return ProcessResponse(result=result)
    except Exception as e:
        return ProcessResponse(result="", error=str(e))

# Simple HTML frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Linguify 🪶</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; background: linear-gradient(90deg, #0072ff, #00c6ff); color: white; padding: 2rem; border-radius: 16px; }
        .output { background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 1rem; }
        textarea { width: 100%; height: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #0072ff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .loading { display: none; color: #0072ff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Linguify 🪶</h1>
        <p>Refine and Humanize AI-generated content into polished academic writing</p>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px; margin-top: 2rem;">
        <div>
            <h3>⚙️ Linguify Options</h3>
            <label><input type="checkbox" id="passive"> Convert to Passive Voice</label><br>
            <label><input type="checkbox" id="synonyms"> Use Formal Synonyms</label>
            <p><small>Max: 10,000 characters, 2,000 words</small></p>
        </div>
        
        <div>
            <h3>📝 Input Text</h3>
            <textarea id="inputText" placeholder="Type or paste your text here..."></textarea>
            <button onclick="processText()">✨ Refine with Linguify</button>
            <div id="loading" class="loading">Processing...</div>
            
            <h3>🎓 Refined Output</h3>
            <div id="output" class="output">Results will appear here...</div>
        </div>
    </div>

    <script>
        async function processText() {
            const text = document.getElementById('inputText').value;
            const passive = document.getElementById('passive').checked;
            const synonyms = document.getElementById('synonyms').checked;
            const loading = document.getElementById('loading');
            const output = document.getElementById('output');
            
            loading.style.display = 'block';
            output.innerHTML = 'Processing...';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, use_passive: passive, use_synonyms: synonyms })
                });
                
                const data = await response.json();
                output.innerHTML = data.result || data.error;
            } catch (error) {
                output.innerHTML = '❌ Error connecting to server';
            } finally {
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
