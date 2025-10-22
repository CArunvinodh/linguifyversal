# app.py - FastAPI version with strict limits
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformer.app import AcademicTextHumanizer, NLP_GLOBAL, download_nltk_resources
from nltk.tokenize import word_tokenize
import os
import sys

download_nltk_resources()

app = FastAPI(title="Linguify")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strict limits for serverless
MAX_INPUT_CHARS = 5000  # Reduced from 10000
MAX_INPUT_WORDS = 1000  # Reduced from 2000
MAX_OUTPUT_CHARS = 10000  # Maximum output size

class ProcessRequest(BaseModel):
    text: str
    use_passive: bool = False
    use_synonyms: bool = False

class ProcessResponse(BaseModel):
    result: str
    error: str = None

def humanize_text_safe(text, use_passive, use_synonyms):
    """Safe text processing with strict limits"""
    if not text or not text.strip():
        return "⚠️ Please enter text to begin refinement."
    
    text = text.strip()
    
    # Strict input validation
    if len(text) > MAX_INPUT_CHARS:
        return f"⚠️ Text too long! Maximum {MAX_INPUT_CHARS:,} characters allowed. You entered {len(text):,} characters."
    
    try:
        # Tokenize and check word count
        input_word_count = len(word_tokenize(text, language='english', preserve_line=True))
        
        if input_word_count > MAX_INPUT_WORDS:
            return f"⚠️ Text too large! Maximum {MAX_INPUT_WORDS:,} words allowed. You entered {input_word_count:,} words."
        
        # Create humanizer with conservative settings
        humanizer = AcademicTextHumanizer(
            p_passive=0.2,  # Reduced from 0.3
            p_synonym_replacement=0.2,  # Reduced from 0.3
            p_academic_transition=0.3  # Reduced from 0.4
        )
        
        # Process text
        transformed = humanizer.humanize_text(text, use_passive, use_synonyms)
        
        # Validate output size
        if not transformed:
            return "❌ Processing returned empty result."
        
        if len(transformed) > MAX_OUTPUT_CHARS:
            transformed = transformed[:MAX_OUTPUT_CHARS] + "\n\n⚠️ [Output truncated due to size limits]"
        
        output_word_count = len(word_tokenize(transformed, language='english', preserve_line=True))
        
        # Format response with size check
        result = f"""📊 **Text Statistics:**
- **Input:** {input_word_count} words ({len(text)} chars)
- **Output:** {output_word_count} words ({len(transformed)} chars)

🎓 **Refined Text:**

{transformed}
"""
        
        # Final size check
        if len(result) > MAX_OUTPUT_CHARS:
            return "❌ Output too large. Please try with shorter text."
        
        return result
        
    except MemoryError:
        return "❌ Out of memory. Please try with shorter text."
    except Exception as e:
        error_msg = str(e)
        # Limit error message size
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."
        return f"❌ Error processing text: {error_msg}"

@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    """Process text with strict response size limits"""
    try:
        # Validate input
        if not request.text:
            return ProcessResponse(result="⚠️ No text provided.")
        
        # Process with limits
        result = humanize_text_safe(
            request.text, 
            request.use_passive, 
            request.use_synonyms
        )
        
        # Final size validation
        if sys.getsizeof(result) > 1_000_000:  # 1MB limit
            return ProcessResponse(
                result="",
                error="Response too large. Please try with shorter text."
            )
        
        return ProcessResponse(result=result)
        
    except Exception as e:
        error_msg = str(e)[:200]  # Limit error message
        return ProcessResponse(result="", error=f"Server error: {error_msg}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

# Simple HTML frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Linguify 🪶</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f8fafc;
        }
        .header { 
            text-align: center; 
            background: linear-gradient(135deg, #0072ff, #00c6ff); 
            color: white; 
            padding: 2rem; 
            border-radius: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .container {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
            margin-top: 2rem;
        }
        .sidebar {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: fit-content;
        }
        .main {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .output { 
            background: #f8fafc; 
            padding: 1.5rem; 
            border-radius: 8px; 
            border: 1px solid #e2e8f0; 
            margin-top: 1rem;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 500px;
            overflow-y: auto;
        }
        textarea { 
            width: 100%; 
            height: 250px; 
            padding: 12px; 
            border: 2px solid #e2e8f0; 
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #0072ff;
        }
        button { 
            background: linear-gradient(135deg, #0072ff, #00c6ff);
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 8px; 
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            margin-top: 10px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .loading { 
            display: none; 
            color: #0072ff;
            text-align: center;
            padding: 10px;
            font-weight: 600;
        }
        label {
            display: flex;
            align-items: center;
            margin: 12px 0;
            cursor: pointer;
        }
        input[type="checkbox"] {
            margin-right: 8px;
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .limits {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 6px;
            margin-top: 15px;
            font-size: 13px;
        }
        .char-count {
            text-align: right;
            margin-top: 5px;
            font-size: 13px;
            color: #64748b;
        }
        h3 { margin-top: 0; color: #1e293b; }
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🪶 Linguify</h1>
        <p>Refine and Humanize AI-generated content into polished academic writing</p>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h3>⚙️ Options</h3>
            <label>
                <input type="checkbox" id="passive">
                <span>Convert to Passive Voice</span>
            </label>
            <label>
                <input type="checkbox" id="synonyms">
                <span>Use Formal Synonyms</span>
            </label>
            
            <div class="limits">
                <strong>⚡ Limits:</strong><br>
                • Max: 5,000 characters<br>
                • Max: 1,000 words
            </div>
        </div>
        
        <div class="main">
            <h3>📝 Input Text</h3>
            <textarea 
                id="inputText" 
                placeholder="Type or paste your text here..."
                maxlength="5000"
            ></textarea>
            <div class="char-count">
                <span id="charCount">0</span> / 5,000 characters
            </div>
            
            <button id="processBtn" onclick="processText()">
                ✨ Refine with Linguify
            </button>
            <div id="loading" class="loading">⏳ Processing your text...</div>
            
            <h3>🎓 Refined Output</h3>
            <div id="output" class="output">Results will appear here...</div>
        </div>
    </div>

    <script>
        const inputText = document.getElementById('inputText');
        const charCount = document.getElementById('charCount');
        const processBtn = document.getElementById('processBtn');
        
        // Update character count
        inputText.addEventListener('input', () => {
            const length = inputText.value.length;
            charCount.textContent = length;
            charCount.style.color = length > 4500 ? '#ef4444' : '#64748b';
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
            
            // Disable button and show loading
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
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    output.innerHTML = `❌ ${data.error}`;
                } else {
                    output.innerHTML = data.result;
                }
                
            } catch (error) {
                console.error('Error:', error);
                output.innerHTML = `❌ Error connecting to server: ${error.message}`;
            } finally {
                processBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        // Allow Enter to submit (Ctrl+Enter or Cmd+Enter)
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
