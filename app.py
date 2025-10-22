from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformer.app import AcademicTextHumanizer, download_nltk_resources
import os

# Initialize
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

@app.post("/api/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    try:
        # Validate input size
        if len(request.text) > 10000:
            return ProcessResponse(
                result="", 
                error="Text too long! Maximum 10,000 characters allowed."
            )
        
        humanizer = AcademicTextHumanizer(
            p_passive=0.2,
            p_synonym_replacement=0.2,
            p_academic_transition=0.2
        )
        
        result = humanizer.humanize_text(
            request.text,
            use_passive=request.use_passive,
            use_synonyms=request.use_synonyms
        )
        
        return ProcessResponse(result=result)
        
    except Exception as e:
        return ProcessResponse(result="", error=f"Processing error: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Linguify"}

# HTML Frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Linguify 🪶 - Academic Text Refiner</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .sidebar {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .content {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: #374151;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .checkbox-group input {
            margin-right: 0.5rem;
        }
        
        textarea {
            width: 100%;
            min-height: 200px;
            padding: 1rem;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #0072ff;
            box-shadow: 0 0 0 3px rgba(0, 114, 255, 0.1);
        }
        
        button {
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 114, 255, 0.3);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .output {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            min-height: 200px;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #0072ff;
            padding: 1rem;
        }
        
        .error {
            color: #dc2626;
            background: #fef2f2;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #fecaca;
        }
        
        .success {
            color: #059669;
            background: #f0fdf4;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #bbf7d0;
        }
        
        .examples {
            margin-top: 2rem;
        }
        
        .example-text {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
        }
        
        .example-text:hover {
            background: #e2e8f0;
            border-color: #0072ff;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Linguify 🪶</h1>
            <p>Refine and Humanize AI-generated content into polished academic writing</p>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                <div class="form-group">
                    <h3>⚙️ Linguify Options</h3>
                    <p>Customize your refinement settings:</p>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="usePassive">
                        <label for="usePassive">Convert sentences to Passive Voice</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="useSynonyms">
                        <label for="useSynonyms">Replace with Formal Synonyms</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <h3>📝 Text Limits</h3>
                    <p><strong>Maximum limits:</strong></p>
                    <ul style="margin-left: 1.5rem; color: #64748b;">
                        <li>10,000 characters</li>
                        <li>2,000 words</li>
                    </ul>
                </div>
                
                <div class="examples">
                    <h3>💡 Example Inputs</h3>
                    <div class="example-text" onclick="loadExample(0)">
                        AI technology enhances productivity across organizations.
                    </div>
                    <div class="example-text" onclick="loadExample(1)">
                        We should optimize efficiencies through innovative solutions.
                    </div>
                    <div class="example-text" onclick="loadExample(2)">
                        The company uses cutting-edge solutions to drive innovation.
                    </div>
                </div>
            </div>
            
            <div class="content">
                <div class="form-group">
                    <label for="inputText">📝 Input Text</label>
                    <textarea 
                        id="inputText" 
                        placeholder="Type or paste your text here to refine... (Max 10,000 characters)"
                        maxlength="10000"
                    ></textarea>
                    <div style="text-align: right; margin-top: 0.5rem; color: #64748b;">
                        <span id="charCount">0</span>/10000 characters
                    </div>
                </div>
                
                <button onclick="processText()" id="processBtn">
                    ✨ Refine with Linguify
                </button>
                
                <div class="loading" id="loading">
                    Processing your text... Please wait.
                </div>
                
                <div class="form-group" style="margin-top: 2rem;">
                    <label for="outputText">🎓 Refined Output</label>
                    <div class="output" id="outputText">
                        Results will appear here...
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🪶 Linguify — Academic Text Refiner | Crafted with care by Arunsystems</p>
        </div>
    </div>

    <script>
        const examples = [
            "AI technology enhances productivity across organizations.",
            "We should optimize efficiencies through innovative solutions.",
            "The company uses cutting-edge solutions to drive innovation."
        ];
        
        function loadExample(index) {
            document.getElementById('inputText').value = examples[index];
            updateCharCount();
        }
        
        function updateCharCount() {
            const textarea = document.getElementById('inputText');
            const charCount = document.getElementById('charCount');
            charCount.textContent = textarea.value.length;
        }
        
        document.getElementById('inputText').addEventListener('input', updateCharCount);
        
        async function processText() {
            const inputText = document.getElementById('inputText').value.trim();
            const usePassive = document.getElementById('usePassive').checked;
            const useSynonyms = document.getElementById('useSynonyms').checked;
            const outputElement = document.getElementById('outputText');
            const loadingElement = document.getElementById('loading');
            const processBtn = document.getElementById('processBtn');
            
            if (!inputText) {
                outputElement.innerHTML = '<div class="error">⚠️ Please enter some text to begin refinement.</div>';
                return;
            }
            
            if (inputText.length > 10000) {
                outputElement.innerHTML = '<div class="error">⚠️ Text too long! Maximum 10,000 characters allowed.</div>';
                return;
            }
            
            // Show loading state
            loadingElement.style.display = 'block';
            processBtn.disabled = true;
            outputElement.innerHTML = 'Processing...';
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: inputText,
                        use_passive: usePassive,
                        use_synonyms: useSynonyms
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    outputElement.innerHTML = `<div class="error">❌ ${data.error}</div>`;
                } else {
                    outputElement.innerHTML = `<div class="success">${data.result}</div>`;
                }
                
            } catch (error) {
                outputElement.innerHTML = `<div class="error">❌ Network error: Please check your connection and try again.</div>`;
            } finally {
                loadingElement.style.display = 'none';
                processBtn.disabled = false;
            }
        }
        
        // Initialize character count
        updateCharCount();
    </script>
</body>
</html>
"""
