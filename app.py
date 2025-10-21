import gradio as gr
import os
import sys
from transformer.app import AcademicTextHumanizer, NLP_GLOBAL, download_nltk_resources
from nltk.tokenize import word_tokenize

# Much stricter limits for cloud deployment
def get_server_config():
    """Get server configuration with very conservative limits"""
    is_cloud_deployment = os.environ.get('VERCEL') or os.environ.get('CLOUD_DEPLOYMENT')
    
    # For cloud deployments, use VERY conservative limits
    if is_cloud_deployment:
        config = {
            'max_file_size_mb': 1,  # Only 1MB for cloud
            'max_text_length': 10000,  # 10K characters
            'max_word_count': 2000,   # 2K words
        }
    else:
        config = {
            'max_file_size_mb': 5,
            'max_text_length': 50000,
            'max_word_count': 10000,
        }
    return config

config = get_server_config()
download_nltk_resources()

def humanize_text(text, use_passive, use_synonyms):
    """
    Main processing function with strict size validation
    """
    if not text.strip():
        return "⚠️ Please enter text to begin refinement."
    
    # Strict validation
    if len(text) > config['max_text_length']:
        return f"⚠️ Text too long! Maximum {config['max_text_length']} characters allowed. Your text: {len(text)} characters"
    
    try:
        input_word_count = len(word_tokenize(text, language='english', preserve_line=True))
        
        if input_word_count > config['max_word_count']:
            return f"⚠️ Text too large! Please split into smaller sections (max {config['max_word_count']} words)."
            
        # Your processing logic here
        doc_input = NLP_GLOBAL(text)
        input_sentence_count = len(list(doc_input.sents))

        humanizer = AcademicTextHumanizer(
            p_passive=0.3,
            p_synonym_replacement=0.3,
            p_academic_transition=0.4
        )
        
        transformed = humanizer.humanize_text(
            text,
            use_passive=use_passive,
            use_synonyms=use_synonyms
        )

        output_word_count = len(word_tokenize(transformed, language='english', preserve_line=True))
        doc_output = NLP_GLOBAL(transformed)
        output_sentence_count = len(list(doc_output.sents))
        
        stats = f"""
📊 **Text Statistics:**
- **Input:** {input_word_count} words, {input_sentence_count} sentences
- **Output:** {output_word_count} words, {output_sentence_count} sentences

🎓 **Refined Text:**
{transformed}
"""
        return stats
        
    except Exception as e:
        return f"❌ Error processing text: {str(e)}"

def process_file(file):
    """
    Ultra-conservative file processing
    """
    if file is None:
        return ""
    
    try:
        # Immediate size check before any processing
        if hasattr(file, 'size'):
            file_size = file.size
        elif hasattr(file, 'name'):
            file_size = os.path.getsize(file.name)
        else:
            return "❌ Unable to determine file size."
        
        max_size_bytes = config['max_file_size_mb'] * 1024 * 1024
        if file_size > max_size_bytes:
            return f"❌ File too large! Max allowed is {config['max_file_size_mb']} MB. Your file: {file_size / (1024*1024):.1f} MB"
        
        # Read with strict limits
        with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_size_bytes)  # Don't read beyond limit
            
            # Additional safety check
            if len(content) > config['max_text_length']:
                return f"❌ File content too long! Max {config['max_text_length']} characters."
                
            return content
                
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"

# Create Gradio interface with ULTRA conservative settings
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue"),
    css="""
    .gradio-container { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    .header {
        text-align: center; padding: 2rem;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        border-radius: 16px; color: white; margin-bottom: 2rem;
    }
    .output-box {
        background: white; padding: 1.5rem; border-radius: 12px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    """
) as demo:
    
    gr.HTML("""
    <div class="header">
        <h1 style="margin:0; font-size:2.5em; font-weight:800;">Linguify 🪶</h1>
        <p style="margin:0; font-size:1.2em; opacity:0.9;">
            Refine and Humanize AI-generated content into polished academic writing
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Linguify Options")
            use_passive = gr.Checkbox(label="Convert sentences to Passive Voice", value=False)
            use_synonyms = gr.Checkbox(label="Replace with Formal Synonyms", value=False)
            
            gr.Markdown("---")
            gr.Markdown("### 📂 File Upload")
            file_upload = gr.File(
                label=f"Upload a .txt File (Max {config['max_file_size_mb']}MB)",
                file_types=[".txt"],
                file_count="single",
                # CRITICAL: This must match your config
                file_size_limit=config['max_file_size_mb'] * 1024 * 1024
            )
            gr.Markdown(f"*Max {config['max_file_size_mb']}MB • {config['max_text_length']} characters*")

        with gr.Column(scale=2):
            gr.Markdown("### 📝 Input Text")
            input_text = gr.Textbox(
                placeholder=f"Type or paste your text here... (Max {config['max_text_length']} characters)",
                lines=8,
                show_label=False,
                max_lines=8
            )
            
            process_btn = gr.Button("✨ Refine with Linguify", variant="primary", size="lg")
            
            gr.Markdown("### 🎓 Refined Output")
            output_text = gr.Markdown(show_label=False)
    
    # Examples with smaller text
    gr.Markdown("### 💡 Example Inputs")
    gr.Examples(
        examples=[
            ["AI technology enhances productivity across organizations."],
            ["We should optimize efficiencies through innovative solutions."],
            ["The company uses cutting-edge solutions to drive innovation."]
        ],
        inputs=input_text
    )
    
    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
        <p style="margin: 0; color: #64748b;">🪶 Linguify — Academic Text Refiner</p>
    </div>
    """)
    
    def process_all(text, file, passive, synonyms):
        """Process text input or uploaded file safely"""
        if file is not None:
            file_content = process_file(file)
            if file_content.startswith("❌"):
                return file_content
            text = file_content
        
        return humanize_text(text, passive, synonyms)
    
    process_btn.click(
        fn=process_all,
        inputs=[input_text, file_upload, use_passive, use_synonyms],
        outputs=output_text
    )
    
    # Remove auto-processing on file upload to reduce memory pressure
    # file_upload.change(fn=process_file, inputs=file_upload, outputs=input_text)

# Launch with memory-conscious settings
if __name__ == "__main__":
    try:
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
            show_error=True,
            inbrowser=False,
            # Critical for memory management
            max_file_size=f"{config['max_file_size_mb']}MB",
            quiet=True,  # Reduce logging overhead
            # Prevent Gradio from preloading large files
            prevent_thread_lock=True
        )
    except Exception as e:
        print(f"Failed to launch app: {e}")
        sys.exit(1)
