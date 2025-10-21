import gradio as gr
import os
import sys
from transformer.app import AcademicTextHumanizer, NLP_GLOBAL, download_nltk_resources
from nltk.tokenize import word_tokenize

# Environment-specific settings
def get_server_config():
    """Get server configuration based on environment"""
    is_cloud_deployment = os.environ.get('VERCEL') or os.environ.get('CLOUD_DEPLOYMENT')
    
    config = {
        'max_file_size_mb': 2 if is_cloud_deployment else 5,  # Reduced for safety
        'max_text_length': 25000 if is_cloud_deployment else 50000,
        'max_word_count': 5000 if is_cloud_deployment else 10000,
    }
    return config

# Get configuration
config = get_server_config()

# Download NLTK resources if needed
download_nltk_resources()

def humanize_text(text, use_passive, use_synonyms):
    """
    Main processing function with size validation
    """
    if not text.strip():
        return "⚠️ Please enter text to begin refinement."
    
    # Validate input size using config
    if len(text) > config['max_text_length']:
        return f"⚠️ Text too long! Maximum {config['max_text_length']} characters allowed. Your text: {len(text)} characters"
    
    try:
        input_word_count = len(word_tokenize(text, language='english', preserve_line=True))
        
        # Check for very large documents using config
        if input_word_count > config['max_word_count']:
            return f"⚠️ Text too large! Please split into smaller sections (max {config['max_word_count']} words)."
            
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
        
        # Format the output with stats
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
    Safely handle uploaded text files with better memory management.
    Uses the configured max_file_size_mb from config.
    """
    if file is None:
        return ""
    
    max_size_mb = config['max_file_size_mb']
    
    try:
        # Get file size safely
        if hasattr(file, 'size'):
            file_size = file.size
        elif hasattr(file, 'name'):
            file_size = os.path.getsize(file.name)
        else:
            return "❌ Unable to determine file size."
        
        # Enforce smaller size limit
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            return f"❌ File too large! Max allowed is {max_size_mb} MB. Your file: {file_size / (1024*1024):.1f} MB"
        
        # Read file with explicit encoding and size limits
        try:
            with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size_bytes + 1024)  # Read with limit
                
                # Check if we hit the limit
                if len(content.encode('utf-8')) >= max_size_bytes:
                    return f"❌ File content exceeds {max_size_mb} MB limit"
                    
                return content
                
        except UnicodeDecodeError:
            # Fallback for encoding issues
            with open(file.name, 'r', encoding='latin-1', errors='ignore') as f:
                content = f.read(max_size_bytes + 1024)
                if len(content.encode('utf-8')) >= max_size_bytes:
                    return f"❌ File content exceeds {max_size_mb} MB limit"
                return content
                
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"

# Create Gradio interface with better file handling
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue"
    ),
    css="""
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }
    .output-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    """
) as demo:
    
    # Header
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
            # Settings Panel (like sidebar)
            gr.Markdown("### ⚙️ Linguify Options")
            gr.Markdown("Customize your refinement settings:")
            
            use_passive = gr.Checkbox(
                label="Convert sentences to Passive Voice", 
                value=False
            )
            use_synonyms = gr.Checkbox(
                label="Replace with Formal Synonyms", 
                value=False
            )
            
            gr.Markdown("---")
            gr.Markdown("### 📂 File Upload")
            # Update file upload with smaller limits using config
            file_upload = gr.File(
                label=f"Upload a .txt File (Max {config['max_file_size_mb']}MB)",
                file_types=[".txt"],
                file_count="single",
                file_size_limit=config['max_file_size_mb'] * 1024 * 1024
            )

            gr.Markdown("*You can also paste text directly in the input box*")
            
        with gr.Column(scale=2):
            # Input Area
            gr.Markdown("### 📝 Input Text")
            input_text = gr.Textbox(
                label="Enter or paste your text here:",
                placeholder=f"Type or paste your text here to refine... (Max {config['max_text_length']} characters)",
                lines=8,
                show_label=False
            )
            
            # Process Button
            process_btn = gr.Button(
                "✨ Refine with Linguify", 
                variant="primary",
                size="lg"
            )
            
            # Output Area
            gr.Markdown("### 🎓 Refined Output")
            output_text = gr.Markdown(
                label="Results will appear here...",
                show_label=False
            )
    
    # Examples
    gr.Markdown("### 💡 Example Inputs")
    gr.Examples(
        examples=[
            ["The utilization of AI technology facilitates enhanced productivity outcomes across multiple organizational domains."],
            ["It is imperative that we optimize our operational efficiencies through strategic implementation of innovative technological solutions."],
            ["The company leverages cutting-edge solutions to drive innovation and maintain competitive advantage in the global marketplace."]
        ],
        inputs=input_text
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
        <p style="margin: 0; color: #64748b;">🪶 Linguify — Academic Text Refiner | Crafted with care by Arunsystems</p>
    </div>
    """)
    
    # Event handlers
    def process_all(text, file, passive, synonyms):
        """
        Process text input or uploaded file safely.
        """
        # If a file is uploaded, replace the text input
        if file is not None:
            file_content = process_file(file)
            # If file reading fails, show error
            if file_content.startswith("❌"):
                return file_content
            text = file_content
        
        # Process the text normally
        return humanize_text(text, passive, synonyms)

    
    # Connect the button
    process_btn.click(
        fn=process_all,
        inputs=[input_text, file_upload, use_passive, use_synonyms],
        outputs=output_text
    )
    
    # Auto-process when file is uploaded
    file_upload.change(
        fn=process_file,
        inputs=file_upload,
        outputs=input_text
    )

# Launch the app with better error handling
if __name__ == "__main__":
    try:
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
            show_error=True,
            favicon_path=None,
            inbrowser=False,
            max_file_size=f"{config['max_file_size_mb']}MB"  # Add Gradio-specific limit
        )
    except Exception as e:
        print(f"Failed to launch app: {e}")
        sys.exit(1)
