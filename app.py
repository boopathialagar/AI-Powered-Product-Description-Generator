import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "google/flan-t5-small"
adapter_path = os.path.join(os.getcwd(), "fine_tuned_model")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
model = model.merge_and_unload()
model = model.to(device)
model.eval()
print("Main model loaded successfully!")

translator_name = "facebook/nllb-200-distilled-600M"

print("Loading translation tokenizer...")
trans_tokenizer = AutoTokenizer.from_pretrained(translator_name, use_fast=False)
trans_tokenizer.src_lang = "eng_Latn"

print("Loading translation model...")
trans_model = AutoModelForSeq2SeqLM.from_pretrained(
    translator_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
trans_model.eval()
print("Translation model loaded successfully!")

def generate_english(product, features, audience, category, brand):
    prompt = f"""Write a professional, engaging, SEO-optimized product description.

Use persuasive language.
Highlight benefits clearly.
Make it attractive and easy to read.

Use words like: best, premium, high-quality, affordable, durable.

Product Name: {product}
Category: {category}
Brand: {brand}
Features: {features}
Target Audience: {audience}

Generate a compelling product description:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_beams=4,
            repetition_penalty=1.2
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_to_tamil(text):
    inputs = trans_tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        translated = trans_model.generate(
            **inputs,
            forced_bos_token_id=trans_tokenizer.convert_tokens_to_ids("tam_Taml"),
            max_length=180
        )

    return trans_tokenizer.decode(translated[0], skip_special_tokens=True)

def generate_description(product, features, audience, category, brand):
    if product.strip() == "":
        return "Please enter a product name.", ""
    try:
        english = generate_english(product, features, audience, category, brand)
        tamil = translate_to_tamil(english)
        return english, tamil
    except Exception as e:
        return f"Error: {str(e)}", ""

css = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400&display=swap');

/* ── Root & Reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #f7f6f3 !important;
    color: #1a1a1a !important;
}

/* ── App wrapper ── */
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 2.5rem 1.25rem !important;
}

/* ── Header ── */
.app-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e4e2dc;
}

.app-header h1 {
    font-size: 22px;
    font-weight: 500;
    color: #1a1a1a;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}

.app-header p {
    font-size: 14px;
    color: #7a7a72;
    margin: 0;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #eeecea;
    border: 1px solid #dddbd4;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 500;
    color: #5a5a52;
    margin-top: 10px;
}

/* ── Section labels ── */
.section-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9a9a90;
    margin-bottom: 0.75rem !important;
}

/* ── Input cards ── */
.input-card {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Gradio inputs ── */
.gradio-container input[type="text"],
.gradio-container textarea {
    background: #f7f6f3 !important;
    border: 1px solid #e4e2dc !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1a1a1a !important;
    padding: 9px 12px !important;
    transition: border-color 0.15s !important;
    box-shadow: none !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: #aaa89e !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── Labels ── */
.gradio-container label span,
.gradio-container .svelte-1gfkn6j {
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #5a5a52 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Generate button ── */
.generate-btn {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 11px 28px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
    letter-spacing: -0.1px !important;
}

.generate-btn:hover { opacity: 0.82 !important; }

/* ── Output boxes ── */
.output-box .gradio-container textarea,
.output-box textarea {
    background: #f0ede8 !important;
    border: 1px solid #e4e2dc !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    color: #1a1a1a !important;
    min-height: 160px !important;
}

/* ── Output lang tag ── */
.lang-tag {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9a9a90;
    margin-bottom: 6px;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: #e4e2dc;
    margin: 1.25rem 0;
}

/* ── Info section ── */
.info-block {
    background: #ffffff;
    border: 1px solid #e4e2dc;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 14px;
    color: #5a5a52;
    line-height: 1.6;
}

.info-block strong {
    color: #1a1a1a;
    font-weight: 500;
}

/* ── Tabs ── */
.gradio-container .tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #7a7a72 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 8px 16px !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
}

.gradio-container .tab-nav button.selected {
    color: #1a1a1a !important;
    border-bottom-color: #1a1a1a !important;
    font-weight: 500 !important;
}

/* ── Footer ── */
.app-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e4e2dc;
    font-size: 12px;
    color: #b0ae a6;
    text-align: center;
}
"""
with gr.Blocks(css=css, title="Product Description Generator") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>Product description generator</h1>
        <p>Generate SEO-optimized descriptions in English and Tamil</p>
        <div class="model-badge">
            ⚡ FLAN-T5 (LoRA) &nbsp;·&nbsp; NLLB-200 Translation
        </div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Generate"):

            gr.HTML('<div class="section-label">Product identity</div>')
            with gr.Row():
                product  = gr.Textbox(label="Product name",  placeholder="e.g. Wireless Earbuds", scale=2)
                category = gr.Textbox(label="Category",       placeholder="e.g. Electronics",     scale=1)
                brand    = gr.Textbox(label="Brand",          placeholder="e.g. Sony",             scale=1)

            gr.HTML('<div class="divider"></div>')

            gr.HTML('<div class="section-label">Description inputs</div>')
            with gr.Row():
                audience = gr.Textbox(label="Target audience", placeholder="e.g. Young professionals", scale=1)
                features = gr.Textbox(label="Key features",    placeholder="e.g. Noise-cancelling, 30hr battery, Bluetooth 5.3", lines=3, scale=2)

            gr.HTML('<div class="divider"></div>')

            generate_btn = gr.Button("Generate description →", elem_classes=["generate-btn"])

            gr.HTML('<div class="section-label" style="margin-top:1.5rem;">Output</div>')
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="lang-tag">🇬🇧 English</div>')
                    english_output = gr.Textbox(label="", lines=8, show_label=False, elem_classes=["output-box"])
                with gr.Column():
                    gr.HTML('<div class="lang-tag">🇮🇳 Tamil</div>')
                    tamil_output = gr.Textbox(label="", lines=8, show_label=False, elem_classes=["output-box"])

        with gr.Tab("How it works"):
            gr.HTML("""
            <div style="padding: 0.5rem 0;">

                <div class="info-block">
                    <strong>Workflow</strong><br>
                    Enter product details → AI generates an English description →
                    Automatically translated into Tamil using NLLB-200.
                </div>

                <div class="info-block">
                    <strong>Models</strong><br>
                    Description: <code>google/flan-t5-small</code> fine-tuned with LoRA adapter<br>
                    Translation: <code>facebook/nllb-200-distilled-600M</code>
                </div>

                <div class="info-block">
                    <strong>Tips for best results</strong><br>
                    List 3–5 specific features. A clear target audience helps the model tune the tone.
                    Avoid vague inputs like "good product" — the more detail, the better the output.
                </div>

            </div>
            """)

    gr.HTML("""
    <div class="app-footer">
        Running on · <strong>FLAN-T5 small + LoRA</strong> · NLLB-200 Translation
    </div>
    """)

    generate_btn.click(
        fn=generate_description,
        inputs=[product, features, audience, category, brand],
        outputs=[english_output, tamil_output]
    )

demo.launch(share=True)