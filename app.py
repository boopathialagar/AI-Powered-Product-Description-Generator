import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD MAIN MODEL (LoRA)
# ==============================
base_model_name = "google/flan-t5-base"
adapter_path = "fine-tuned-model"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()

# ==============================
# LOAD TRANSLATION MODEL
# ==============================
# ==============================
# LOAD TRANSLATION MODEL (EN → TA)
# ==============================

translator_name = "facebook/nllb-200-distilled-600M"

trans_tokenizer = AutoTokenizer.from_pretrained(
    translator_name,
    use_fast=False
)

# ✅ VERY IMPORTANT LINE (ADD HERE)
trans_tokenizer.src_lang = "eng_Latn"

trans_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name).to(device)
# ==============================
# GENERATE ENGLISH (IMPROVED)
# ==============================
def generate_english(product, features, audience):
    prompt = f"""
Write a high-quality, engaging, and SEO-optimized product description.

Use persuasive language.
Highlight benefits clearly.
Include words like: best, premium, high-quality, affordable.

Product: {product}
Features: {features}
Audience: {audience}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=120,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ==============================
# TRANSLATE TO TAMIL (SAFE VERSION)
# ==============================
def translate_to_tamil(text):
    inputs = trans_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    translated = trans_model.generate(
        **inputs,
        forced_bos_token_id=trans_tokenizer.convert_tokens_to_ids("tam_Taml"),
        max_length=150
    )

    return trans_tokenizer.decode(translated[0], skip_special_tokens=True)
# ==============================
# FINAL FUNCTION
# ==============================
def generate_description(product, features, audience):
    english = generate_english(product, features, audience)
    tamil = translate_to_tamil(english)
    return english, tamil

# ==============================
# GRADIO UI
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ AI Product Description Generator")
    gr.Markdown("Generate SEO-optimized product descriptions in English & Tamil")

    with gr.Row():
        product = gr.Textbox(label="Product Name", placeholder="e.g., Smartphone")
        audience = gr.Textbox(label="Target Audience", placeholder="e.g., Students")
        category = gr.Textbox(label="Category")
        brand = gr.Textbox(label="Brand")

    features = gr.Textbox(label="Features", placeholder="e.g., 5G, Fast charging")

    btn = gr.Button("🚀 Generate Description")

    output = [
        gr.Textbox(label="English Description", lines=4),
        gr.Textbox(label="Tamil Description", lines=4)
    ]

    btn.click(
        generate_description,
        inputs=[product, features, audience],
        outputs=output
    )

demo.launch()