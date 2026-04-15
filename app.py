import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ==============================
# LOAD MODEL (OPTIMIZED)
# ==============================
base_model_name = "google/flan-t5-small"
adapter_path = "fine-tuned-model"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

model = model.to(device)
model.eval()

# ==============================
# GENERATION FUNCTION (IMPROVED)
# ==============================
def generate_description(product, features, audience, language):
    prompt = f"""
Write a professional, engaging, and SEO-friendly product description.

Product: {product}
Features: {features}
Audience: {audience}
Language: {language}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# GRADIO UI (PRO VERSION)
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ AI Product Description Generator")
    gr.Markdown("Generate high-quality product descriptions in English & Tamil using AI.")

    with gr.Row():
        product = gr.Textbox(label="Product Name", placeholder="e.g., Smartphone")
        audience = gr.Textbox(label="Target Audience", placeholder="e.g., Students")

    features = gr.Textbox(label="Features", placeholder="e.g., 5G, Fast charging")

    language = gr.Dropdown(["English", "Tamil"], label="Language", value="English")

    btn = gr.Button("🚀 Generate Description")

    output = gr.Textbox(label="Generated Description", lines=6)

    btn.click(
        generate_description,
        inputs=[product, features, audience, language],
        outputs=output
    )

# ==============================
# RUN APP
# ==============================
demo.launch()