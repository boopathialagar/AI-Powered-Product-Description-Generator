# ==============================
# 1. LOAD DATASET
# ==============================
import json

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []

for item in data:
    formatted_input = "Write a professional and SEO-friendly product description:\n" + item["input"]
    
    formatted_data.append({
        "input": formatted_input,
        "output": item["output"]
    })

# ==============================
# 2. CONVERT TO DATASET
# ==============================
from datasets import Dataset

dataset = Dataset.from_list(formatted_data)

# ==============================
# 3. LOAD MODEL & TOKENIZER
# ==============================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ==============================
# 4. APPLY LORA (PEFT)
# ==============================
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ==============================
# 5. TOKENIZATION (FIXED)
# ==============================
def tokenize_function(example):
    inputs = tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    targets = tokenizer(
        example["output"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = targets["input_ids"]

    # IMPORTANT: Replace padding token with -100
    labels = [
        (l if l != tokenizer.pad_token_id else -100)
        for l in labels
    ]

    inputs["labels"] = labels
    return inputs

tokenized_dataset = dataset.map(tokenize_function)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
tokenized_dataset.set_format("torch")

# ==============================
# 6. TRAINING CONFIG
# ==============================
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_dir="./logs",
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=False  # Set True only if GPU available
)

# ==============================
# 7. TRAINER
# ==============================
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# ==============================
# 8. TRAIN MODEL
# ==============================
model.train()
trainer.train()

# ==============================
# 9. SAVE MODEL
# ==============================
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")

# ==============================
# 10. INFERENCE
# ==============================
model.eval()

def generate_description(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(generate_description(
    "Write a professional and SEO-friendly product description:\nProduct: Smartphone | Features: 5G, Fast charging | Audience: Professionals"
))