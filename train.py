import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# ==============================
# LOAD DATA
# ==============================
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []

for item in data:
    formatted_input = (
    "You are an expert e-commerce copywriter.\n"
    "Write a compelling, SEO-optimized product description.\n"
    "Focus on benefits, clarity, and persuasive tone.\n"
    "Keep it natural and engaging.\n\n"
    + item["input"]
)
    formatted_data.append({
        "input": formatted_input,
        "output": item["output"]
    })

dataset = Dataset.from_list(formatted_data)

# Train/validation split (90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ==============================
# MODEL
# ==============================
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ❌ REMOVED: model.config.tie_word_embeddings = False
# Flan-T5 is pretrained with tied embeddings.
# Untying introduces random weights and destabilizes training.

# ==============================
# LORA
# ==============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # ✅ Expanded from ["q", "v"] to include key & output projections
    target_modules=["q", "k", "v", "o"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================
# TOKENIZATION
# ==============================
def tokenize(example):
    inputs = tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    targets = tokenizer(
        example["output"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # Mask padding tokens in labels so they don't contribute to loss
    labels = [
        (l if l != tokenizer.pad_token_id else -100)
        for l in targets["input_ids"]
    ]

    inputs["labels"] = labels
    return inputs

tokenized = dataset.map(tokenize)
tokenized["train"] = tokenized["train"].remove_columns(["input", "output"])
tokenized["test"] = tokenized["test"].remove_columns(["input", "output"])
tokenized["train"].set_format("torch")
tokenized["test"].set_format("torch")

# ==============================
# TRAINING
# ==============================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    # ✅ Reduced from 8 to avoid overfitting on small datasets
    num_train_epochs=5,

    # ✅ Evaluate every epoch to track overfitting
    eval_strategy="epoch",

    # ✅ Only keep the best checkpoint, not all 5
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ✅ Enable mixed precision on GPU for faster training
    fp16=torch.cuda.is_available(),

    logging_dir="./logs",
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# ==============================
# SAVE (adapter only, not tokenizer)
# ==============================
model.save_pretrained("fine-tuned-model")

# ✅ Tokenizer is identical to base model — no need to duplicate it here.
# At inference time, load with: AutoTokenizer.from_pretrained("google/flan-t5-base")