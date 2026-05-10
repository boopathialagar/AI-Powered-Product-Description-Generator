import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

<<<<<<< HEAD
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
=======
# =========================================================
# LOAD DATASET
# =========================================================
DATA_PATH = "/kaggle/input/your-dataset-name/dataset.json"
>>>>>>> 1908346 (Updated project files)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
formatted_data = []

for item in raw_data:

    prompt = f"""
You are an expert e-commerce SEO copywriter.

Generate a compelling and SEO-optimized product description.

Focus on:
- Product benefits
- Natural tone
- Readability
- Persuasive marketing language
- Customer engagement

Product Details:
Category: {item.get("category", "")}
Brand: {item.get("brand", "")}
Features: {item.get("features", "")}
Target Audience: {item.get("audience", "")}

Generate the product description:
"""

    formatted_data.append({
        "input": prompt.strip(),
        "output": item["output"].strip()
    })

dataset = Dataset.from_list(formatted_data)
<<<<<<< HEAD
dataset = dataset.train_test_split(test_size=0.1, seed=42)
=======

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

# =========================================================
# MODEL
# =========================================================

>>>>>>> 1908346 (Updated project files)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
<<<<<<< HEAD
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
=======

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# =========================================================
# LORA CONFIG
# =========================================================

>>>>>>> 1908346 (Updated project files)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "k", "v", "o"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

<<<<<<< HEAD
def tokenize(example):
    inputs = tokenizer(
=======
# =========================================================
# TOKENIZATION
# =========================================================

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

def preprocess_function(example):

    model_inputs = tokenizer(
>>>>>>> 1908346 (Updated project files)
        example["input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )
<<<<<<< HEAD
    labels = [
        (l if l != tokenizer.pad_token_id else -100)
        for l in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs
=======

    labels = tokenizer(
        text_target=example["output"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
>>>>>>> 1908346 (Updated project files)

    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=False
)

# =========================================================
# DATA COLLATOR (DYNAMIC PADDING)
# =========================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(
    output_dir="./results",

    # Training
    num_train_epochs=5,
    learning_rate=2e-4,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
<<<<<<< HEAD
    num_train_epochs=5,
    eval_strategy="epoch",
=======

    gradient_accumulation_steps=2,

    # Evaluation
    evaluation_strategy="epoch",
>>>>>>> 1908346 (Updated project files)
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
<<<<<<< HEAD
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10,
=======

    # Logging
    logging_dir="./logs",
    logging_steps=20,

    # Optimization
    fp16=torch.cuda.is_available(),

    # Regularization
    weight_decay=0.01,

    # Save
    save_total_limit=2,

    # Reproducibility
    seed=42,

    report_to="none"
>>>>>>> 1908346 (Updated project files)
)

# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],

    tokenizer=tokenizer,
    data_collator=data_collator,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)
<<<<<<< HEAD
trainer.train()
model.save_pretrained("fine-tuned-model")
=======

# =========================================================
# TRAIN
# =========================================================

trainer.train()

# =========================================================
# SAVE MODEL
# =========================================================

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Training Completed Successfully!")
>>>>>>> 1908346 (Updated project files)
