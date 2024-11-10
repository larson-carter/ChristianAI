import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the Tokenizer and Base Model
model_name = "meta-llama/Llama-3.2-1B"  # Replace with the correct Hugging Face model path
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer does not add special tokens if not needed
tokenizer.pad_token = tokenizer.eos_token

# Load the model and move it to the MPS device
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for MPS compatibility
    low_cpu_mem_usage=True
).to(device)

# 2. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Adjust based on the model's architecture
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Load and Prepare the Dataset
dataset = load_dataset(
    'text',
    data_files={'train': 'bible_text.txt'},
    split='train',
    cache_dir='./cache',
)

# 4. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding='max_length',
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# 5. Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,          # Increased batch size
    gradient_accumulation_steps=1,
    num_train_epochs=6,                     # Adjust as needed
    learning_rate=3e-4,                     # Increased learning rate
    weight_decay=0.01,
    logging_steps=1000,                     # Less frequent logging
    save_steps=5000,                        # Less frequent checkpointing
    save_total_limit=2,
    fp16=False,                             # Set to False for MPS compatibility
    bf16=False,                             # Set to False for MPS compatibility
    report_to="none",
    warmup_steps=100,
    optim="adamw_torch",
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    # Uncomment the following lines to customize the DataLoader
    # train_dataloader=DataLoader(
    #     tokenized_dataset,
    #     shuffle=True,
    #     collate_fn=data_collator,
    #     batch_size=training_args.per_device_train_batch_size,
    #     num_workers=4,  # Adjust based on your CPU cores
    # ),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 8. Start Training
trainer.train()

# 9. Save the Fine-Tuned Model
trainer.save_model("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")
