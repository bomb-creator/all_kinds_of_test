from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # <-- Add this line

# Load dataset
dataset = load_dataset("text", data_files={"train": "data/my_poems.txt"})

# Tokenize the data
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Build a tiny GPT model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_embd=128,
    n_layer=2,
    n_head=2
)

model = GPT2LMHeadModel(config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tiny-gpt",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./tiny-gpt")
tokenizer.save_pretrained("./tiny-gpt")
