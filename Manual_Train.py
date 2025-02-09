import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    pipeline,
    DataCollatorForLanguageModeling  # Step 1: Import DataCollator
)
from datasets import load_dataset

def load_and_setup_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_data(data, tokenizer, max_length=128):
    # Join each dialogue into a single string
    texts = [" ".join(dialogue) for dialogue in data["dialog"]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"  # Ensure tensors are returned
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

def prepare_dataset(dataset_name="daily_dialog", model_name="gpt2", max_length=128):
    raw_dataset = load_dataset(dataset_name)
    tokenizer = load_and_setup_tokenizer(model_name)

    # Remove original columns after tokenization to avoid data conflicts
    tokenized_dataset = raw_dataset.map(
        lambda example: tokenize_data(example, tokenizer, max_length),
        batched=True,
        remove_columns=raw_dataset["train"].column_names  # Remove original text columns
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset, tokenizer

def create_dataloaders(tokenized_dataset, tokenizer, batch_size=4):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    return train_loader, val_loader
def train_model_manual(
    model_name="gpt2",
    dataset_name="daily_dialog",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    max_length=128,
    weight_decay=0.01
):
    tokenized_dataset, tokenizer = prepare_dataset(dataset_name, model_name, max_length)
    train_loader, val_loader = create_dataloaders(tokenized_dataset, tokenizer, batch_size)  # Pass tokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            """When you call model(**batch), the model internally:
            Shifts the labels: The input sequence [token_1, token_2, ..., token_N] becomes:
            Input IDs: [token_1, token_2, ..., token_{N-1}]
            Labels: [token_2, ..., token_N}]"""

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 100 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")

        evaluate(model, val_loader, device)

    model.save_pretrained("./gpt2-dialogue-manual")
    tokenizer.save_pretrained("./gpt2-dialogue-manual")
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}")
    model.train()

def evaluate_model(checkpoint_path="./gpt2-dialogue-manual", text="Hello!"):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = load_and_setup_tokenizer("gpt2")
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = chatbot(f"User: {text}\nAI:", max_length=50)
    print(response[0]["generated_text"])

if __name__ == "__main__":
    train = False  # Toggle between True (train) or False (evaluate)
    if train:
        train_model_manual()
    else:
        evaluate_model("./gpt2-dialogue-manual", "How are you today?")
