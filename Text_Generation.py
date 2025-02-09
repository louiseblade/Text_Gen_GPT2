from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline,
)
from huggingface_hub import login


def load_and_setup_tokenizer(model_name="gpt2"):
    """
    Load and configure the tokenizer for the specified model.

    Args:
        model_name (str): Name of the model to load the tokenizer for. Defaults to "gpt2".

    Returns:
        tokenizer: Configured tokenizer with padding token set to eos_token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_data(data, model_name="gpt2"):
    """
    Tokenize the input data for the model.

    Args:
        data: Dataset containing dialog text.
        model_name (str): Name of the model to tokenize for. Defaults to "gpt2".

    Returns:
        dict: Tokenized data including input_ids, attention_mask, and labels.
    """
    tokenizer = load_and_setup_tokenizer(model_name)
    tokenized = tokenizer(
        [" ".join(dialogue) for dialogue in data["dialog"]],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized


def evaluate_model(text, checkpoint_path="./gpt2-dialogue/checkpoint-8340"):
    """
    Evaluate the model by generating a response to the input text.

    Args:
        text (str): Input text to generate a response for.
        checkpoint_path (str): Path to the model checkpoint. Defaults to "./gpt2-dialogue/checkpoint-8340".

    Returns:
        str: Generated response from the model.
    """
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = load_and_setup_tokenizer()
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = chatbot(f"User: {text}\nAI:", max_length=100)
    return response[0]["generated_text"]


def train_model():
    """
    Train the GPT-2 model on the daily dialog dataset.
    """
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    tokenized_datasets = dataset.map(tokenize_data, batched=True)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir="./gpt2-dialogue",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()

    # Evaluate the model after training
    chatbot = pipeline("text-generation", model=model, tokenizer=load_and_setup_tokenizer())
    response = chatbot("User: Hello, how are you?\nAI:", max_length=50)
    print(response[0]["generated_text"])


if __name__ == "__main__":
    train = False  # Set to True to train the model, False to evaluate

    if train:
        train_model()
    else:
        # Evaluate the model with a sample input
        response = evaluate_model("how are you , boi")
        print(response)
        #