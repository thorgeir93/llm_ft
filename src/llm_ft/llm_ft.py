from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Any

import logging

import torch
from datasets import load_dataset
from datasets import arrow_dataset
from datasets.dataset_dict import DatasetDict

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from sklearn.metrics import accuracy_score

PRETRAINED_MODEL: Final[str] = "bert-base-uncased"

OUTPUT_DIR: Final[str] = "./result_02"

log: logging.Logger = logging.getLogger(__name__)


# tuple[arrow_dataset.Dataset, arrow_dataset.Dataset]
def load_imdb_dataset() -> arrow_dataset.Dataset:
    """"""
    return load_dataset("imdb")


def preprocess_data(
    data: arrow_dataset.Dataset, tokenizer: AutoTokenizer
) -> arrow_dataset.Dataset:
    # Tokenize the dataset
    def tokenize_function(examples: dict[str, Any]) -> AutoTokenizer:
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    tokenized_data: arrow_dataset.Dataset = data.map(tokenize_function, batched=True)

    # Prepare for PyTorch
    tokenized_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized_data


def create_trainer(model, tokenized_train, tokenized_test, output_dir: Path) -> Trainer:
    """"""

    # Define accuracy metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    return trainer


def get_dataset(
    dataset: arrow_dataset.Dataset, key: str, percent: float = 1.0
) -> arrow_dataset.Dataset:
    """"""
    return dataset[key].shuffle(seed=42).select(range(int(percent * len(dataset[key]))))

def validate_model_with_data(model_path: Path, test_data: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the input text
    inputs = tokenizer(test_data, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    # Print the results
    print(f"Input text: {test_data}")
    print(f"Predicted label: {predicted_label}")

def fine_tune_model(model_name: str, output_dir: Path):
    """Fine tune given model with IMDB data.

    All related data is stored in the given output directory.
    """
    log.info("Load IMDB dataset ...")
    dataset: arrow_dataset.Dataset = load_imdb_dataset()

    # Split dataset into training and test
    train_data: arrow_dataset.Dataset = get_dataset(dataset, "train", percent=0.01)
    test_data: arrow_dataset.Dataset = get_dataset(dataset, "test", percent=0.01)

    # Load pre-trained tokenizer
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info("Preprocess the data (tokenize) ...")
    tokenized_train: arrow_dataset.Dataset = preprocess_data(
        data=train_data, tokenizer=tokenizer
    )
    tokenized_test: arrow_dataset.Dataset = preprocess_data(
        data=test_data, tokenizer=tokenizer
    )

    # Load pre-trained BERT model with a classification head
    log.info("Load pretrained model %s ...", model_name)
    model: AutoModelForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
    )

    log.info("Model type: %s", type(model))

    trainer: Trainer = create_trainer(
        model=model,
        tokenized_train=tokenized_train,
        tokenized_test=tokenized_test,
        output_dir=output_dir,
    )

    trainer.train()
    results = trainer.evaluate()

    model.save_pretrained(output_dir / "fine_tuned_bert")
    tokenizer.save_pretrained(output_dir / "fine_tuned_bert")

def get_dir_with_ts() -> Path:
    """Create directory in the current dir with current timestamp."""
    dir_ts: Path = Path(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"))
    dir_ts.mkdir()
    return dir_ts


def main():
    output_dir: Path = get_dir_with_ts()
    # model_result_filename: str = "fine_tuned_bert"
    fine_tune_model(model_name=PRETRAINED_MODEL, output_dir=output_dir)


if __name__ == "main":
    main()
    # Example text to test the model
    # test_text = "This is an example sentence for testing."
    # validate_model_with_data()
