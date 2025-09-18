import pandas as pd
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from src.vectorization import clinical_vectorizer

def train_model(train_df):
    train_df["text"] = train_df["question"] + " Answer: " + train_df["answer"].astype(str)
    train_dataset = Dataset.from_pandas(train_df[["text"]])

    model_name = "unsloth/Meta-Llama-3.1-8B"  # Example model name
    max_seq_length = 2048
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            max_steps=80,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    trainer_stat = trainer.train()
    return trainer_stat

if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    train_model(train_df)