from __future__ import annotations
from typing import List
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from src.models.ai_small_model import CodeGenModelWrapper

def train_codegen_lora(
    model_wrapper: "CodeGenModelWrapper",
    train_samples: List[str],
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
) -> "CodeGenModelWrapper":
    """Fine-tune `model_wrapper`'s LoRA adapters on `train_samples` (raw text,
    e.g. mined commit diffs, code snippets, or docstring pairs formatted as
    single strings). Saves the resulting adapter + tokenizer to `output_dir`."""

    def tokenize_fn(examples):
        return model_wrapper.tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )

    raw_dataset = Dataset.from_dict({"text": train_samples})
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=model_wrapper.tokenizer, mlm=False
        ),
    )

    trainer.train()
    model_wrapper.save_lora_weights(output_dir)
    return model_wrapper