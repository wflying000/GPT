import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import GPTLMModel
from dataset import LMDataset
from trainer import LMTrainer
from utils import ComputeLoss, compute_metrics
from configuration import GPTConfig, TrainingArguments

def train():

    train_data_path = "../data/wikitext-103-v1/wikitext-103-v1-train.csv"
    eval_data_path = "../data/wikitext-103-v1/wikitext-103-v1-validation.csv"
    pretrained_model_path = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    hidden_size = 768
    num_heads = 12
    max_sequence_length = 512
    intermediate_size = 3072
    num_layers = 6
    vocab_size = tokenizer.vocab_size
    dropout = 0.1
    bias = False

    gpt_config = GPTConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        dropout=dropout,
        bias=bias,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTLMModel(gpt_config)
    model = model.to(device)

    train_data = pd.read_csv(train_data_path)
    eval_data = pd.read_csv(eval_data_path)

    max_length = 128
    ignore_index = -100
    train_dataset = LMDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        ignore_index=ignore_index,
    )

    eval_dataset = LMDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_length=max_length,
        ignore_index=ignore_index,
    )

    batch_size = 8
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.generate_batch,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = None
    compute_loss = ComputeLoss(ignore_index=ignore_index)
    output_dir = "../output/"
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = 10
    num_grad_accumulation = 4
    write_step = 1
    cuda_item = ["input_ids", "labels"]
    is_master_process = True

    training_args = TrainingArguments(
        num_epochs=num_epochs,
        num_grad_accumulation=num_grad_accumulation,
        write_step=write_step,
        cuda_item=cuda_item,
        is_master_process=is_master_process,
    )

    trainer = LMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_loss=compute_loss,
        compute_metrics=compute_metrics,
        output_dir=output_dir,
        training_args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    os.chdir(sys.path[0])
    train()



