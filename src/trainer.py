import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import compute_metrics_batch


class LMTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        compute_loss,
        compute_metrics,
        output_dir,
        training_args,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_loss = compute_loss
        self.compute_metrics = compute_metrics
        self.output_dir = output_dir

        self.training_args = training_args
        
        self.writer = None
        if training_args.is_master_process:
            self.writer = SummaryWriter(output_dir)
        
        self.device = None
        for _, p in model.named_parameters():
            self.device = p.device
            break
        
    def train(self):
        model = self.model
        args = self.training_args
        optimizer = self.optimizer
        scheduler = self.scheduler

        global_step = 0
        total_loss = 0
        total_preds_true = 0
        total_valid_token = 0
        best_accuracy = 0

        for epoch in tqdm(range(args.num_epochs), total=args.num_epochs, leave=False):
            model.train()
            train_loss = 0
            train_preds_true = 0
            train_valid_token = 0

            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in args.cuda_item}

                outputs = model(inputs)

                logits = outputs["logits"]
                labels = inputs["labels"]

                loss = self.compute_loss(logits, labels)

                loss.backward()
                
                if (batch_idx + 1) % args.num_grad_accumulation == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                
                total_loss += loss.item()
                train_loss += loss.item()
                metrics = compute_metrics_batch(logits.detach().cpu(), labels.detach().cpu(), batch["attention_mask"].detach().cpu())
                total_preds_true += metrics["num_preds_true"]
                total_valid_token += metrics["num_valid_token"]
                train_preds_true += metrics["num_preds_true"]
                train_valid_token += metrics["num_valid_token"]

                global_step = epoch * len(self.train_dataloader) + batch_idx
                if (global_step + 1) % args.write_step == 0 and self.writer:
                    avg_loss = total_loss / (global_step + 1)
                    self.writer.add_scalar("Train-Step-Loss", avg_loss, global_step=global_step)

                    accuracy = total_preds_true / total_valid_token
                    self.writer.add_scalar("Train-Step-Accuracy", accuracy, global_step=global_step)
                
            if not self.writer:
                continue
        
            train_loss /= len(self.train_dataloader)
            self.writer.add_scalar("Train--Epoch-Loss", train_loss, global_step=epoch)
            accuracy = train_preds_true / train_valid_token
            self.writer.add_scalar("Train-Epoch-Accuracy", accuracy, global_step=epoch)
            
            eval_result = self.evaluation()
            eval_loss = eval_result["loss"]
            eval_metrics = eval_result["metrics"]
            eval_accuracy = eval_metrics["eval_metrics"]
            self.writer.add_scalar("Eval-Epoch-Loss", eval_loss, global_step=epoch)
            self.writer.add_scalar("Eval-Epoch-Accuracy", eval_accuracy, global_step=epoch)

            if eval_accuracy > best_accuracy:
                save_path = os.path.join(self.output_dir, "model.pth")
                torch.save(model.state_dict(), save_path)
                best_accuracy = eval_accuracy
                

    def evaluation(self):
        eval_loss = 0
        model = self.model
        model.eval()
        logits_list = []
        labels_list = []
        mask_list = []
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.args.cuda_item}

                outputs = model(inputs)

                logits = outputs["logits"]
                labels = inputs["labels"]
                loss = self.compute_loss(logits, labels)

                eval_loss += loss.item()
                logits_list.append(logits.detach().cpu().argmax(-1))
                labels_list.append(labels.detach().cpu())
                mask_list.append(batch["attention_mask"].detach().cpu())
        
        eval_loss /= len(self.eval_dataloader)
        metrics = self.compute_metrics(logits_list, labels_list, mask_list)

        eval_result = {
            "loss": eval_loss,
            "metrics": metrics,
        }

        return eval_result
                




            

