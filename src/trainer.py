import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter


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
        for epoch in tqdm(range(args.num_epochs), total=args.num_epochs, leave=False):
            model.train()
            epoch_loss = 0

            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items()}

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
                epoch_loss += loss.item()

                global_step = epoch * len(self.train_dataloader) + batch_idx
                if (global_step + 1) % args.write_step == 0 and self.writer:
                    avg_loss = total_loss / (global_step + 1)
                    self.writer.add_scalar("Train-Loss-Step", avg_loss, global_step=global_step)
                

            if not self.writer:
                continue
        
            epoch_loss /= len(self.train_dataloader)
            self.writer.add_scalar("Train-Loss-Epoch", epoch_loss, global_step=epoch)
            
            eval_result = self.evaluation()
            eval_loss = eval_result["eval_result"]
            self.writer.add_scalar("Eval-Loss-Epoch", eval_loss, global_step=epoch)

    
    def evaluation(self):
        eval_loss = 0
        model = self.model
        model.eval()

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, total=len(self.eval_dataloader), leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(inputs)

                logits = outputs["logits"]
                labels = inputs["labels"]
                loss = self.compute_loss(logits, labels)

                eval_loss += loss.item()
        
        eval_loss /= len(self.eval_dataloader)

        eval_result = {
            "loss": eval_loss,
        }

        return eval_result
                




            

