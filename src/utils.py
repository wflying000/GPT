import torch
from tqdm import tqdm


class ComputeLoss:
    def __init__(self, ignore_index):
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        loss = self.loss_fct(inputs, targets)

        return loss


def compute_metrics_batch(logits, labels, mask):
    """
        args:
            logits: [bsz, seq_len, num_labels] or [bsz, seq_len]
            labels: [bsz, seq_len]
            mask: [bsz, seq_len]
    """
    
    if logits.dim() == 3:
        preds = logits.argmax(-1)
    else:
        preds = logits

    num_preds_true = ((preds == labels) & mask).sum() # 预测正确的有效token数量 (有效token指除PAD之外的token)

    num_valid_token = mask.sum() # 有效token数量

    accuracy = num_preds_true / num_valid_token

    metrics = {
        "num_preds_true": num_preds_true,
        "num_valid_token": num_valid_token,
        "accuracy": accuracy,
    }

    return metrics


def compute_metrics(logits_list, labels_list, mask_list):
    num_preds_true = 0
    num_valid_token = 0

    for logits, labels, mask in tqdm(logits_list, labels_list, mask_list):
        res = compute_metrics_batch(logits, labels, mask)

        num_preds_true += res["num_preds_true"]
        num_valid_token += res["num_valid_token"]
    
    accuracy = num_preds_true / num_valid_token

    metrics = {
        "num_preds_true": num_preds_true,
        "num_valid_token": num_valid_token,
        "accuracy": accuracy,
    }

    return metrics


