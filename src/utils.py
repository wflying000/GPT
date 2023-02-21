import torch

class ComputeLoss:
    def __init__(self, ignore_index):
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        loss = self.loss_fct(inputs, targets)

        return loss