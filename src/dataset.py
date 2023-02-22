import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length,
        ignore_index,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index 
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        item = {"text": text}
        return item
    
    def generate_batch(self, item_list):
        text_list = [x["text"] for x in item_list]

        tokenized_text = self.tokenizer(
            text_list,
            add_special_tokens=True, # 借用分割符当作起始token与终止token
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        raw_input_ids = tokenized_text.input_ids
        input_ids = torch.LongTensor([x[:-1] for x in raw_input_ids])
        labels = torch.LongTensor([x[1:] for x in raw_input_ids])
        labels = torch.where(labels == self.pad_token_id, self.ignore_index, labels)
        attention_mask = tokenized_text.attention_mask
        attention_mask = torch.ByteTensor([x[1:] for x in attention_mask])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return batch



