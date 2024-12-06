from typing import List
from datasets import load_dataset, DatasetDict
from transformers.tokenization_utils_base import BatchEncoding
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pdb

class DatasetTokenization(Dataset):
    def __init__(self, labels, input_ids, instruction, tokenizer, max_length):
        self.inputs = input_ids
        self.instructions = instruction
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.inputs)
    
    def prepare_prompt(self, input, label) -> str | List[int] | List[str] | List[List[int]] | BatchEncoding:

        messages = [
            {"role": "user", "content": f"{input}"},
            {
                "role": "assistant",
                "content": f"{label}",
            },
        ]

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False).strip()
        encoded_x = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.max_length)
        encoded_y = self.tokenizer.encode(label + self.tokenizer.eos_token)

        labels_ignore_idx = len(encoded_x["input_ids"][:sum(encoded_x["attention_mask"])]) - len(self.tokenizer.encode(label + self.tokenizer.eos_token))
        if 0 in encoded_x["attention_mask"]:
            encoded_y = [-100]*labels_ignore_idx + encoded_y + [-100]*(len(encoded_x["attention_mask"])-encoded_x["attention_mask"].index(0))
        else:
            encoded_y = [-100]*labels_ignore_idx + encoded_y

        

        return encoded_x['input_ids'][:self.max_length], encoded_y[:self.max_length], encoded_x['attention_mask'][:self.max_length]

    def __getitem__(self, idx):
        input = " ".join([self.inputs[idx], self.instructions[idx]]).strip()
        label = self.labels[idx].strip()
        input_ids, labels, mask = self.prepare_prompt(input, label)
        
        return {
            "input_ids":torch.tensor(input_ids),
            "attention_mask": torch.tensor(mask),
            "labels":torch.tensor(labels)
        }


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=max_length)

def build_custom_dataset(dataset_path, tokenizer, max_length, dataset_type):
    dataset = load_dataset(dataset_type, data_files=dataset_path)
    dataset = dataset['train']

    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_train = DatasetTokenization(dataset['output'], dataset['input'], dataset['instruction'], tokenizer, max_length)

    return dataset_train

if __name__=="__main__":
    tokenizer_path = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_path = "/data/nvme7n1p1/critical_all/critical_instruct/2024.09.06_instruct_kk_benchmarks-arc_119687.json"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ds = build_custom_dataset(dataset_path, tokenizer)
    pdb.set_trace()
    print(".")
