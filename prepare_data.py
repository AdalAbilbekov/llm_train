from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pdb

class DatasetTokenization(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_tokens = torch.tensor(self.input_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])
        labels = input_tokens.clone()
        return {
            "input_ids":input_tokens,
            "attention_mask":attention_mask,
            "labels":labels
        }

def prepare_prompt(sample, tokenizer):
    messages = [
        {"role": "user", "content": f"{sample["input"]} {sample["instruction"]}"},
        {
            "role": "assistant",
            "content": f"{sample["output"]}",
        },
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    sample['prompt'] = input_text
    return sample

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=max_length)

def build_custom_dataset(dataset_path, tokenizer, max_length):
    dataset = load_dataset("json", data_files=dataset_path)

    ds = dataset.map(lambda x: prepare_prompt(x, tokenizer), num_proc=8)

    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    ds = ds.map(lambda x: tokenize_function(x, tokenizer, max_length), num_proc=8)

    dataset_train = DatasetTokenization(ds['train']['input_ids'], ds['train']['attention_mask'])

    return dataset_train

if __name__=="__main__":
    tokenizer_path = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_path = "/data/nvme7n1p1/critical_all/critical_instruct/2024.09.06_instruct_kk_benchmarks-arc_119687.json"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ds = build_custom_dataset(dataset_path, tokenizer)
    pdb.set_trace()
    print(".")
