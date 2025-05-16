import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class NMTDataset(Dataset):
    """
    Dataset class for English-Hindi translation.
    Args:
        dataset: The dataset to use.
        en_tokenizer: The tokenizer for English.
        hi_tokenizer: The tokenizer for Hindi.
        max_length: The maximum sequence length for tokenization.
    """
    def __init__(self, dataset, en_tokenizer, hi_tokenizer, max_length=50):
        self.dataset = dataset
        self.en_tokenizer = en_tokenizer
        self.hi_tokenizer = hi_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # print(self.dataset[idx])
        en_sentence = self.dataset[idx]['translation']['en']
        hi_sentence = self.dataset[idx]['translation']['hi']

        en_encoding = self.en_tokenizer(
            en_sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        hi_encoding = self.hi_tokenizer(
            hi_sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'en_input_ids': en_encoding['input_ids'].squeeze(),
            'en_attention_mask': en_encoding['attention_mask'].squeeze(),
            'hi_input_ids': hi_encoding['input_ids'].squeeze(),
            'hi_attention_mask': hi_encoding['attention_mask'].squeeze()
        }