import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class NMTDataset(Dataset):
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

def get_dataloaders(batch_size=32, max_length=128):
    # Load the IITB dataset
    dataset = load_dataset('cfilt/iitb-english-hindi')['train']
    
    dataset = dataset.train_test_split(test_size=0.1)
    train_data = dataset['train']
    val_data = dataset['test']

    # Initialize tokenizers
    en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # English
    hi_tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')  # Hindi

    # Create datasets
    train_dataset = NMTDataset(train_data, en_tokenizer, hi_tokenizer, max_length)
    val_dataset = NMTDataset(val_data, en_tokenizer, hi_tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, en_tokenizer, hi_tokenizer

# Example usage
if __name__ == "__main__":
    train_loader, val_loader, en_tokenizer, hi_tokenizer = get_dataloaders()
    
    # Get a sample batch
    batch = next(iter(train_loader))
    
    print("English input IDs shape:", batch['en_input_ids'].shape)
    print("Hindi input IDs shape:", batch['hi_input_ids'].shape)
    print("English attention mask shape:", batch['en_attention_mask'].shape)
    print("Hindi attention mask shape:", batch['hi_attention_mask'].shape)