import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any
from transformers import PreTrainedTokenizerBase

class OptimizedDataset(Dataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizerBase, max_length: int, preload: bool = False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preload = preload
        self.preloaded_items = None
        
        # Preload a subset for faster access if requested
        if self.preload:
            print("Preloading dataset subset...")
            self.preloaded_items = []
            for idx in range(min(10000, len(self.dataset))):
                self.preloaded_items.append(self._process_item(idx))
            print(f"Preloaded {len(self.preloaded_items)} items")
            
    def _process_item(self, idx):
        text = self.dataset[idx]['text'] if 'text' in self.dataset[idx] else self.dataset[idx]['content']
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.preload and idx < len(self.preloaded_items):
            return self.preloaded_items[idx]
        return self._process_item(idx)

class SimpleDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                  padding="max_length", return_tensors="pt")
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class MPSDataLoader:
    """DataLoader optimized for MPS with prefetching and pinning workarounds"""
    def __init__(self, dataset, batch_size, num_workers, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # For MPS, we need custom handling since pin_memory isn't supported
        if torch.backends.mps.is_available():
            # Create a standard dataloader without pin_memory
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=3 if self.num_workers > 0 else None  # Increase prefetch factor
            )
        else:
            # For other systems, use standard pinned memory
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
                prefetch_factor=2 if self.num_workers > 0 else None
            )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader) 