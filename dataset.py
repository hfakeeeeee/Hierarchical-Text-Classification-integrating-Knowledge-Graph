import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess import preprocess_text

# Lớp tạo tập dữ liệu cho bộ Tiếng việt
class CustomDataset(Dataset):
    def __init__(self, chunk_dir, num_labels, tokenizer):
        self.chunk_files = [os.path.join(chunk_dir, fname) for fname in os.listdir(chunk_dir) if fname.endswith('.json')]
        self.tokenizer = tokenizer
        self.num_labels = num_labels

    def __len__(self):
        return sum(len(json.load(open(file, 'r', encoding='utf-8'))) for file in self.chunk_files)

    def __getitem__(self, idx):
        current_idx = 0
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                if idx < current_idx + len(chunk_data):
                    item = chunk_data[idx - current_idx]
                    text = preprocess_text(" ".join(item["title"])) + " " + preprocess_text(" ".join(item["abstract"]))
                    encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
                    label_vector = np.zeros(self.num_labels)
                    label_vector[item["labels"]] = 1
                    return {
                        'input_ids': encoding['input_ids'].flatten(),
                        'attention_mask': encoding['attention_mask'].flatten(),
                        'labels': torch.tensor(label_vector, dtype=torch.float)
                    }
                current_idx += len(chunk_data)

# Đếm số nhãn trong tập dữ liệu
def get_num_labels(chunk_dirs):
    all_labels = set()
    for chunk_dir in chunk_dirs:
        for chunk_file in os.listdir(chunk_dir):
            with open(os.path.join(chunk_dir, chunk_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    all_labels.update(item["labels"])
    return max(all_labels) + 1

# Đọc dữ liệu JSON từ các phần
def read_json_from_chunks(chunk_dir):
    data = []
    for file_name in os.listdir(chunk_dir):
        with open(os.path.join(chunk_dir, file_name), 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data