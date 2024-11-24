import os
import torch
from transformers import AlbertTokenizer, AlbertForMaskedLM
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForMaskedLM.from_pretrained('albert-base-v2').to(device)


def read_papers_from_excel(file_path, column_name):
    df = pd.read_excel(file_path)
    paper_texts = df[column_name].dropna().tolist()
    return paper_texts


class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)
        segments = [tokens[i:i + self.max_length] for i in range(0, len(tokens), self.max_length)]

        inputs = []
        for segment in segments:
            segment_inputs = self.tokenizer.prepare_for_model(segment, max_length=self.max_length,
                                                              padding='max_length', truncation=True,
                                                              return_tensors="pt", return_special_tokens_mask=True)
            segment_inputs = {key: val.squeeze(0) for key, val in segment_inputs.items()}
            segment_inputs["labels"] = segment_inputs["input_ids"].detach().clone()

            # 15%的随机掩蔽
            probability_matrix = torch.full(segment_inputs["labels"].shape, 0.15)
            special_tokens_mask = segment_inputs.pop("special_tokens_mask", None)
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.bool()
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            segment_inputs["labels"][~masked_indices] = -100
            inputs.append(segment_inputs)

        return inputs


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    for sample in batch:
        for segment in sample:
            input_ids.append(segment['input_ids'])
            attention_mask.append(segment['attention_mask'])
            labels.append(segment['labels'])
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


file_path = '../dataset/train2.xlsx'
column_name = 'content'
paper_texts = read_papers_from_excel(file_path, column_name)
train_texts, val_texts = train_test_split(paper_texts, test_size=0.2, random_state=42)
train_dataset = PaperDataset(train_texts, tokenizer)
val_dataset = PaperDataset(val_texts, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
best_eval_loss = float('inf')
save_path = '../best_model/best_model_albert2'

for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        inputs = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")

    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(val_loader)
    print(f"Average validation loss: {avg_eval_loss:.4f}")
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved the best model with validation loss: {best_eval_loss:.4f}")
