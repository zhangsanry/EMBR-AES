import os
import torch
from transformers import BertTokenizerFast, AutoModelForMaskedLM
from pdfminer.high_level import extract_text
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-base-chinese').to(device)


def read_papers_from_folder(folder_path):
    paper_texts = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading papers"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                text = extract_text(file_path)
                text = text.replace('\n', ' ')
                paper_texts.append(text)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
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

        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors="pt", return_special_tokens_mask=True)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = inputs["input_ids"].detach().clone()


        probability_matrix = torch.full(inputs["labels"].shape, 0.15)
        special_tokens_mask = inputs.pop("special_tokens_mask", None)
        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.bool()
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs["labels"][~masked_indices] = -100
        return inputs


folder_path = '../dataset/TDBSW'
paper_texts = read_papers_from_folder(folder_path)


train_texts, val_texts = train_test_split(paper_texts, test_size=0.2, random_state=42)


train_dataset = PaperDataset(train_texts, tokenizer)
val_dataset = PaperDataset(val_texts, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


best_eval_loss = float('inf')
save_path = '../best_model/best_model_chinese'


model.train()
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        inputs = {key: val.to(device) for key, val in batch.items()}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(val_loader)
    print(f'Validation Loss: {avg_eval_loss:.4f}')


    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f'Best model saved with validation loss: {best_eval_loss:.4f}')
