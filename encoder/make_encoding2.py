import os
import torch
from transformers import AlbertTokenizer, AlbertModel
import pandas as pd
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = '../best_model/best_model_albert2'
tokenizer = AlbertTokenizer.from_pretrained(model_path)
albert_model = AlbertModel.from_pretrained(model_path).to(device)


def read_papers_from_excel(file_path, column_name='content'):
    df = pd.read_excel(file_path)
    paper_texts = df[column_name].dropna().tolist()
    return paper_texts


def generate_and_save_embeddings(paper_texts, output_file):
    embeddings = []

    for text in tqdm(paper_texts, desc="Generating embeddings"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = albert_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            cls_token_embedding = last_hidden_states[:, 0, :].cpu().numpy()
            embeddings.append(cls_token_embedding.squeeze(0))


    np.save(output_file, np.array(embeddings))


excel_file_path = '../dataset/train2.xlsx'
output_file = '../embedding/embeddings_albert/train2.npy'


paper_texts = read_papers_from_excel(excel_file_path)


if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))


generate_and_save_embeddings(paper_texts, output_file)
