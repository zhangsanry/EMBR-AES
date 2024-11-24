import os
import torch
from transformers import BertTokenizerFast, AutoModel
from pdfminer.high_level import extract_text
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = '../best_model/best_model_chinese'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
roberta_model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)



def read_papers_from_folder(folder_path):
    paper_texts = {}
    for filename in tqdm(os.listdir(folder_path), desc="Reading papers"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                text = extract_text(file_path)
                text = text.replace('\n', ' ')

                paper_texts[filename] = text
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    return paper_texts



def generate_and_save_embeddings(paper_texts, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename, text in tqdm(paper_texts.items(), desc="Generating embeddings"):

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)


            cls_token_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()


        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_embedding.npy"


        np.save(os.path.join(output_dir, output_filename), cls_token_embedding)



folder_path = '../dataset/TDBSW'
output_dir = '../embedding/embeddings'


paper_texts = read_papers_from_folder(folder_path)


generate_and_save_embeddings(paper_texts, output_dir)
