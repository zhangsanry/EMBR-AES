import os
import torch
from transformers import BertTokenizerFast, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler



tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-base-chinese').to(device)


def read_papers_from_excel(file_path, column_name):
    df = pd.read_excel(file_path)
    paper_texts = df[column_name].dropna().tolist()
    return paper_texts

def split_into_segments(text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    segments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return segments


def process_papers(paper_texts):
    final_embeddings = []
    for paper_text in tqdm(paper_texts, desc="Processing papers"):
        segments = split_into_segments(paper_text)
        embeddings = []
        with torch.no_grad():
            for segment in segments:
                if len(segment) > 512:
                    segment = segment[:512]
                inputs = torch.tensor([segment])
                outputs = model(inputs)
                last_hidden_states = outputs.last_hidden_state
                segment_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
                embeddings.append(segment_embedding)
        if embeddings:
            final_embedding = np.mean(embeddings, axis=0)
            final_embeddings.append(final_embedding)
    return np.array(final_embeddings)


file_path = './dataset/train.xlsx'
column_name = 'content'


paper_texts = read_papers_from_excel(file_path, column_name)


final_embeddings = process_papers(paper_texts)


scaler = StandardScaler()
final_embeddings = scaler.fit_transform(final_embeddings)


standardized_data_path = '../output/standardized_embeddings_by_roberta.csv'
pd.DataFrame(final_embeddings).to_csv(standardized_data_path, index=False)

print(f'Embeddings have been saved to {standardized_data_path}')
