import os
import torch
from transformers import BertTokenizerFast, AutoModelForMaskedLM
import numpy as np
from pdfminer.high_level import extract_text
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler



tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-base-chinese').to(device)



def read_papers_from_folder(folder_path):
    paper_texts = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading papers"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text(file_path)
            text = text.replace('\n', ' ')
            paper_texts.append(text)
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


folder_path = '../paper_file'


paper_texts = read_papers_from_folder(folder_path)


final_embeddings = process_papers(paper_texts)


scaler = StandardScaler()
final_embeddings = scaler.fit_transform(final_embeddings)


standardized_data_path = '../output/standardized_embeddings_by_albert.csv'
pd.DataFrame(final_embeddings).to_csv(standardized_data_path, index=False)

print(f'Embeddings have been saved to {standardized_data_path}')
