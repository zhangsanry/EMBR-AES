import os
import torch
from transformers import AlbertTokenizer, AlbertModel
import pandas as pd
from tqdm import tqdm
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 ALBERT 模型和分词器
model_path = '../best_model/best_model_albert2'  # 确保这是ALBERT模型的正确路径
tokenizer = AlbertTokenizer.from_pretrained(model_path)
albert_model = AlbertModel.from_pretrained(model_path).to(device)

# 从Excel文件中读取论文文本的方法
def read_papers_from_excel(file_path, column_name='content'):
    df = pd.read_excel(file_path)
    paper_texts = df[column_name].dropna().tolist()  # 读取指定列并去掉缺失值
    return paper_texts

# 生成词向量并保存到一个文件中
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

    # 将所有论文的词向量保存到一个.npy文件中
    np.save(output_file, np.array(embeddings))

# 指定Excel文件的路径
excel_file_path = '../dataset/train2.xlsx'
output_file = '../embedding/embeddings_albert/train2.npy'

# 读取所有论文文本
paper_texts = read_papers_from_excel(excel_file_path)

# 检查输出目录
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

# 生成并保存所有论文的词向量
generate_and_save_embeddings(paper_texts, output_file)
