import os
import torch
from transformers import BertTokenizer, BertModel
from pdfminer.high_level import extract_text
from tqdm import tqdm
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 RoBERTa 模型和分词器
model_path = '../best_model/best_model_chinese'  # 修改为模型保存的路径
tokenizer = BertTokenizer.from_pretrained(model_path)
roberta_model = BertModel.from_pretrained(model_path).to(device)

# 定义读取文件夹中所有论文的方法
def read_papers_from_folder(folder_path):
    paper_texts = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading papers"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                text = extract_text(file_path)
                text = text.replace('\n', ' ')  # 简单预处理
                paper_texts.append(text)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    return paper_texts

# 生成词向量并保存
def generate_and_save_embeddings(paper_texts, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, text in enumerate(tqdm(paper_texts, desc="Generating embeddings")):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            cls_token_embedding = last_hidden_states[:, 0, :].cpu().numpy()

        # 保存每篇论文的词向量到文件中
        np.save(os.path.join(output_dir, f"paper_{i+1}_embedding.npy"), cls_token_embedding)

# 指定论文数据集的文件夹路径
folder_path = '../other_pdfs'  # 替换为您的实际文件夹路径
output_dir = '../embedding/embeddings'  # 保存词向量的文件夹

# 读取所有论文文本
paper_texts = read_papers_from_folder(folder_path)

# 生成并保存所有论文的词向量
generate_and_save_embeddings(paper_texts, output_dir)
