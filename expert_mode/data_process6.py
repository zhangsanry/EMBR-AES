import os
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 加载xlm-roberta-large模型和分词器
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
model = XLMRobertaModel.from_pretrained('xlm-roberta-large')

# 定义从Excel文件中读取数据的方法
def read_papers_from_excel(file_path, column_name):
    df = pd.read_excel(file_path)
    paper_texts = df[column_name].dropna().tolist()  # 读取指定列并去掉缺失值
    return paper_texts

# 分段函数
def split_into_segments(text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    segments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return segments

# 处理所有论文
def process_papers(paper_texts):
    final_embeddings = []
    for paper_text in tqdm(paper_texts, desc="Processing papers"):
        segments = split_into_segments(paper_text)
        embeddings = []
        with torch.no_grad():
            for segment in segments:
                if len(segment) > 512:
                    segment = segment[:512]  # 确保输入长度不超过512
                inputs = torch.tensor([segment])
                outputs = model(inputs)
                last_hidden_states = outputs.last_hidden_state
                segment_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
                embeddings.append(segment_embedding)
        if embeddings:
            final_embedding = np.mean(embeddings, axis=0)
            final_embeddings.append(final_embedding)
    return np.array(final_embeddings)

# 指定Excel文件的路径和列名
file_path = './dataset/train.xlsx'
column_name = 'content'  # 替换为你实际要读取的列名

# 读取所有论文文本
paper_texts = read_papers_from_excel(file_path, column_name)

# 获取所有论文的最终嵌入向量
final_embeddings = process_papers(paper_texts)

# 数据标准化
scaler = StandardScaler()
final_embeddings = scaler.fit_transform(final_embeddings)

# 将标准化后的数据保存到CSV文件中
standardized_data_path = '../output/standardized_embeddings_by_roberta.csv'
pd.DataFrame(final_embeddings).to_csv(standardized_data_path, index=False)

print(f'Embeddings have been saved to {standardized_data_path}')
