import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pdfminer.high_level import extract_text
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 定义读取文件夹中所有论文的方法
def read_papers_from_folder(folder_path):
    paper_texts = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading papers"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text(file_path)
            text = text.replace('\n', ' ')  # 简单预处理
            paper_texts.append(text)
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

# 指定论文数据集的文件夹路径
folder_path = '../dataset/'

# 读取所有论文文本
paper_texts = read_papers_from_folder(folder_path)

# 获取所有论文的最终嵌入向量
final_embeddings = process_papers(paper_texts)

# 将原始嵌入向量保存到CSV文件中
raw_data_path = './raw_embeddings.csv'
pd.DataFrame(final_embeddings).to_csv(raw_data_path, index=False)

# 使用方差阈值法选择特征
selector = VarianceThreshold(threshold=0.1)
selected_features = selector.fit_transform(final_embeddings)

# 数据标准化
scaler = StandardScaler()
selected_features = scaler.fit_transform(selected_features)

# 自编码器模型定义
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_dim = selected_features.shape[1]
encoding_dim = 50  # 压缩到50维
autoencoder = Autoencoder(input_dim, encoding_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 准备数据
selected_features_tensor = torch.tensor(selected_features, dtype=torch.float32)
dataset = TensorDataset(selected_features_tensor, selected_features_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 增加batch size

# 训练自编码器
num_epochs = 100  # 增加训练次数
for epoch in tqdm(range(num_epochs), desc="Training autoencoder"):
    for data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        inputs, _ = data
        optimizer.zero_grad()
        encoded, decoded = autoencoder(inputs)
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 特征压缩
with torch.no_grad():
    compressed_features, _ = autoencoder(selected_features_tensor)
    compressed_features = compressed_features.numpy()

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(compressed_features)

# 分析每个聚类簇中的特征
cluster_centers = kmeans.cluster_centers_

# 打印聚类结果和代表性的簇中心或特征
print("Cluster centers:")
print(cluster_centers)
print("Cluster assignments:")
for i, cluster in enumerate(clusters):
    print(f"Document {i} belongs to Cluster {cluster}")

# 可视化聚类结果
n_samples = compressed_features.shape[0]
perplexity = min(30, n_samples - 1)  # 确保perplexity小于样本数量

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
tsne_results = tsne.fit_transform(compressed_features)

plt.figure(figsize=(10, 7))
for i in range(4):  # 假设我们有4个聚类
    cluster_points = tsne_results[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.6)

plt.title('t-SNE Visualization of Clustered Documents')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
