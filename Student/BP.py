import os
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义改进后的自定义BP神经网络模型
class PaperScoringModel(nn.Module):
    def __init__(self):
        super(PaperScoringModel, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.squeeze(1)  # 移除大小为1的维度
        print(f"Input shape: {x.shape}")

        x = self.fc1(x)
        print(f"After fc1 shape: {x.shape}")
        x = self.bn1(x)
        print(f"After bn1 shape: {x.shape}")
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        print(f"After fc2 shape: {x.shape}")
        x = self.bn2(x)
        print(f"After bn2 shape: {x.shape}")
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        print(f"After fc3 shape: {x.shape}")
        x = self.bn3(x)
        print(f"After bn3 shape: {x.shape}")
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        print(f"After fc4 shape: {x.shape}")
        x = self.bn4(x)
        print(f"After bn4 shape: {x.shape}")
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        print(f"After fc5 shape: {x.shape}")
        return x

# 初始化自定义BP神经网络模型
paper_scoring_model = PaperScoringModel().to(device)

# 读取保存的词向量文件并构建数据集
def create_dataset_from_embeddings(embedding_dir, score_data):
    X, y = [], []
    for filename in tqdm(os.listdir(embedding_dir), desc="Loading embeddings"):
        if filename.endswith('.npy'):
            embedding = np.load(os.path.join(embedding_dir, filename))
            paper_id = filename.split('_')[1]  # 提取编号
            paper_name = f"C{int(paper_id):03d}.pdf"  # 构建文件名，例如 C001.pdf
            matching_rows = score_data.loc[score_data['Filename'] == paper_name]
            if matching_rows.empty:
                print(f"Warning: No matching score found for {paper_name}")
                continue
            actual_score = matching_rows['Total Score'].values[0]
            X.append(embedding)
            y.append(actual_score)
    return np.array(X), np.array(y)

# 指定保存词向量的文件夹路径
embedding_dir = '../embedding/embeddings'  # 替换为保存词向量的实际文件夹路径

# 读取评分数据
score_data = pd.read_csv('../output/dataset.csv')

# 创建数据集
X, y = create_dataset_from_embeddings(embedding_dir, score_data)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(paper_scoring_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    print(f"-------第 {epoch+1} 轮训练开始-------")
    paper_scoring_model.train()
    for i in range(0, X_train_tensor.size(0), 32):  # 批量大小为32
        batch_X = X_train_tensor[i:i+32]
        batch_y = y_train_tensor[i:i+32]

        optimizer.zero_grad()
        outputs = paper_scoring_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        if (i // 32 + 1) % 10 == 0:
            print(f"训练批次: {i // 32 + 1}, Loss: {loss.item()}")

    # 评估模型
    paper_scoring_model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for i in range(0, X_test_tensor.size(0), 32):  # 批量大小为32
            batch_X = X_test_tensor[i:i+32]
            batch_y = y_test_tensor[i:i+32]
            outputs = paper_scoring_model(batch_X)
            loss = criterion(outputs, batch_y)
            total_test_loss += loss.item()
            total_accuracy += ((outputs - batch_y).abs() < 1).float().mean().item()

    total_test_loss /= (X_test_tensor.size(0) / 32)
    total_accuracy /= (X_test_tensor.size(0) / 32)
    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy}")

    # 保存模型
    torch.save(paper_scoring_model.state_dict(), f"paper_scoring_model_epoch_{epoch+1}.pth")
    print(f"模型已保存: paper_scoring_model_epoch_{epoch+1}.pth")

# 读取保存的词向量文件并进行评分预测
def predict_scores_from_embeddings(embedding_dir):
    scores = []
    for filename in tqdm(os.listdir(embedding_dir), desc="Predicting scores"):
        if filename.endswith('.npy'):
            embedding = np.load(os.path.join(embedding_dir, filename))
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

            with torch.no_grad():
                score = paper_scoring_model(embedding_tensor)
                scores.append((filename, score.item()))
    return scores

# 读取词向量并进行评分预测
predicted_scores = predict_scores_from_embeddings(embedding_dir)

# 创建一个新的DataFrame用于存储预测得分和实际得分
results = []
for filename, predicted_score in predicted_scores:
    paper_id = filename.split('_')[1]  # 提取编号
    paper_name = f"C{int(paper_id):03d}.pdf"  # 构建文件名，例如 C001.pdf
    matching_rows = score_data.loc[score_data['Filename'] == paper_name]
    if matching_rows.empty:
        print(f"Warning: No matching score found for {paper_name}")
        continue
    actual_score = matching_rows['Total Score'].values[0]
    results.append((paper_name, predicted_score, actual_score))

results_df = pd.DataFrame(results, columns=['Filename', 'Predicted Score', 'Actual Score'])

# 打印结果
print(results_df)

# 保存结果到CSV文件
results_df.to_csv('./output/predicted_vs_actual_scores.csv', index=False)
